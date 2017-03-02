// Copyright 2017 Robert Grosse

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


use std::cmp;
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Instant};

extern crate rayon;

/// Get elapsed time in seconds (for debug/logging purposes)
fn time(now: Instant) -> f64 {
    let d = now.elapsed();
    let t = d.as_secs() as f64;
    let t = t + (d.subsec_nanos() as f64) / 1000000000.0;
    t
}

const CHARS: [u8; 10] = *b"0123456789";

const CBITS: usize = 6;
const P: u64 = 1000003;
const PINV: u64 = 16109806864799210091;

/// Return P^exp
fn pow_p(exp: usize) -> u64 {
    match exp {
        0 => 1,
        1 => P,
        _ => {
            let t = pow_p(exp / 2);
            t.wrapping_mul(t).wrapping_mul(pow_p(exp & 1))
        }
    }
}

/// Return P^-exp
fn pow_pinv(exp: usize) -> u64 {
    match exp {
        0 => 1,
        1 => PINV,
        _ => {
            let t = pow_pinv(exp / 2);
            t.wrapping_mul(t).wrapping_mul(pow_pinv(exp & 1))
        }
    }
}

/// Return mask of lower bits bits
fn to_mask(bits: usize) -> u64 {
    assert!(bits <= 64 && bits > 0);
    !0u64 >> (64 - bits)
}

/// Calculate one step of the Python hash
fn step(x: u64, c: u8) -> u64 {
    x.wrapping_mul(P) ^ c as u64
}

/// Type level boolean parameter
trait Bool {
    fn get() -> bool;
}
struct True;
impl Bool for True {
    fn get() -> bool {
        true
    }
}
struct False;
impl Bool for False {
    fn get() -> bool {
        false
    }
}


/// Trait to choose whether to generate strings or not
trait Output {
    fn new() -> Self;
    fn append(&self, u8) -> Self;
}
impl Output for () {
    fn new() -> Self {
        ()
    }
    fn append(&self, c: u8) -> Self {
        ()
    }
}
impl Output for Vec<u8> {
    fn new() -> Self {
        Vec::new()
    }
    fn append(&self, c: u8) -> Self {
        let mut t = self.clone();
        t.push(c);
        t
    }
}

fn get_strings_all<T: Output>(n: usize, start: u64) -> Vec<(u64, T)> {
    let mut vals = vec![(start, T::new())];
    for _ in 0..n {
        let mut new = Vec::with_capacity(vals.len() * CHARS.len());

        for (x, t) in vals {
            for c in CHARS.iter().cloned() {
                new.push((step(x, c), t.append(c)));
            }
        }
        vals = new;
    }
    vals
}

fn get_strings_vec(n: usize, start: u64, endbits: usize, end: u64) -> Vec<u64> {
    let mut results = Vec::new();
    get_strings(n, start, endbits, end, |p| results.push(p));
    results
}

/// Determines the number of bits to split with, if any
fn calculate_split(n: usize, freebits: usize, endbits: usize) -> Option<(usize, usize, usize)> {
    let rightn = n / 2;
    let leftn = n - rightn;

    let split = ((CHARS.len() as f64).powi(leftn as i32).log2() / 2.0 - 7.0).floor();
    let split = if split < CBITS as f64 {
        CBITS
    } else {
        split as usize
    };
    let split = cmp::min(cmp::max(split, freebits), endbits);

    // Based on testing, it appears that 6 is the optimal cutoff for splitting (for 10 chars)
    if split >= CBITS && n >= 6 {
        Some((leftn, rightn, split))
    } else {
        None
    }
}


/// Returns all pairs a, b with a in left, b in right such that a mask == b mod mask + 1
/// If IsSingle = True, stop at first result (if any)
fn get_equal_pairs<F, IsSingle: Bool>(left_strings: &mut [u64],
                                      right_strings: &mut [u64],
                                      mask: u64,
                                      mut cb: F)
    where F: FnMut(u64, u64)
{
    if IsSingle::get() {
        // no mask in single mode, so we just need to find an exact match in the lists
        assert!(mask == !0u64);
        // We only care about the first match, if any.
        // Try to avoid cost of sorting by just putting one list in a hashset
        let mut rhashes = HashSet::with_capacity(right_strings.len());
        for &mut rx in right_strings {
            rhashes.insert(rx + 0);
        }
        for &mut lx in left_strings {
            if rhashes.contains(&lx) {
                cb(lx, lx);
                break;
            }
        }
        return;
    }

    extern crate pdqsort;
    pdqsort::sort_by_key(left_strings, |&x| x & mask);
    pdqsort::sort_by_key(right_strings, |&x| x & mask);

    // and now to find the matches
    let mut rpos = 0;
    for &lx in left_strings.iter() {
        let lkey = lx & mask;

        while rpos < right_strings.len() && right_strings[rpos] & mask < lkey {
            rpos += 1;
        }
        let current_start = rpos;

        while rpos < right_strings.len() && right_strings[rpos] & mask == lkey {
            let rx = right_strings[rpos];
            cb(lx, rx);
            rpos += 1;
        }
        rpos = current_start;
    }
}

/// Returns all pairs a, b with a in left, b in right such that (a - mid) * phn + b == end mod mask+1
/// pinvhn must be inverse of phn
fn get_axb_pairs<F, IsSingle: Bool>(left_strings: &mut [u64],
                                    right_strings: &mut [u64],
                                    mid: u64,
                                    phn: u64,
                                    pinvhn: u64,
                                    mask: u64,
                                    end: u64,
                                    mut cb: F)
    where F: FnMut((u64, u64), u64)
{
    assert!(end & mask == end);
    assert!(mid <= mask);

    // try to solve (lhs - mid) * phn + rhs = end
    // lhs' = end - (lhs - mid) * phn
    // so goal is lhs' = rhs
    for x in left_strings.iter_mut() {
        *x = end.wrapping_sub(phn.wrapping_mul(*x - mid));
    }

    get_equal_pairs::<_, IsSingle>(left_strings, right_strings, mask, |lx, rx| {
        let x = end.wrapping_sub(lx).wrapping_add(rx);

        // undo transform on lhs in case caller wants access to the matched pair
        // lhs' = end - (lhs - mid) * phn
        // (lhs - mid) * phn = end - lhs'
        // lhs = mid + (end - lhs') * pinvhn
        let lx = mid.wrapping_add(end.wrapping_sub(lx).wrapping_mul(pinvhn));
        cb((lx, rx), x);
    });
}

fn get_pairs_block<F, IsSingle: Bool>(leftn: usize,
                                      rightn: usize,
                                      start: u64,
                                      mid: u64,
                                      splitbits: usize,
                                      mask: u64,
                                      end: u64,
                                      cb: F)
    where F: FnMut((u64, u64), u64)
{
    assert!(mid < (1u64 << splitbits));
    let phn = pow_p(rightn);
    let pinvhn = pow_pinv(rightn);
    let mut left_strings = get_strings_vec(leftn, start, splitbits, mid);
    let mut right_strings = get_strings_vec(rightn, mid, splitbits, end);
    get_axb_pairs::<_, IsSingle>(&mut left_strings,
                                 &mut right_strings,
                                 mid,
                                 phn,
                                 pinvhn,
                                 mask,
                                 end,
                                 cb);
}

fn get_strings_no_split<F, T: Output>(n: usize, start: u64, endbits: usize, end: u64, mut func: F)
    where F: FnMut((u64, T))
{
    let mask = to_mask(endbits);
    let mut func2 = |x, t| if x & mask == end {
        func((x, t))
    };

    if n == 0 {
        func2(start, T::new());
    } else {
        let strs = get_strings_all::<T>(n - 1, start);
        for (x, t) in strs {
            for c in CHARS.iter().cloned() {
                let newx = step(x, c);
                func2(newx, t.append(c));
            }
        }
    }
}

fn get_strings<F>(n: usize, start: u64, endbits: usize, end: u64, mut func: F)
    where F: FnMut(u64)
{
    let mask = to_mask(endbits);
    let end = mask & end;

    if let Some((leftn, rightn, splitbits)) = calculate_split(n, 0, endbits) {
        for mid in 0..(1u64 << splitbits) {
            get_pairs_block::<_, False>(leftn,
                                        rightn,
                                        start,
                                        mid,
                                        splitbits,
                                        mask,
                                        end,
                                        |_, p| func(p));
        }
    } else {
        get_strings_no_split::<_, ()>(n, start, endbits, end, |(x, _)| func(x))
    }
}

fn recover_string_single(n: usize, start: u64, end: u64) -> String {
    let mut result = None;
    if let Some((leftn, rightn, splitbits)) = calculate_split(n, 0, 64) {
        for mid in 0..(1u64 << splitbits) {
            let mut pair = None;
            get_pairs_block::<_, True>(leftn,
                                       rightn,
                                       start,
                                       mid,
                                       splitbits,
                                       !0u64,
                                       end,
                                       |(lx, rx), _| {
                                           // assume for simplicity that lx and rx are unique (i.e not themselves collisions of length n/2)
                                           pair = Some((lx, rx));
                                       });

            if let Some((lx, rx)) = pair {
                let mut left = recover_string_single(leftn, start, lx);
                let right = recover_string_single(rightn, mid, rx);
                left += &right;
                return left;
            }
        }
    } else {
        get_strings_no_split::<_, Vec<u8>>(n, start, 64, end, |(_, s)| {
            result = String::from_utf8(s).ok();
        });
    }
    result.unwrap()
}

fn recover_strings(n: usize, start: u64, freebits: usize, free: u64, end: u64) -> Vec<String> {
    let mut combined_results = Mutex::new(Vec::new());
    let endbits = 64;
    let mask = to_mask(endbits);
    let end = mask & end;

    if let Some((leftn, rightn, splitbits)) = calculate_split(n, freebits, endbits) {
        use rayon::prelude::*;
        // for mid in 0..(1u64 << (splitbits-freebits)) {
        (0..(1u64 << (splitbits-freebits))).into_par_iter().for_each(|mid| {
            let mut results = Vec::new();
            let mid = (mid << freebits) + free;
            let mut pairs = Vec::new();
            get_pairs_block::<_, False>(leftn,
                                     rightn,
                                     start,
                                     mid,
                                     splitbits,
                                     mask,
                                     end,
                                     |(lx, rx), _| {
                // assume for simplicity that lx and rx are unique (i.e not themselves collisions of length n/2)
                pairs.push((lx, rx));
            });
            for (lx, rx) in pairs.into_iter() {
                let mut lstr = recover_string_single(leftn, start, lx);
                let rstr = recover_string_single(rightn, mid, rx);
                lstr += &rstr;
                results.push(lstr);
            }
            combined_results.lock().unwrap().extend(results);
        });
    } else {
        let results = combined_results.get_mut().unwrap();
        get_strings_no_split::<_, Vec<u8>>(n, start, 64, end, |(_, s)| {
            results.push(String::from_utf8(s).unwrap());
        });
    }
    combined_results.into_inner().unwrap()
}

fn main() {
    assert!((CHARS.iter().cloned().min().unwrap() as u64) < (1 << CBITS));
    println!("starting");

    let now = Instant::now();
    let n = 24;
    let target = 0;
    let freebits = 1;
    let free = 0b1;
    assert!(free < (1 << freebits));

    println!("n {} freebits {} free {}",
             {
                 n
             },
             {
                 freebits
             },
             {
                 free
             });
    for i in 0u64..1 {
        let now = Instant::now();
        let strings = recover_strings(n, i, freebits, free, target);
        let t = time(now);

        println!("{{'start':{}, 'count':{}, 'time':{} }}",
                 i,
                 strings.len(),
                 t);
        for s in strings {
            println!("{}", s);
        }
    }
    println!("total time: {}", time(now));
}

/// This function is included for posterity, but I ended up not using it in the final version
fn get_common_lower(start: u64, target_count: u64, bits: usize) -> (usize, usize, u64) {
    let size = 1usize << bits;
    let sparse_len = (bits as f64 / (CHARS.len() as f64).log2()).floor() as usize;

    let mut counts = vec![0u64; size];
    for (x, _) in get_strings_all::<()>(sparse_len, start) {
        counts[x as usize % size] += 1
    }
    let mut len = sparse_len;

    loop {
        let now = Instant::now();
        let mut counts2 = vec![0u64; size];

        for (old, count) in counts.iter().cloned().enumerate() {
            for c in CHARS.iter().cloned() {
                counts2[step(old as u64, c) as usize % size] += count;
            }
        }

        counts = counts2;
        len += 1;

        let max = counts.iter().cloned().enumerate().max_by_key(|&(_, item)| item).unwrap();
        println!("10^{}: {:?} {:?}", len, max, now.elapsed());
        if max.1 >= target_count {
            return (len, max.0, max.1);
        }
    }
}
