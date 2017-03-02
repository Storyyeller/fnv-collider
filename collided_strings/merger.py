# Copyright 2017 Robert Grosse

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import ast
import collections
import heapq
import itertools
import json
import os
import math
import random
import sys

P = 1000003
M = 1<<64
def fnv(s, x=0):
    for c in s:
        x *= P
        x ^= ord(c)
        x %= M
    return x

START_CHARS = 'abcdefghijklmnopqrstuvwxyz_'
ALL_CHARS = START_CHARS + '0123456789'
CMOD = 64
def find_prefix(start, target):
    assert target < CMOD
    q = collections.deque()
    for c in START_CHARS:
        x = start ^ ord(c) << 7
        x = x * P ^ ord(c)
        q.append((c, x % CMOD))

    while q:
        s, x = q.popleft()
        if x == target:
            return s

        for c in ALL_CHARS:
            x2 = x * P ^ ord(c)
            q.append((s+c, x2 % CMOD))






def parse(data):
    lines = data.splitlines()[::-1]
    while not lines[-1].startswith('{'):
        lines.pop()

    while lines and lines[-1].startswith('{'):
        meta = ast.literal_eval(lines.pop())
        i = meta['start']
        strings = [lines.pop() for _ in range(meta['count'])]
        assert all(len(s) == len(strings[0]) for s in strings)
        yield i, strings
    print(lines)

def parsefiles(fnames):
    resultd = [[] for _ in range(64)]
    for fname in fnames:
        for i, strings in parse(open(fname, 'r').read()):
            resultd[i] += strings
    results = sorted((i, sorted(strs)) for i, strs in enumerate(resultd))

    for i, strs in results:
        n = len(strs[0])
        h = fnv(strs[0], i)
        for s in strs:
            assert len(s) == n
            assert fnv(s, i) == h
    return results

fnames = 'collided_all_f000000.txt', 'collided_all_f100000.txt', 'collided_all_f10000.txt', 'collided_0_f1000.txt', 'collided_0_f100.txt', 'collided_0_f10.txt', 'collided_0_f1.txt.txt'
parsed = parsefiles(fnames)
strs0 = parsed[0][1]
assert fnv(strs0[0]) == fnv(strs0[-1])
print(len(strs0), 'colliding strings for 0 of length', len(strs0[0]))


if not os.path.exists('random.txt'):
    with open('random.txt', 'w') as f:
        for _ in range(10000000):
            f.write(random.choice('0123456789'))

with open('random.txt', 'r') as rand:
    randstrs = [rand.read(len(s)) for s in strs0]
    assert rand.read(1)


def writeJ(fname, strings, start):
    with open(fname, 'w') as f:
        for i in range(CMOD):
            prefix = find_prefix(i, start)

            # f.write('.class collide{}\n.super [0]\n'.format(i))
            # for s in strings:
            #     f.write('.const [{}{}] = [0]\n'.format(prefix, s))
            # f.write('.end class\n')

            f.write('.class [{}{}]\n.super [0]\n'.format(prefix, strings[0]))
            f.write('.const [{}{}] = Class collide{}\n'.format(prefix, strings[-1], i))
            for s, s2 in zip(strings, strings[1:]):
                f.write('.const [{}{}] = [{}{}]\n'.format(prefix, s, prefix, s2))
            f.write('.end class\n')

writeJ('collided.j', strs0, 0)
writeJ('random.j', randstrs, 0)

