#!/usr/bin/env python3
"""Latency cover per global load (E109): for each buffer/global load in an instruction range of a gfx11 hsaco, the
number of instructions until the s_waitcnt vmcnt that retires it (in-order vmcnt queue). ~2-30 = the load's latency is
exposed (waited right after issue); ~150-500 = it overlaps work (a working prefetch). Check this before timing any
reorder or prefetch: Loom places a pending load's consumer at its earliest legal point (E110).
usage: load_cover.py KERNEL.hsaco FIRST_INSTR LAST_INSTR   (indices as load_events.py prints them)"""
import os,re,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import disasm  # noqa: E402  (one decoder for every tool: E230)
ins=disasm(sys.argv[1])[0]
lo,hi=int(sys.argv[2]),int(sys.argv[3])
q=[]; out=[]
for i,l in enumerate(ins):
    if i<lo or i>hi: continue
    m=re.match(r'(global|buffer)_load_(\w+)\s+(\S+)',l)
    if m: q.append((i,m.group(2),m.group(3).rstrip(',')))
    w=re.search(r'vmcnt\((\d+)\)',l) if l.startswith('s_waitcnt') else None
    if w:
        n=int(w.group(1))
        while len(q)>n:
            j,wd,r=q.pop(0); out.append((j,wd,r,i-j))
for j,wd,r,s in out: print(f'  load@{j:5d} {wd:5s} {r:10s} retired after {s:4d} instr')
