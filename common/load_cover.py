#!/usr/bin/env python3
"""Latency cover per global load (E109): for each buffer/global load in an instruction range of a gfx11 hsaco, the
number of instructions until the s_waitcnt vmcnt that retires it (in-order vmcnt queue). ~2-30 = the load's latency is
exposed (waited right after issue); ~150-500 = it overlaps work (a working prefetch). Check this before timing any
reorder or prefetch: Loom places a pending load's consumer at its earliest legal point (E110).
usage: load_cover.py KERNEL.hsaco FIRST_INSTR LAST_INSTR   (indices as load_events.py prints them)"""
import sys,re,subprocess
d=subprocess.run(['llvm-objdump-21','-d','--mcpu=gfx1151',sys.argv[1]],capture_output=True,text=True).stdout
ins=[x.strip() for x in d.split('\n') if re.match(r'^\s+[a-z]',x)]
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
