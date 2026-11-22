#!/usr/bin/env python3
"""Order of loads (by width), vmcnt waits, WMMAs, barriers, branches and conversions in a gfx11 hsaco, run-length
encoded with instruction indices (E108-E110): `LDx4@808 W3@953 BAR@996 MMAx16@1002` reads "four loads at 808, a wait
at 953, barrier at 996, 16 WMMAs from 1002". usage: load_events.py KERNEL.hsaco"""
import os,re,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import disasm  # noqa: E402  (one decoder for every tool: E230)
ev=[]
for i,l in enumerate(disasm(sys.argv[1])[0]):
    op=l.split()[0]; t=None
    m=re.match(r'(buffer|global)_load_(\w+)',op)
    if m: t='L'+m.group(2).replace('b','').replace('dword','32')
    elif op=='s_waitcnt' and 'vmcnt' in l: t='W'+(re.search(r'vmcnt\((\d+)\)',l).group(1) if 'vmcnt(' in l else '')
    elif op.startswith('v_wmma'): t='MMA'
    elif op=='s_barrier': t='BAR'
    elif op.startswith('s_cbranch'): t='BR'
    elif op.startswith('v_cvt') : t='CVT'
    if t:
        if ev and ev[-1][0]==t: ev[-1][1]+=1
        else: ev.append([t,1,i])
print(' '.join(f'{t}x{n}@{i}' if n>1 else f'{t}@{i}' for t,n,i in ev))
