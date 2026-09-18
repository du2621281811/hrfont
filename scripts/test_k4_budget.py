"""Budget boundaries, smooth A extension, and matched B/C schedule."""
import json
from pathlib import Path
from scripts.k4_budget import update_factor,schedule_spec,original_factor

def main():
    root=Path(__file__).resolve().parents[1];a=json.loads((root/'experiments/K4/AUTHORIZATION.json').read_text());anchor=a['budget_extension']['A_anchor']
    assert a['successful_updates']==20000 and abs(original_factor(anchor['step'])-anchor['factor'])<1e-12
    A=[update_factor(s,a,'K4-A') for s in range(anchor['step'],20001)]
    assert abs(A[0]-anchor['factor'])<1e-12 and abs(A[-1]-.1)<1e-12
    assert all(.1<=y<=x for x,y in zip(A,A[1:])) and abs(A[1]-A[0])<1e-6
    B=[update_factor(s,a,'K4-B') for s in range(1,20001)];C=[update_factor(s,a,'K4-C') for s in range(1,20001)]
    assert B==C and B[499]==B[9999]==1 and abs(B[-1]-.1)<1e-12
    assert all(.1<=y<=x for x,y in zip(B[9999:],B[10000:]))
    result=dict(status='PASS',budget=20000,A_continuity=True,A_anchor=anchor,A_next_factor=A[1],B_C_identical=True,schedules={arm:schedule_spec(a,arm) for arm in ['K4-A','K4-C','K4-B']})
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
