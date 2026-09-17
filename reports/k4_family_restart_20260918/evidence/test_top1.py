import sys
sys.path.insert(0,'scripts')
from test_k4_family import regression,T,atomic_json,STORE
original=T.DeltaConfig
T.DeltaConfig=lambda **kw:original(**dict(kw,k_top=1,k_max=1))
checks=regression()
atomic_json(STORE/'control/FAMILY_TOP1_PASSED.json',dict(status='PASS',checks=checks,scope='Force top-k/truncation budget 1 while same-family donors have highest similarity; unrelated donor must still be returned with alpha 1.'))
print('FAMILY_TOP1_PASS',len(checks))
