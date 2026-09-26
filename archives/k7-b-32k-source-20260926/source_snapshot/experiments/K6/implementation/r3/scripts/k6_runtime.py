import json
import k5_runtime as K5
from k5_runtime import *
STORE=CODE.parent.parent/'data1/hrfont_k6_20260920' # replaced below with explicit path
from pathlib import Path
STORE=Path('/root/data1/hrfont_k6_20260920')
ASSETS=CODE/'experiments/K6'
def model_for(arm,device,output,evaluation=False):
    assert arm in ['K6-A','K6-B','K6-B-RSI-NOOFFSETLOSS','K7-B']
    model,args=K5.model_for('K5-B',device,output,evaluation=evaluation)
    model.arm=arm
    return model,args
def code_identity():
    meta=json.loads((CODE/'K7_CODE_IDENTITY.json').read_text())
    for p,h in meta['files'].items():assert sha256_file(CODE/p)==h,p
    return meta
