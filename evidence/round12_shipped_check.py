"""Fresh final-state nonregression checks against immutable earlier evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'evidence'),str(ROOT/'port'),str(ROOT)]
import round10_validate as check
import benchmark_quality as quality

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sweep','heldout','adobe','quality']);a=ap.parse_args()
    assert (ROOT/'port/upright_port.py').read_bytes()==(ROOT/'evidence/round12_before_port.py').read_bytes()
    result={'sweep':check.sweep,'heldout':check.holdout_check,'adobe':check.adobe_check,
            'quality':lambda:quality.run(range(50))}[a.stage]()
    (ROOT/f'evidence/round12_shipped_{a.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SHIPPED CHECK PASSED',a.stage,hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest(),flush=True)
