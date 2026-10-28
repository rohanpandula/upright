"""Run the full suite against the retained candidate without replacing production."""
from pathlib import Path
import importlib.util
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('upright_port',ROOT/'evidence/round12_candidate_port.py')
up=importlib.util.module_from_spec(spec);sys.modules['upright_port']=up;spec.loader.exec_module(up)
raise SystemExit(pytest.main(['-q',str(ROOT)]))
