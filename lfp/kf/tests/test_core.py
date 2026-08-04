import numpy as np
import pandas as pd

from src.hysteresis_model import propagate_hysteresis
from src.ekf import project_psd
from src.lfp_ecm import OCVMap


def test_hysteresis_sign_and_bounds():
    h,_=propagate_hysteresis(0.0,1.0,100.0,1.0,30.0)
    assert -1.0 <= h < 0.0
    h,_=propagate_hysteresis(0.0,-1.0,100.0,1.0,30.0)
    assert 0.0 < h <= 1.0


def test_psd_projection():
    p=project_psd(np.array([[1.0,2.0],[2.0,1.0]]))
    assert np.linalg.eigvalsh(p).min() >= -1e-10


def test_ocv_plateau_slope_not_floored():
    s=np.linspace(0,1,11); rows=[]
    for t in (-10,50):
        for x in s:
            rows.append({'temperature_C':t,'soc':x,'ocv_base_monotonic_V':3.3+0.01*x,
                         'hysteresis_half_monotonic_V':0.01,'ocv_base_raw_V':3.3+0.01*x,
                         'hysteresis_half_raw_V':0.01})
    ocv=OCVMap(pd.DataFrame(rows),'monotonic')
    assert abs(ocv.docv_dsoc(0.5,25)-0.01)<1e-6
