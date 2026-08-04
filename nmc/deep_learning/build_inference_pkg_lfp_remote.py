#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

ROOT = Path('/home/lab/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel')
sys.path.insert(0, str(ROOT))
import build_inference_pkg_nmc as b

b.HERE = ROOT
b.SOURCE = ROOT
b.RAW_ROOT = ROOT / 'prepared_data_ocv_discharge_soc'
b.RESULTS = ROOT / 'runs/all8_train_all8_test/tier1/nmc_goal_vcorr_it_train_dst_selector_results'
b.OUTPUT = ROOT / 'inference_pkg_lfp'
b.TEMPLATE = ROOT / 'lfp_frozen_inference.py'
b.TEMPERATURES = (-10, 0, 10, 20, 25, 30, 40, 50)
b.FOLDS = ('DST', 'FUDS', 'US06')
b.PROFILES = b.FOLDS
b.FEATURES = {
    'T6': {'source_name': 'paper_t6_voltage_ema_all', 'prefix': 'lfpconfirm_all8_train_all8_test_tier1_gru_residual_t6_holdout{fold}_s012_b2048_e200'},
    'T7': {'source_name': 'paper_t7_current_abs_ema_all', 'prefix': 'lfpconfirm_all8_train_all8_test_tier1_gru_residual_t7_holdout{fold}_s012_b2048_e200'},
    'G4': {'source_name': 'paper_g4_all_ema', 'prefix': 'lfpconfirm_all8_train_all8_test_tier1_gru_residual_g4_holdout{fold}_s012_b2048_e200'},
}

def paths_for_fold(fold):
    train = tuple(p for p in b.PROFILES if p != fold)
    train_paths = [b.RAW_ROOT / f'{t}C' / f'LFP_{t}C_{p}.csv' for t in b.TEMPERATURES for p in train]
    test_paths = [b.RAW_ROOT / f'{t}C' / f'LFP_{t}C_{fold}.csv' for t in b.TEMPERATURES]
    return train_paths, test_paths

b.paths_for_fold = paths_for_fold

if __name__ == '__main__':
    b.build()
