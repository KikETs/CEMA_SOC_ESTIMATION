#!/usr/bin/env python3
from pathlib import Path
import shutil, datetime

OUT=Path(__file__).resolve().parent/'provenance_hunt_clean'; OUT.mkdir(exist_ok=True)
sources=[
 Path(__file__).resolve().parent/'nmc_goal_vcorr_it_train_dst_selector_results/s3_three_profile_3lopo_ranked_all_jobs.csv',
 Path(__file__).resolve().parent/'nmc_goal_vcorr_it_train_dst_selector_results/s3_3l_bm_gru_g4_r_dst_s012_b2048_e200_test_summary.csv',
 Path(__file__).resolve().parent/'nmc_goal_vcorr_it_train_dst_selector_results/s3_3l_bm_gru_g4_r_fuds_s012_b2048_e200_test_summary.csv',
 Path(__file__).resolve().parent/'nmc_goal_vcorr_it_train_dst_selector_results/s3_3l_bm_gru_g4_r_us06_s012_b2048_e200_test_summary.csv',
 Path('/home/user/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel_results/holdout_temperature_results.csv'),
 Path('/home/user/바탕화면/DL/CEMA_LFP/lfp_confirmatory_minipanel_results/validation_exports/seed_holdout_temperature_mae.csv'),
]
lines=['# Provenance hunt report','','Search only; no experiment was executed or modified.','', '| file | mtime | reproduced keys |','|---|---|---|']
for p in sources:
 if not p.exists(): continue
 shutil.copy2(p,OUT/p.name)
 keys='NMC grid snapshot candidate' if 's3_' in p.name else 'LFP requested T6/G4 holdout-temperature and seed values'
 lines.append(f'| `{p}` | {datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()} | {keys} |')
lines += ['','The LFP values are recoverable from the two listed exports. The NMC exact historical 0.324 family is not jointly reproduced by the current locked G4 GRU summaries; the ranked file is retained as the closest surviving candidate. Conclusion: the exact NMC old snapshot is not currently recoverable as one coherent export (loss by overwrite/re-run is possible, not proven).']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
