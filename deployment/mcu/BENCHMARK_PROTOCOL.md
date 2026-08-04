# MCU 실측 벤치마크 지시서 — CEMA SOC 추정기 (latency · memory · ONNX 정확도 변동)

> 이 문서는 새 작업 세션에 그대로 붙여넣거나 "이 파일을 읽고 그대로 수행해"로
> 전달하는 실행 지시서다. 작성 시점(2026-07-24)에 아래 §3의 모든 가중치·데이터 경로가
> 실재함을 확인했다. 작업 기계가 다르면 §3의 폴백 사다리를 따른다.

---

## 0. 임무와 3대 원칙

**임무**: Applied Energy 투고 원고의 배포 절(§4.4.3 Deployment cost)을 뒷받침할 MCU 실측
3종을 산출한다 — NMC/LFP 두 화학의 고정(frozen) 가중치에 대해:

1. **latency** — MCU에서 SOC 추정 1회(50샘플 윈도우 1개, batch 1)의 실행 시간 (사이클 수 + µs)
2. **memory** — Flash 점유(가중치+코드)와 RAM 점유(활성화 + 버퍼), 링커 맵/도구 리포트 근거 포함
3. **ONNX 변환 정확도 변동** — PyTorch 원본 대비 (a) 샘플별 SOC 출력 차이의 최대/평균,
   (b) held-out 전체 레코드에서 재계산한 MAE의 변화(ΔMAE)

**3대 원칙** (위반 시 결과 무효):

- **정밀(precise)**: 보고되는 모든 수치는 스크립트가 산출한 CSV에서만 나온다. 손으로 옮겨
  적거나 화면에서 읽은 값을 기억으로 인용하지 않는다. CSV에는 원시 정밀도를 그대로 저장한다
  (반올림 금지 — 반올림은 이후 원고 세션의 추출기가 담당).
- **재현(reproducible)**: 제3자가 이 지시서 + 산출물의 `README_rerun.md`만 보고 처음부터 같은
  숫자를 다시 만들 수 있어야 한다. 모든 명령·버전·시드·해시·보드 설정을 기록한다. 무작위성이
  필요한 곳은 없다 — 무작위 샘플링 대신 결정적 규칙(등간격 추출)만 쓴다.
- **가중치 불변(frozen)**: 기존 잠금 가중치를 그대로 사용한다. 재학습·재튜닝·체크포인트 선택
  변경 금지. 원본 패키지 파일은 읽기 전용으로 취급하고, 모든 산출물은 새 드롭 디렉터리
  `MCU_BENCH_DROP/`에만 쓴다. 가중치를 끝내 찾지 못한 경우에만 §12의 재학습 폴백을 따른다.

---

## 1. 대상 모델과 우선순위 (사용자 지시 반영)

| 우선순위 | 모델 | 화학 | 체크포인트 수 | ONNX 변환+PC 파리티 | MCU 온보드 측정 |
|---|---|---|---|---|---|
| **Tier A (필수)** | G4 (proposed, 17채널) | NMC + LFP | 3폴드 × 시드 0–2 = 9 + 9 | **필수** | **필수** (latency·memory·온디바이스 파리티) |
| **Tier B (추가, 필수)** | T6 (전압 기억만) | NMC + LFP | 9 + 9 | **필수** | 필수 (최소 latency·memory; 같은 아키텍처이므로 입력 폭 차이 효과 확인) |
| **Tier B (추가, 필수)** | T7 (전류 기억 중심) | LFP만 (패키지에 그것만 있음) | 9 | **필수** | Flash 여유 시 수행, 아니면 사유 기록 |

총 45개 체크포인트. **proposed(G4)는 무조건 수행**하고, 패키지에 동봉된 나머지 모델(T6, T7)도
전부 추가로 돌린다. 파라미터 수 검증값(불일치 시 즉시 중단·보고): **G4 120068 · T6 119044 ·
T7 118532** (LFP 패키지 `inference_benchmark.csv` 기준; NMC도 동일해야 함).

모델 구조(검증용; 소스는 각 패키지의 `code/inference.py`가 유일한 진실):
`_AnchorResidualGRU` — 입력 (B, 50, C) 정규화 윈도우 → ① anchor head: 채널 부분집합
(`config.json`의 `anchor_indices`)에 MLP+sigmoid → ② dynamic: Linear+LayerNorm+SiLU 투영
→ 1층 GRU(hidden 128) → LayerNorm → residual head → 학습형 한계 `residual_limit_param` ×
tanh → ③ (anchor + residual).clamp(0,1)의 **마지막 타임스텝** = SOC **fraction [0,1]**.
%SOC로 보고할 때는 ×100 (모든 파리티·MAE 지표는 %SOC 단위로 저장).

---

## 2. 산출물이 흘러갈 곳 (건드리지는 말 것)

산출 드롭은 이후 원고 리포(`SOC논문_NEW`)의 `data/locked/`로 복사되어
`scripts/extract_numbers.py`의 `mcu_*` 키로 소비된다. **이 세션에서 원고(.tex),
`numbers.yaml`, `data/locked`의 기존 파일을 수정하는 것은 금지** — 원고 반영은 원고 세션이
별도로 수행한다. 이 세션의 임무는 §11 규격의 드롭 디렉터리 완성까지다.

---

## 3. 자산 위치와 폴백 사다리

기준 리포 루트: `C:\Users\yyy25\Documents\문서\SOC논문_NEW` (이하 `<ROOT>`).

### 3.1 고정 가중치 패키지 (2026-07-24 실재 확인됨)

- `<ROOT>\data\locked\inference_pkg_nmc\{G4|T6}\{DST|FUDS|US06}\{0|1|2}\`
- `<ROOT>\data\locked\inference_pkg_lfp\{G4|T6|T7}\{DST|FUDS|US06}\{0|1|2}\`

각 항목 디렉터리: `config.json`(채널·window·EMA τ·eval 규칙), `scaler.json`(mean/std),
`r0_table.json`(온도별 R̂₀), `*_weights.pt`(state_dict + feature_cols), `manifest.json`
(소스 아티팩트 SHA-256), `code/inference.py`(자립 추론 구현 — 소스 리포 import 0개).
패키지 루트: `loader.py`, `manifest.json`, `golden_test_results.csv`(전 항목 <1e-6 통과
이력), LFP에는 `inference_benchmark.csv`(CPU 벤치), NMC에는 `PACKAGING.md`(패키지 규약).

로딩 방법 (둘 다 유효, 택1 후 README에 기록):

```python
import sys; sys.path.insert(0, r"<ROOT>\data\locked")
from inference_pkg_lfp import load_package          # 루트 __init__/loader 경유
pkg = load_package(r"<ROOT>\data\locked\inference_pkg_lfp\G4\DST\0")
soc_fraction = pkg.predict(df_record)               # 처음 49샘플 NaN, 이후 stride-1
```

### 3.2 held-out 레코드 (파리티·MAE 재계산 입력)

- **LFP 24레코드 (로컬 실재 확인됨)**:
  `<ROOT>\data\locked\CEMA_SOC_REVIEW_DEFENSE_C\c1_baselines\prepared_data_ocv_discharge_soc\{-10C..50C}\LFP_{T}C_{DST|FUDS|US06}.csv`
  — 컬럼: `Test_Time(s)`, `Step_Time(s)`, `Current(A)`, `Voltage(V)`, `TempLabel`(예 "25C";
  `predict()`가 이걸 파싱), 참조 라벨 = **`SOC_CC`** (fraction). 폴드 X의 평가 대상 =
  holdout 프로파일 X의 8개 온도 레코드.
- **NMC 9레코드 (이 Windows 기계에는 없음)** — 파일명 규약 `NMC_{0|25|45}C_{DST|FUDS|US06}.csv`.
  확보 사다리(순서대로 시도하고 어느 단계에서 성공했는지 기록):
  1. 디스크 전역 검색: `NMC_*C_*.csv` (실험 리포가 이 기계/랩 리눅스 기계
     `/home/user/바탕화면/DL/NMC_KF_3LOPO_BASELINES` · `/home/lab/바탕화면/DL/` 계열에 존재)
  2. GitHub `KikETs/CEMA_SOC_ESTIMATION`의 데이터 준비 스크립트 + CALCE 공개 원본
     (INR18650-20R, DST/US06/FUDS, 0/25/45 °C)으로 재생성 — 다운로드 URL·일자·파일 해시 기록
  3. 전부 실패 시: G1-NMC(전체 레코드 게이트)를 "미수행 + 사유"로 명시 기록하고,
     NMC 파리티는 §5.3의 골든 윈도우 세트만으로 수행(ΔMAE-NMC는 산출 불가로 보고). 숨기지 말 것.

### 3.3 아카이브 per-sample 예측 (골든 게이트 기준값 + 참조 라벨의 이중 소스)

- NMC(G4, 시드 0–4 15파일):
  `<ROOT>\data\locked\CEMA_MLP_OCVSTART_FULLGRID\nmc_5seed_promotion\nmc_goal_vcorr_it_train_dst_selector_results\*g4_r_*prediction_rows.csv.gz`
- LFP(G4, 시드 0–4 15파일):
  `<ROOT>\data\locked\CEMA_LFP\lfp_confirmatory_minipanel_results\prediction_rows\lfpconfirm_all8_train_all8_test_*g4_holdout*prediction_rows.csv.gz`
- 공통 컬럼: `file_name`(예 `LFP_-10C_DST.csv` — §3.2 레코드 파일명과 정확히 일치),
  `end_index`(윈도우 끝 샘플 인덱스, 49부터), `y_true`, `y_pred`(둘 다 fraction).
  조인 키 = `(file_name, end_index)`.
- T6/T7의 per-sample 아카이브가 같은 디렉터리에 있으면 같은 방식으로 쓰고, 없으면 G1은
  §5.2의 축약 규칙을 따른다.

### 3.4 가중치를 못 찾는 경우

3.1 경로 + 디스크 검색 + 랩 기계 + GitHub 리포에서 전부 실패한 모델에 한해 §12(재학습
폴백)를 발동한다. 탐색한 경로 목록과 실패 로그를 남긴 뒤에만.

---

## 4. 0단계 — 환경 기록 (게이트 G0)

`MCU_BENCH_DROP/environment_host.txt`:
OS/버전, CPU 모델, Python 버전, `pip freeze` 중 numpy·pandas·torch·onnx·onnxruntime
(+양자화/변환 도구), 이 지시서 파일의 SHA-256, 실행 일자.

`MCU_BENCH_DROP/environment_mcu.txt` (보드 도착분 그대로 기록):
보드명, MCU 부품번호, 코어(예 Cortex-M4F/M7/M33), 동작 클럭(MHz), Flash/RAM 총량, FPU 유무,
툴체인(컴파일러+버전), 컴파일 플래그(-O 레벨, FPU ABI), 추론 런타임(이름+정확한 버전),
전원/보드 리비전. **클럭은 측정 펌웨어에서 실제 SystemCoreClock을 출력해 확인한 값**을 쓴다.

`checkpoints_manifest.csv`: 45행 —
`chemistry,feature,fold,seed,weights_file,sha256,param_count,weights_provenance`
(`weights_provenance`는 `frozen_package`; §12 발동 시에만 `retrained`).
param_count는 로드한 state_dict에서 실제 합산해 §1의 검증값과 대조(assert).

---

## 5. 1단계 — 변환 전 골든 게이트 (G1)

목적: "지금 이 기계에서 로드한 가중치+전처리가 논문의 그 추정기"임을 변환 전에 증명.

### 5.1 전체 레코드 골든 (G4 — 필수)

각 화학 × 폴드 × 시드 0–2 (레코드 확보된 화학에서):
1. 해당 폴드 holdout 레코드 전부에 `pkg.predict(df_record)` 실행 (섭동 인자 전부 기본값).
2. §3.3 아카이브에서 같은 (폴드, 시드) 파일을 열어 `(file_name, end_index)`로 조인,
   `max |predict − y_pred|` 계산 (fraction).
3. **판정: 전 항목 max < 1e-6.** 동일 호출 2회가 bit-identical한지도 기록
   (NMC 패키지 규약상 bit-identical이 기대값).
4. 부수 확인: 재계산 slice-unweighted MAE(슬라이스 = holdout×온도별 MAE → 슬라이스 평균 →
   시드 평균)가 3-seed 잠금 기준점과 일치해야 함(±0.002 %SOC):
   **LFP tier-1 — G4 0.535 · T6 0.620 · T7 0.796 / NMC(시드 0–2) — G4 0.328.**
   ⚠️ 5-seed 헤드라인(NMC 0.325 / LFP 0.542)과 비교 금지 — 시드 구성이 다르다.

### 5.2 T6·T7 골든 (아카이브 per-sample이 없는 경우의 축약 규칙)

동일 호출 2회 bit-identical + 파라미터 수 일치 + (LFP) 5.1-4의 MAE 기준점 일치 +
패키지 동봉 `golden_test_results.csv`(패키징 시점 전 항목 <1e-6 통과)를 근거로 인용.
NMC T6는 MAE 기준점이 없으므로 bit-identical + param_count + (레코드 확보 시) 자체 2회
재계산 일치로 대신하고 그 사실을 README에 명시.

### 5.3 골든 윈도우 세트 (이후 모든 단계의 공용 파리티 입력)

체크포인트마다: holdout(또는 가용) 레코드를 `code/inference.py`의 `build_feature_matrix`
→ scaler 정규화 → `unfold(0, 50, 1).permute(0, 2, 1)`로 윈도우화한 뒤, 유효 끝 인덱스
(49..N−1)에서 **결정적 규칙 — 레코드를 파일명 순으로 정렬하고 레코드당 등간격 8개
(LFP: 8레코드×8=64개, NMC: 3레코드×8=24개)** 추출해
`golden_windows/{chem}_{feature}_{fold}_s{seed}.npz`로 저장(float32, 키: `windows`,
`torch_out`, `end_indices`, `file_names`). npz의 SHA-256을 매니페스트에 기록.
결과: `golden_gate_results.csv` —
`chemistry,feature,fold,seed,n_pred_samples,max_abs_delta_vs_archived,bit_identical,mae_anchor_check,status`.

**G1 미통과 항목이 하나라도 있으면 변환으로 넘어가지 말고 사실을 보고하고 사용자 판단을 받는다.**

---

## 6. 2단계 — ONNX 내보내기 규정

- **내보내기 경계 = `_AnchorResidualGRU.forward`**: 입력 `window` (1, 50, C) float32
  **정규화된** 윈도우 → 출력 `soc` (1, 1) fraction. 전처리(R̂₀ 보정, vcorr 시간-EMA
  τ=120 s, 채널 EMA, 스케일러, 윈도우잉)는 경계 밖 — PC에서는 `code/inference.py`를
  그대로 재사용하고, MCU에서는 호스트가 만들어 주입(§8 최소 요건) 또는 C 이식(§9 선택).
- `torch.onnx.export`: **opset 17**, TorchScript 경로(dynamo=False),
  `do_constant_folding=True`, 입력명 `window`, 출력명 `soc`. 두 벌 산출:
  (a) 정적 (1, 50, C) — MCU용 본선, (b) 동적 batch — PC 대량 평가용(선택; 안 만들면
  (a)를 배치 루프로 평가). 모델을 `eval()` 상태로(드롭아웃 비활성) 내보낼 것.
- 검증: `onnx.checker` 통과 + 그래프 op 목록 덤프(예상: Gather, Gemm/MatMul, Sigmoid,
  Mul, LayerNormalization, GRU, Tanh, Clip, Slice류). 목록을 매니페스트에 저장.
- **그래프 수정은 "수학적 항등 재작성"만 허용** (예: SiLU→Sigmoid×Mul 분해,
  index_select→상수 Gather, onnxsim 상수 접기). 적용 항목을 전부 목록화하고, 재작성 후
  골든 윈도우 파리티 <1e-6(fraction)을 다시 확인. 근사 최적화·양자화는 이 단계에서 금지.
- 산출: `onnx_export_manifest.csv` —
  `chemistry,feature,fold,seed,onnx_file,sha256,opset,static_shape,rewrites_applied`.

---

## 7. 3단계 — PC 파리티 + 정확도 변동 (게이트 G2)

- onnxruntime CPU, `SessionOptions`에서 intra/inter op 스레드 = 1, 그래프 최적화 레벨
  기록. 같은 입력 2회 실행 bit-identical 여부 기록.
- 체크포인트별 지표 (`onnx_parity_pc.csv`, 45행, 전부 %SOC 단위):
  `chemistry,feature,fold,seed,n_pred_samples,golden_max_abs_pct,record_max_abs_pct,`
  `record_mean_abs_pct,mae_torch_pct,mae_onnx_pct,delta_mae_pct,bit_identical`
  — `golden_*`는 §5.3 윈도우, `record_*`·`mae_*`는 전체 held-out 레코드의 유효 구간
  (인덱스 49..N−1; NaN 제외; 라벨 = 레코드의 `SOC_CC`, 아카이브 `y_true`와 교차 확인).
- 집계 (`onnx_parity_agg.csv`): 화학×모델별 slice-unweighted 3-seed 집계 —
  `chemistry,feature,agg_mae_torch_pct,agg_mae_onnx_pct,agg_delta_pct,display_3dp_changed`
  (`display_3dp_changed` = 소수 3자리 표기가 바뀌는지 여부 — 논문에 "표시 자릿수 불변"
  주장 가능 여부를 판정하는 컬럼).
- **기대치(초과해도 실측값 그대로 보고)**: fp32에서 `record_max_abs_pct` < 1e-5 %SOC 수준,
  `display_3dp_changed = False`. 기대를 벗어나면 원인(op별 이분 탐색: anchor head만 /
  GRU만 따로 export해 비교)을 조사해 README에 남긴다.

---

## 8. 4단계 — MCU 배포와 온디바이스 파리티 (게이트 G3)

- **런타임 결정표** (보드 벤더에 따라; 선택 경로+도구 버전 전부 기록):
  STM32 → STM32Cube.AI(X-CUBE-AI)의 ONNX 임포트 / NXP → eIQ / 그 외 ARM Cortex-M →
  onnx2tf→TFLite Micro 또는 CMSIS-NN 이식 / 임포터가 op를 거부하면 → ONNX 파일을 유일한
  소스로 삼은 순수 C 구현(가중치를 ONNX에서 추출). **어느 경로든 유효 조건은 하나 —
  ONNX가 소스이고 G3 파리티를 통과하는 것.** 중간 변환 단계가 있으면 각 단계 산출물의
  해시와 그 단계 나름의 PC 파리티를 기록.
- **정밀도 규정**: **fp32가 본선.** fp32 가중치 ≈ 470 KiB(120068×4 B) — 보드 Flash가
  부족하면 그 사실 자체를 결과로 기록하고, fp16 또는 INT8(post-training)을 **별도
  `precision` 라벨로** 측정한다. INT8은 캘리브레이션 세트(= §5.3 골든 윈도우, 명시)·
  양자화 도구/설정 기록, PC 시뮬레이션과 온보드 양쪽에서 §7 지표 전부 재산출. fp32
  행과 절대 섞지 않는다.
- **G3 온디바이스 파리티**: 골든 윈도우 K ≥ 64개를 UART(또는 벤더 브리지)로 주입 →
  float 출력 회수 → PC ONNX 출력과 대조. `mcu_onchip_parity.csv`:
  `model,precision,n_windows,max_abs_diff_pct,mean_abs_diff_pct,transport`.
  기대: fp32 max < 1e-3 %SOC(FMA/누적 순서 차이 허용); INT8은 실측값을 그대로(별도 행).
  주입 스크립트·펌웨어의 수신 프로토콜(바이트 순서, float 인코딩)을 README에 명세.

---

## 9. 5단계 — latency 측정 규정

- 측정 대상: **1추정(윈도우 1개, batch 1) 네트워크 추론**. Cortex-M이면 DWT CYCCNT
  사이클 카운터(권장), 아니면 하드웨어 타이머 — 방법과 타이머 해상도 기록.
- 프로토콜: 워밍업 10회 폐기 → 골든 윈도우를 순환 입력하며 **N ≥ 1000회** → 사이클
  중앙값/P90/최대 기록, 기록된 실측 클럭으로 µs 환산. 측정 중 인터럽트·주변장치 영향을
  최소화한 상태(어떤 것을 껐는지)를 명시.
- 선택 심화: 전처리(스케일러 + EMA 갱신 + 윈도우 버퍼 갱신)를 C로 이식했다면 그
  per-sample 비용을 **별도 행**으로 측정(`includes_preprocessing` 컬럼으로 구분).
  이식하지 않았으면 그 사실만 기록(전처리는 채널당 O(1) 곱셈-덧셈 몇 개 수준).
- 산출: `mcu_latency.csv` —
  `model,precision,clock_mhz,n_reps,cycles_median,cycles_p90,cycles_max,us_median,`
  `us_p90,estimates_per_second,includes_preprocessing`
  — 모델별(G4 필수, T6·T7 추가), 1 Hz 샘플링 대비 여유 배수를 README에 계산.
- 참고용(선택): 같은 호스트 PC에서 onnxruntime 단일 스레드 1추정 µs를 같은 골든
  윈도우로 측정해 별도 파일(`pc_latency_reference.csv`)로 — 논문의 CPU 114 µs/sample과
  단위가 다름(그쪽은 특징 재생성 포함 per-input-sample)을 README에 주석.

## 10. 6단계 — memory 측정 규정

- **Flash**: ① 모델 상수(가중치 blob) 크기 ② 런타임 커널 코드 포함 총 펌웨어 이미지 —
  링커 맵(.map)의 섹션(.text/.rodata) 근거를 인용하고, 벤더 도구 리포트(예: X-CUBE-AI
  analyze)가 있으면 병기. "측정 대상 모델 없이 빌드한 베이스 펌웨어" 크기를 함께 재서
  모델 몫을 차감 산출.
- **RAM**: ① 활성화 아레나(벤더 리포트 또는 계측) ② 입력 윈도우 버퍼 50×C×4 B
  (G4 3400 B / T6·T7은 해당 채널 수로) ③ (전처리 이식 시) EMA 상태 변수
  ④ 스택 최고 사용량(패턴-필 워터마크 등 — 방법 기록). model-only와 firmware-total 분리.
- 산출: `mcu_memory.csv` —
  `model,precision,flash_model_bytes,flash_total_bytes,ram_arena_bytes,`
  `ram_window_buffer_bytes,ram_other_bytes,ram_total_bytes,evidence`
  (`evidence`: mapfile/도구 리포트 파일명 — 해당 리포트 원본도 드롭에 동봉).

---

## 11. 산출물 드롭 규격

```
MCU_BENCH_DROP/
├── environment_host.txt / environment_mcu.txt
├── checkpoints_manifest.csv          (§4)
├── golden_gate_results.csv           (§5)
├── golden_windows/*.npz              (§5.3)
├── onnx/                             (모든 .onnx; 원본 패키지 밖!)
├── onnx_export_manifest.csv          (§6)
├── onnx_parity_pc.csv / onnx_parity_agg.csv   (§7)
├── mcu_onchip_parity.csv             (§8)
├── mcu_latency.csv                   (§9)  [+ pc_latency_reference.csv 선택]
├── mcu_memory.csv (+ 링커 맵·도구 리포트 원본)   (§10)
├── firmware/                         (측정 펌웨어 소스 + 빌드 스크립트)
├── scripts/                          (이 작업에서 작성한 모든 PC 스크립트)
├── README_rerun.md                   (처음부터 재현하는 전체 명령 시퀀스, 실패-재시도 이력 포함)
└── MANIFEST.sha256                   (드롭 내 전 파일의 SHA-256)
```

- CSV 스키마(컬럼명·순서)는 위 정의를 따르고 임의 변경 금지(추가 컬럼은 뒤에만).
- 게이트 실패·미수행 항목은 해당 CSV의 `status`/README에 그대로 남긴다. 실패를 숨긴
  채 latency/memory만 보고하는 것 금지 — 모든 수치 행은 파리티 상태와 함께 읽히게 한다.

---

## 12. 재학습 폴백 (최후 수단 — §3.4 조건 충족 시에만)

1. **발동 기록**: 어떤 모델의 가중치를 어디어디서 찾았고 전부 실패했는지 로그를 먼저 남긴다.
2. **레시피(잠금 — 한 글자도 바꾸지 말 것)**: 코드 = `github.com/KikETs/CEMA_SOC_ESTIMATION`.
   구성 G4(id `paper_g4_all_ema`, 17채널)/T6/T7 · GRU 1층 hidden 128 · dropout 0.06 ·
   anchor-residual head(r_max 학습형 [0, 0.2]) · AdamW lr 8e-4, weight decay 2e-4 ·
   Huber β 0.02 · batch 2048 · 200 epochs · **last-epoch 체크포인트**(선택 없음) ·
   λ_REx 2.0, λ_cond 0.02, λ_anchor 0.1 · window 50, 학습 stride 3, 평가 stride 1 ·
   온도 균등 가중 · seeds 0–2 · LOPO(DST/US06/FUDS 순환 holdout) ·
   NMC 0/25/45 °C · LFP 8온도(−10…50 °C), LFP 라벨 규약 `ocvdischarge_soc0qref`.
3. **합격 기준**: slice-unweighted 3-seed 평균이 §5.1-4의 잠금 기준점과 |Δ| ≤ 0.01,
   per-seed |Δ| ≤ 0.03 (%SOC). GPU 비결정성 때문에 bit 재현은 기대하지 않는다.
4. 미달 시: 진행을 멈추고 사용자에게 보고. 계속하라는 승인을 받으면 모든 산출 CSV에
   `weights_provenance=retrained` + README에 mismatch 표를 강제한다.
5. 재학습 가중치는 원 패키지와 절대 혼합 금지(별도 디렉터리).

---

## 13. 금지 사항 요약

- 원고 리포의 기존 파일(.tex, numbers.yaml, data/locked 기존 내용) 수정 금지.
- 수치 수기 전사 금지 — CSV → (이후 원고 세션의 추출기) 경로만.
- 5-seed 헤드라인(0.325/0.542)과 3-seed 재계산의 직접 비교 금지.
- 게이트를 통과하지 못한 상태로 다음 단계 진행 금지(사용자 확인 예외만).
- 근사 최적화·양자화를 fp32 본선에 섞는 것 금지.
- 갈림길(미지원 op, Flash 부족, 게이트 실패, 레코드 미확보)에서 임의 우회 금지 —
  실측 사실을 제시하고 사용자에게 질문.

## 14. 최종 보고 형식

1. 게이트 통과표: G0/G1/G2/G3 × (화학·모델) — 통과/실패/미수행+사유.
2. 핵심 표: 모델 × 정밀도별 — latency 중앙값(µs, @클럭), Flash(model/total),
   RAM(arena/total), 온디바이스 max|Δ|(%SOC), ΔMAE(%SOC), `display_3dp_changed`.
3. 이상 징후·편차 목록(빈 목록이어도 명시).
4. 드롭 디렉터리 절대 경로 + `MANIFEST.sha256` 파일 수.
5. 원고 반영 제안은 **하지 않는다** — 그건 별도 원고 작업 세션의 몫.
