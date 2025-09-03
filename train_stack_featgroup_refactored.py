"""
train_stack_featgroup_refactored.py
date created : 2025.09.03

==========================================================
Stacked Regression with Feature-Group Specific Training
==========================================================

This script implements per-target stacked regression models with
feature-group-aware training, Optuna integration, and API-compatible bundle saving.

Purpose
-------
- Keep API compatibility with the existing FastAPI/index.html by **preserving the bundle schema**.
- Provide two training modes:
  (A) Optuna-based hyperparameter search (per-target) with pruning and K-Fold CV.
  (B) Fixed-parameter training loaded from JSON (no search), to allow manual adjustment.

Core Design
-----------
- One StackingRegressor is trained for each target in COL_TGTS.
- Base estimators are trained on full X, while the final estimator
  is trained on out-of-fold (OOF) predictions via cross_val_predict
  (StackingRegressor built-in), using KFold(CV_FOLDS).
- Global configuration is controlled via constants:
  TEST_SIZE, RANDOM_STATE, CV_FOLDS, USE_OPTUNA, etc.

Stacking details (how OOF is used)
----------------------------------
- scikit-learn's `StackingRegressor` internally uses `cross_val_predict` on the **training fold** to create
  out-of-fold (OOF) predictions from base estimators, and fits the **final estimator** on those OOF predictions.
- Our KFold during objective function is **external** to StackingRegressor's internal CV:
  - For each outer fold: we fit a fresh pipeline on the **outer train**, and evaluate on the **outer validation**.
  - Inside that pipeline, StackingRegressor builds its **own internal CV** to generate OOF features for the meta-model.
- This two-level scheme yields a reliable CV estimate and enables Optuna pruning by fold.

Data split and training flow
----------------------------
- The script consumes a CSV specified by `DATA_CSV`
- Data is deduplicated, then split into **train/test** via `train_test_split(TEST_SIZE, RANDOM_STATE)`.
- For each target `t` in `COL_TGTS`:
  1) Build the **feature list** by removing target itself and mutually excluded features (MUT_EXCL).
  2) Make a Pipeline = [ColumnTransformer(preprocessing) -> StackingRegressor].
  3) If Optuna is ON: run per-target study with **KFold(CV_FOLDS)** and **MAE minimization**.
     - During KFold inside the objective, we **report fold-wise MAE** to Optuna via `trial.report()`.
     - **MedianPruner** may prune underperforming trials early.
  4) If Optuna is OFF: load per-target parameters from `input/best_hyperparameters.json` and fit directly on full train.
- After training, we evaluate on the test split and write a concise **scores CSV**.

Feature-Group Specific Training
-------------------------------
- For selected targets (e.g., 'bd', 'rho', 'phi_mn'), the training set
  is further split into feature groups defined by ['f_idx', 'width', 'height'].
- Each group runs an independent Optuna hyperparameter search
  (with reduced trials if the sample size is small).
- Group-level best parameters and metrics are collected.
- **Important:** although multiple groups are tuned,
  *only the single best-performing group’s model and parameters*
  are saved into the bundle and JSON.  
  This ensures API compatibility while still leveraging group-specific search
  to pick a robust representative model.
- The final chosen params are stored in `best_params_by_tgt[tgt]`.

Optuna Integration
------------------
- USE_OPTUNA=True: runs per-target optimization using KFold CV,
  objective = MAE minimization.
- USE_OPTUNA=False: loads previously saved hyperparameters from JSON
  (in input folder) and trains directly with them.
- Optuna MedianPruner is enabled to stop underperforming trials early.
- early stop enabled when 

Bundle Format and API Compatibility
-----------------------------------
The bundle saved to joblib retains the same schema expected by api.py:

    {
        "models": { tgt: sklearn.Pipeline },
        "features_by_target": { tgt: [features] },
        "dtypes_by_target": { tgt: {feat: dtype_str} },
        "targets": [tgt, ...],
        "best_params_by_target": { tgt: {…} },
        "sklearn_version": "x.y.z"
    }

- Only one model per target is stored (string keys, not tuples).
- Group-specific search is internal; only the chosen best pipeline/params
  are persisted in the bundle and JSON.

Directories and file outputs
----------------------------
- All outputs are saved in `output/`:
  * `output/stack_bundle.joblib`        : bundle for API consumption (unchanged schema)
  * `output/stack_scores.csv`           : test-set metrics per target (MAE, RMSE, R2)
  * `output/best_hyperparameters.json`  : best params by target (from Optuna)
- Inputs for fixed-parameter training are read from `input/`:
  * `input/best_hyperparameters.json`   : user-edited hyperparameters for USE_OPTUNA=False

User-configurable knobs
-----------------------
- `TEST_SIZE`, `RANDOM_STATE`, `CV_FOLDS` for reproducible setup.
- `USE_OPTUNA`: True → run Optuna (per-target); False → load JSON from input and train directly.
- `USE_PRUNING`: True → enable pruning; False → disable.
- `PASSTHROUGH`: Set True/False to control StackingRegressor passthrough behavior.
- The preprocessing (scalers/transforms) is centralized in `build_preprocessor()` and kept stable.

File I/O Structure
------------------
- All outputs (bundle, reports, params JSON) are stored in the ./output directory.
- All user-provided or manually edited parameter JSON files are read
  from the ./input directory.

Workflow Summary
----------------
1. Data preparation: select COL_FEAT only
2. For each target:
    a. If group-aware → tune params per group, select best, fit final model.
    b. If not group-aware → tune params on full data, fit final model.
3. Save reports, params JSON, and API-compatible bundle.
4. main() only orchestrates calls to modular functions.

Notes
-----
- If `USE_OPTUNA=False`, you must provide `input/best_hyperparameters.json` covering **all targets** in `COL_TGTS`.
- Fold-wise MAE is printed for visibility during search; average MAE is minimized.
- The feature-group logic improves robustness for targets whose
  distribution shifts strongly by geometry/material grouping.
- Bundle schema is fixed to ensure seamless integration with
  existing api.py and index.html frontends.

"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm.auto import tqdm

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.trial import TrialState  # TrialState 열거형
from sklearn import __version__ as sk_version
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    FunctionTransformer,
    OneHotEncoder,
)
from sklearnex import patch_sklearn
patch_sklearn()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", category=UserWarning)     # 사용자 코드에서 발생하는 경고 무시

# ========================= 전역 설정 =========================
# Data / columns
DATA_CSV = "batch_analysis_rect_result.csv"
COL_FEAT = ["f_idx", "width", "height", "Sm", "bd", "rho", "phi_mn"]
COL_TGTS = ["Sm", "bd", "rho", "phi_mn"]
MUT_EXCL = {"as_provided": ["phi_mn"], "phi_mn": ["as_provided"]}

# Reproducibility
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Optuna
USE_OPTUNA = True
# USE_OPTUNA = False
N_TRIALS = 30
USE_PRUNING = True
PRUNER = MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)

# Stacking passthrough
PASSTHROUGH = False

# I/O directories
OUT_DIR = Path("./output")
IN_DIR = Path("./input")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IN_DIR.mkdir(parents=True, exist_ok=True)

# Files (outputs go to output/)
OUT_BUNDLE = OUT_DIR / "stack_bundle_featgroup.joblib"
OUT_SCORES = OUT_DIR / "stack_scores_featgroup.csv"
OUT_PARAMS_JSON = OUT_DIR / "best_hyperparameters_featgroup.json"

# Inputs (when USE_OPTUNA=False, load this)
IN_PARAMS_JSON = IN_DIR / "best_hyperparameters_featgroup.json"
# =============================================================

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

class StopWhenTrialKeepBeingPrunedCallback:
    # https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
    # Stop optimization after some trials are pruned in a row
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()


def early_stop_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    """
    Optuna 콜백: 조기 종료 조건
    - 3회 연속으로 trial의 MAE 값이 1e-5 이하인 경우 스터디를 종료.
    - 3회 연속으로 trial의 MAE 값이 이전 best MAE와의 차이가 1e-5 이하인 경우 스터디를 종료.
    (이 콜백은 기존 pruning 조건과 독립적으로 동작하며, trial이 pruning되지 않아도 이 조건을 만족하면 종료함)
    
    Note: consecutive_small counter only counts "successful" trials.
    If a trial is pruned, it does NOT reset or increment the counter.
    """
    # 최초 호출 시 상태 변수 초기화
    if not hasattr(early_stop_callback, "consecutive_small"):
        early_stop_callback.consecutive_small = 0
        early_stop_callback.consecutive_small_diff = 0
        early_stop_callback.prev_best = float("inf")
    # 완료된 trial만 검사 (pruned/failed trial은 조건 연속성 초기화)
    if trial.state != TrialState.COMPLETE or trial.value is None:
        early_stop_callback.consecutive_small = 0
        early_stop_callback.consecutive_small_diff = 0
        return
    # 조기 종료 조건 확인
    value = trial.value
    if value <= 1e-5:
        early_stop_callback.consecutive_small += 1
    else:
        early_stop_callback.consecutive_small = 0
    if abs(value - early_stop_callback.prev_best) <= 1e-5:
        early_stop_callback.consecutive_small_diff += 1
    else:
        early_stop_callback.consecutive_small_diff = 0
    # 조건 만족 시 스터디 중단
    if early_stop_callback.consecutive_small >= 3 or early_stop_callback.consecutive_small_diff >= 3:
        study.stop()
    # 다음 trial을 위한 이전 best MAE 업데이트
    if value < early_stop_callback.prev_best:
        early_stop_callback.prev_best = value


def build_feats_by_target(all_feats: List[str], targets: List[str]) -> Dict[str, List[str]]:
    """
    타깃별 학습 피처 목록 생성:
    - 타깃 자신 제외
    - MUT_EXCL에 정의된 상호 배타 피처 제외
    """
    feats_map: Dict[str, List[str]] = {}
    for tgt in targets:
        ban = set(MUT_EXCL.get(tgt, []))
        feats = [c for c in all_feats if c != tgt and c not in ban]
        feats_map[tgt] = feats
    return feats_map


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    전처리 파이프라인(ColumnTransformer) 구성.
    - numeric, log, exp: MinMax
    - categorical: OneHot
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log_cols = [c for c in num_cols if c.lower() in ["phi_mn"]]
    exp_cols = [c for c in num_cols if c.lower() in ["rho"]]
    other_num_cols = [c for c in num_cols if c not in log_cols and c not in exp_cols]

    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", MinMaxScaler()),
        ]
    )
    log_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("sc", MinMaxScaler()),
        ]
    )
    exp_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("exp", FunctionTransformer(np.exp, validate=False)),
            ("sc", MinMaxScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers_: List[Tuple[str, Pipeline, List[str]]] = []
    if other_num_cols:
        transformers_.append(("num", num_pipe, other_num_cols))
    if log_cols:
        transformers_.append(("log", log_pipe, log_cols))
    if exp_cols:
        transformers_.append(("exp", exp_pipe, exp_cols))
    if cat_cols:
        transformers_.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers_, remainder="passthrough")


def make_stacking_pipeline(X: pd.DataFrame, params: Dict[str, Any]) -> Pipeline:
    """
    하이퍼파라미터 dict를 받아 스태킹 파이프라인을 구성한다.
    - Base: RF + GBR
    - Final: Ridge
    - PASSTHROUGH 전역값을 그대로 적용
    """
    pre = build_preprocessor(X)

    rf = RandomForestRegressor(
        n_estimators=params.get("rf_n_estimators", 300),
        max_depth=params.get("rf_max_depth", None),
        min_samples_leaf=params.get("rf_min_samples_leaf", 2),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gbr = GradientBoostingRegressor(
        n_estimators=params.get("gbr_n_estimators", 300),
        learning_rate=params.get("gbr_learning_rate", 0.1),
        max_depth=params.get("gbr_max_depth", 3),
        random_state=RANDOM_STATE,
    )

    base_estimators = [("rf", rf), ("gbr", gbr)]
    final_estimator = Ridge(alpha=params.get("ridge_alpha", 1.0))

    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=PASSTHROUGH,
    )
    pipe = Pipeline([("pre", pre), ("stack", stack)])
    return pipe


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def kfold_mae_for_pipeline(
    pipe: Pipeline,
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = CV_FOLDS,
    trial: Optional[optuna.trial.Trial] = None,
    ) -> Tuple[float, List[float]]:
    """
    외부 KFold로 파이프라인 MAE를 측정.
    - 각 fold에서 파이프라인(clone) 학습 → 검증 예측 → MAE
    - 폴드 종료 시 누적 평균 MAE를 trial.report()로 보고하여 MedianPruner가 중도 중단 가능.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_df), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        p = clone(pipe)
        p.fit(X_tr, y_tr)
        y_hat = p.predict(X_va)
        mae = mean_absolute_error(y_va, y_hat)
        maes.append(mae)

        if trial is not None:
            running = float(np.mean(maes))
            trial.report(running, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(maes)), maes


def optuna_objective(trial: optuna.Trial, X_df: pd.DataFrame, y: np.ndarray) -> float:
    """
    Optuna 목적함수: MAE 최소화.
    - 탐색 파라미터: RF/GBR/Ridge 핵심 하이퍼.
    - 폴드별 MAE와 평균 MAE를 콘솔에 출력.
    """
    params = {
        "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 500, step=20),
        "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 20),
        "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 2, 5),
        "gbr_n_estimators": trial.suggest_int("gbr_n_estimators", 50, 500, step=20),
        "gbr_learning_rate": trial.suggest_float("gbr_learning_rate", 1e-2, 0.2, log=True),
        "gbr_max_depth": trial.suggest_int("gbr_max_depth", 2, 6),
        "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-2, 100, log=True),
    }
    pipeline = make_stacking_pipeline(X_df, params)
    mean_mae, fold_mae = kfold_mae_for_pipeline(
        pipeline, X_df, y, n_splits=CV_FOLDS, trial=trial,
    )
    print(f"[Trial {trial.number:03d}] MAE per fold: " + ", ".join(f"{m:.6f}" for m in fold_mae))
    print(f"[Trial {trial.number:03d}] Mean MAE    : {mean_mae:.6f}")
    return mean_mae


def run_optuna_study(X_df: pd.DataFrame, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
    """Optuna 스터디 실행 후 best_params 반환."""
    
    study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(3)
    pruner = PRUNER if USE_PRUNING else None

    study = optuna.create_study(direction="minimize", pruner=pruner)

    # 초기화: 조기 종료 콜백의 연속 횟수 상태
    early_stop_callback.consecutive_small = 0
    early_stop_callback.consecutive_small_diff = 0
    early_stop_callback.prev_best = float("inf")

    study.optimize(
        lambda t: optuna_objective(t, X_df, y),
        n_trials=n_trials,
        callbacks=[study_stop_cb, early_stop_callback],
        show_progress_bar=True
        )
    print(f"[Optuna] Best value : {study.best_value:.6f}")
    print(f"[Optuna] Best params: {study.best_params}")
    return study.best_params


def train_per_target_with_optuna(
    df: pd.DataFrame,
    targets: List[str],
    feats_by_tgt: Dict[str, List[str]],
    best_params_by_tgt: Dict[str, Dict[str, Any]],
    n_trials: int = 30
) -> Tuple[Dict[str, Pipeline],
           Dict[str, Dict[str, str]],
           Dict[str, Dict[str, Any]]]:

    """타겟별 Optuna → 최적 파라미터로 full train 파이프라인 학습"""
    models: Dict[str, Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}

    for tgt in tqdm(targets, desc="Training models", ncols=100):
        print(f'\n--- Start training model for target "{tgt}" ---')
        feats = feats_by_tgt[tgt]
        X_df = df[feats].copy()
        y = df[tgt].to_numpy(dtype=float)
        dtypes_by_tgt[tgt] = X_df.dtypes.astype(str).to_dict()

        if USE_OPTUNA:
            print(f"\n--- Optuna tuning for target: {tgt} ---")
            best_params_by_tgt: Dict[str, Dict[str, Any]] = {}

            ##### 그룹화 적용 (bd, rho, phi_mn에만) ####
            if tgt in ['bd', 'rho', 'phi_mn']:
                print(f'\n--- Start group training for target "{tgt}", features "{feats}" ---')
                # group 별 탐색 수행, 그러나 최종 번들에는 대표 모델 1개만 저장
                group_best_score, group_best_pipe, group_best_params = float("inf"), None, None
                for group_name, sub_df in df.groupby(['f_idx', 'width', 'height']):
                    X_group = sub_df[feats].copy()
                    y_group = sub_df[tgt].to_numpy(dtype=float)
                    print(f"  Tuning for group: {group_name}")
                    best_params = run_optuna_study(X_group, y_group, n_trials=n_trials if len(sub_df) > 25 else 10)

                    # 최적 파라미터로 full fit
                    best_pipe = make_stacking_pipeline(X_group, best_params)
                    best_pipe.fit(X_group, y_group)
                    y_hat = best_pipe.predict(X_group)
                    score = mean_absolute_error(y_group, y_hat)
                    if score < group_best_score:
                        group_best_score, group_best_pipe, group_best_params = score, best_pipe, best_params

                # group best 만 저장
                models[tgt] = group_best_pipe
                best_params_by_tgt[tgt] = group_best_params
            else:
                # Optuna 탐색
                best_params = run_optuna_study(X_df, y, n_trials=n_trials)
                best_pipe = make_stacking_pipeline(X_df, best_params)
                best_pipe.fit(X_df, y)
                models[tgt] = best_pipe
                best_params_by_tgt[tgt] = best_params
        else:
            best_params = best_params_by_tgt[tgt]
            print(f"\n--- Using fixed params for target: {tgt} ---")
            best_pipe = make_stacking_pipeline(X_df, best_params)
            best_pipe.fit(X_df, y)
            models[tgt] = best_pipe

    return models, dtypes_by_tgt, best_params_by_tgt


def generate_and_save_reports(
    models: Dict[str, Pipeline],
    feats_by_tgt: Dict[str, List[str]],
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    out_scores_csv: Path,
    ) -> None:
    """Train/Test 평가 리포트 생성 후 CSV 저장."""
    rows: List[Dict[str, Any]] = []
    print("\n--- Final Model Performance (hold-out test) ---")
    for tgt, model in models.items():
        feats = feats_by_tgt[tgt]
        X_tr, y_tr = df_tr[feats], df_tr[tgt].to_numpy(dtype=float)
        X_te, y_te = df_te[feats], df_te[tgt].to_numpy(dtype=float)

        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        tr_metrics = calculate_regression_metrics(y_tr, y_tr_pred)
        te_metrics = calculate_regression_metrics(y_te, y_te_pred)

        print(
            f"[{tgt:10s}] Train MAE={tr_metrics['MAE']:.6f} RMSE={tr_metrics['RMSE']:.6f} R2={tr_metrics['R2']:.6f}\n"
            f"[{tgt:10s}] Test  MAE={te_metrics['MAE']:.6f} RMSE={te_metrics['RMSE']:.6f} R2={te_metrics['R2']:.6f}"
        )

        rows.append({"target": tgt, "dataset": "train", **tr_metrics})
        rows.append({"target": tgt, "dataset": "test", **te_metrics})

    pd.DataFrame(rows)[["target", "dataset", "MAE", "RMSE", "R2"]].to_csv(out_scores_csv, index=False)
    print(f"[OK] Scores saved to: {out_scores_csv}")


def save_bundle(
    models: Dict[str, Pipeline],
    feats_by_tgt: Dict[str, List[str]],
    dtypes_by_tgt: Dict[str, Dict[str, str]],
    best_params_by_tgt: Dict[str, Dict[str, Any]],
    out_path: Path,
    ) -> None:
    """API 호환 번들을 저장."""
    bundle = {
        "models": models,
        "features_by_target": feats_by_tgt,
        "dtypes_by_target": dtypes_by_tgt,
        "targets": list(models.keys()),
        "best_params_by_target": best_params_by_tgt,
        "sklearn_version": sk_version,
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"[OK] Bundle saved to: {out_path}")


def main() -> None:
    # 1) Load data
    df = pd.read_csv(DATA_CSV)
    df = df[COL_FEAT]

    # 2) Build features per target and split
    feats_by_tgt = build_feats_by_target(COL_FEAT, COL_TGTS)
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3) load params when USE_OPTUNA = False
    if USE_OPTUNA:
        print("[MODE] Optuna search per target")
    else:
        print("[MODE] Fixed-parameter training from input JSON")
        if not IN_PARAMS_JSON.exists():
            raise FileNotFoundError(f"Missing parameter file: {IN_PARAMS_JSON}")

        with open(IN_PARAMS_JSON, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # loaded must contain all targets
        if not all(t in loaded for t in COL_TGTS):
            raise ValueError(f"Input params JSON must include all targets: {COL_TGTS}")
        best_params_by_tgt = loaded

    # 4) Train per target
    models: Dict[str, Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_params_by_tgt: Dict[str, Dict[str, Any]] = {}

    models, dtypes_by_tgt, best_params_by_tgt = train_per_target_with_optuna(
        df_tr, COL_TGTS, feats_by_tgt, best_params_by_tgt, n_trials=30)

    # 5) Persist best params (if Optuna was used)
    if USE_OPTUNA:
        with open(OUT_PARAMS_JSON, "w", encoding="utf-8") as f:
            json.dump(best_params_by_tgt, f, indent=2)
        print(f"[OK] Best hyperparameters saved to: {OUT_PARAMS_JSON}")

    # 6) Save bundle (API-compatible schema)
    save_bundle(models, feats_by_tgt, dtypes_by_tgt, best_params_by_tgt, OUT_BUNDLE)

    # 7) Report hold-out performance
    generate_and_save_reports(models, feats_by_tgt, df_tr, df_te, OUT_SCORES)


if __name__ == "__main__":
    main()