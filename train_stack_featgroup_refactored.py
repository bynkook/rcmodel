"""
train_stack_featgroup_refactored.py
date created : 2025.09.02

Purpose
-------
- Train **one StackingRegressor per target** using scikit-learn.
- Keep API compatibility with the existing FastAPI/index.html by **preserving the bundle schema**.
- Provide two training modes:
  (A) Optuna-based hyperparameter search (per-target) with pruning and K-Fold CV.
  (B) Fixed-parameter training loaded from JSON (no search), to allow manual adjustment.

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

Stacking details (how OOF is used)
----------------------------------
- scikit-learn's `StackingRegressor` internally uses `cross_val_predict` on the **training fold** to create
  out-of-fold (OOF) predictions from base estimators, and fits the **final estimator** on those OOF predictions.
- Our KFold during objective function is **external** to StackingRegressor's internal CV:
  - For each outer fold: we fit a fresh pipeline on the **outer train**, and evaluate on the **outer validation**.
  - Inside that pipeline, StackingRegressor builds its **own internal CV** to generate OOF features for the meta-model.
- This two-level scheme yields a reliable CV estimate and enables Optuna pruning by fold.

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

Bundle schema (unchanged for API/index compatibility)
-----------------------------------------------------
The saved joblib bundle is a dict:
{
  "models": { target: sklearn.Pipeline },          # preprocessor + StackingRegressor
  "features_by_target": { target: [feat, ...] },
  "dtypes_by_target": { target: {feat: dtype_str} },
  "targets": [ target, ... ],
  "best_params_by_target": { target: {param: value, ...} },
  "sklearn_version": "<x.y.z>"
}

Notes
-----
- If `USE_OPTUNA=False`, you must provide `input/best_hyperparameters.json` covering **all targets** in `COL_TGTS`.
- Fold-wise MAE is printed for visibility during search; average MAE is minimized.

"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
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

# ========================= 사용자 설정 =========================
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
PASSTHROUGH = False  # 사용자 지정: True/False

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
    final_est = Ridge(alpha=params.get("ridge_alpha", 1.0))

    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_est,
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
    prune_metric: str = "MAE",
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
        pipeline, X_df, y, n_splits=CV_FOLDS, trial=trial, prune_metric="MAE"
    )
    print(f"[Trial {trial.number:03d}] MAE per fold: " + ", ".join(f"{m:.6f}" for m in fold_mae))
    print(f"[Trial {trial.number:03d}] Mean MAE    : {mean_mae:.6f}")
    return mean_mae


def run_optuna_study(X_df: pd.DataFrame, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
    """Optuna 스터디 실행 후 best_params 반환."""
    pruner = PRUNER if USE_PRUNING else None
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda t: optuna_objective(t, X_df, y), n_trials=n_trials, show_progress_bar=True)
    print(f"[Optuna] Best value : {study.best_value:.6f}")
    print(f"[Optuna] Best params: {study.best_params}")
    return study.best_params


def fit_target_with_params(X_df: pd.DataFrame, y: np.ndarray, params: Dict[str, Any]) -> Pipeline:
    """최적 파라미터로 전체 학습 데이터를 사용해 최종 파이프라인 학습."""
    pipe = make_stacking_pipeline(X_df, params)
    pipe.fit(X_df, y)
    return pipe


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

    # 3) Train per target
    models: Dict[str, Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_params_by_tgt: Dict[str, Dict[str, Any]] = {}

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

    for tgt in COL_TGTS:
        feats = feats_by_tgt[tgt]
        X_tr_tgt = df_tr[feats].copy()
        y_tr_tgt = df_tr[tgt].to_numpy(dtype=float)
        dtypes_by_tgt[tgt] = X_tr_tgt.dtypes.astype(str).to_dict()

        if USE_OPTUNA:
            print(f"\n--- Optuna tuning for target: {tgt} ---")
            best_params = run_optuna_study(X_tr_tgt, y_tr_tgt, n_trials=N_TRIALS)
            best_params_by_tgt[tgt] = best_params
        else:
            best_params = best_params_by_tgt[tgt]
            print(f"\n--- Using fixed params for target: {tgt} ---")

        final_model = fit_target_with_params(X_tr_tgt, y_tr_tgt, best_params)
        models[tgt] = final_model

    # 4) Persist best params (if Optuna was used)
    if USE_OPTUNA:
        with open(OUT_PARAMS_JSON, "w", encoding="utf-8") as f:
            json.dump(best_params_by_tgt, f, indent=2)
        print(f"[OK] Best hyperparameters saved to: {OUT_PARAMS_JSON}")

    # 5) Save bundle (API-compatible schema)
    save_bundle(models, feats_by_tgt, dtypes_by_tgt, best_params_by_tgt, OUT_BUNDLE)

    # 6) Report hold-out performance
    generate_and_save_reports(models, feats_by_tgt, df_tr, df_te, OUT_SCORES)


if __name__ == "__main__":
    main()