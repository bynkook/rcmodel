"""
train_stack_featgroup_refactored_V2.py
date created : 2025.09.04

================================================================================
Stacked Regression with Optuna Resume, History Tracking, and Visualization
================================================================================

This script implements per-target stacked regression models with several advanced features:
- Feature-group-aware training for specific targets.
- Optuna integration with persistent storage (SQLite) for history tracking.
- Automatic resume capability to continue interrupted studies.
- A comprehensive visualization function to plot and save all optimization analytics.

Purpose
-------
- Add resume, history, and visualization capabilities while maintaining API compatibility
  with the existing FastAPI/index.html by **preserving the bundle schema**.
- Provide two training modes:
  (A) Optuna-based hyperparameter search with persistent storage and resume functionality.
  (B) Fixed-parameter training loaded from a JSON file.

Core Design
-----------
- All Optuna studies are saved to a single SQLite database (`output/optuna_study.db`),
  enabling persistence across multiple script runs.
- A unique `study_name` is assigned to each target and subgroup, allowing for
  fine-grained tracking and resumption.
- The script checks if a study is already complete before running. If so, it skips
  the optimization to save time and uses the previously found best parameters.
- One `StackingRegressor` is trained for each target in `COL_TGTS`. The final model
  for each target is trained on the full training data using the best hyperparameters
  discovered by Optuna.

Optuna Integration & Resume Feature
-----------------------------------
- **Persistence**: All trial results are saved to the SQLite database defined by `STORAGE_URL`.
  This file acts as a complete historical record of all hyperparameter searches.
- **Automatic Resume**: If the script is stopped and restarted, it will automatically
  load the existing studies from the database. The `study.optimize()` call will
  pick up exactly where it left off.
- **Smart Skipping**: For studies that have already completed the target number of trials (`N_TRIALS`),
  the script will skip the optimization step entirely and directly use the best
-parameters
  from the database to train the final model.

Visualization
-------------
- After all models are trained, a new function `generate_all_visualizations` is called.
- This function loads every study from the SQLite database and generates a series of
  analytical plots using `optuna.visualization`.
- Plots include optimization history, parameter importances, slice plots, and more.
- Each plot is saved as an interactive HTML file in the `output/visualizations/` directory,
  organized into subdirectories named after each study.

Feature-Group Specific Training
-------------------------------
- For selected targets (e.g., 'bd', 'rho', 'phi_mn'), the training set
  is split into feature groups defined by ['f_idx', 'width', 'height'].
- Each group runs an independent Optuna search with its own unique `study_name`.
- **Important:** Although multiple groups are tuned, *only the parameters from the
  single best-performing group* are used to train the final model for that target.
  This ensures API compatibility while leveraging group-specific search.

Bundle Format and API Compatibility
-----------------------------------
The bundle saved to joblib retains the same schema expected by the API:

    {
        "models": { tgt: sklearn.Pipeline },
        "features_by_target": { tgt: [features] },
        "dtypes_by_target": { tgt: {feat: dtype_str} },
        "targets": [tgt, ...],
        "best_params_by_target": { tgt: {…} },
        "sklearn_version": "x.y.z"
    }

Directories and File Outputs
----------------------------
- All outputs are saved in `output/`:
  * `output/optuna_study.db`             : SQLite database for all Optuna studies (enables resume).
  * `output/visualizations/`             : Directory containing all generated HTML plots.
  * `output/stack_bundle_featgroup.joblib`: Bundle for API consumption.
  * `output/stack_scores_featgroup.csv`   : Test-set metrics per target.
  * `output/best_hyperparameters.json`   : Best params by target (from Optuna).

Workflow Summary
----------------
1.  Prepare and split data into train/test sets.
2.  For each target (and subgroup, if applicable):
    a. Create or load a persistent Optuna study from the SQLite database.
    b. Check if the study has already completed its trials. If yes, skip to step (d).
    c. If not complete, run or resume the `study.optimize()` process.
    d. Secure the best parameters from the completed study.
    e. Train the final model for the target/group using the best parameters.
3.  Save the API-compatible bundle, performance scores, and best parameters JSON.
4.  Generate and save all visualization plots from the Optuna database.

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
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_intermediate_values,
)
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

# Stacking passthrough
PASSTHROUGH = True

# I/O directories
OUT_DIR = Path("./output_featgroup")
IN_DIR = Path("./input_featgroup")
VIZ_DIR = OUT_DIR / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IN_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)


# Files (outputs go to output/)
OUT_BUNDLE = OUT_DIR / "stack_bundle_featgroup.joblib"
OUT_SCORES = OUT_DIR / "stack_scores_featgroup.csv"
OUT_PARAMS_JSON = OUT_DIR / "best_hyperparameters_featgroup.json"
# Inputs (when USE_OPTUNA=False, load this)
IN_PARAMS_JSON = IN_DIR / "best_hyperparameters_featgroup.json"
# Optuna Storage (for Resume and History features)
# timeout: wait for file lock
STORAGE_URL = f"sqlite:///{OUT_DIR / 'optuna_study.db'}?timeout=60"
# =============================================================


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


def run_optuna_study(
    X_df: pd.DataFrame, y: np.ndarray,
    study_name: str, storage_url: str, n_trials: int
) -> Dict[str, Any]:
    """
    Creates or loads an Optuna study with persistent storage, checks for completion,
    and runs or resumes the optimization. Returns the best parameters.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1) if USE_PRUNING else None,
    )

    # If the study is new, save the intended number of trials as an attribute.
    if len(study.trials) == 0:
        study.set_user_attr("goal_n_trials", n_trials)
    
    goal_n_trials = study.user_attrs.get("goal_n_trials", n_trials)

    completed_trials = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    print(f"INFO: Study [{study_name}] - Found {completed_trials} completed trials out of {goal_n_trials}.")

    if completed_trials < goal_n_trials:
        print(f"INFO: Study [{study_name}] - Running/resuming optimization...")
        study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(3)
        # reset early-stop callback state
        early_stop_callback.consecutive_small = 0
        early_stop_callback.consecutive_small_diff = 0
        early_stop_callback.prev_best = float("inf")
       
        study.optimize(
            lambda t: optuna_objective(t, X_df, y),
            n_trials=n_trials,
            callbacks=[study_stop_cb, early_stop_callback],
            show_progress_bar=True
        )
    else:
        print(f"INFO: Study [{study_name}] - Study is already complete. Skipping.")

    # This marks early-stopped studies as "complete" for the next run.
    final_completed_trials = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    if final_completed_trials < goal_n_trials:
        print(f"INFO: Study [{study_name}] was stopped early. Updating goal to {final_completed_trials}.")
        study.set_user_attr("goal_n_trials", final_completed_trials)

    print(f"[Optuna] Best value for {study_name}: {study.best_value:.6f}")
    print(f"[Optuna] Best params for {study_name}: {study.best_params}")
    return study.best_params


def train_per_target_with_optuna(
    df: pd.DataFrame,
    targets: List[str],
    feats_by_tgt: Dict[str, List[str]],
    storage_url: str,
    n_trials: int
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """
    Handles the main training loop, including group-specific logic.
    Delegates Optuna study execution to `run_optuna_study`.
    Returns trained models, dtypes, best params, and a list of all study names.
    """
    models: Dict[str, Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_params_by_tgt: Dict[str, Dict[str, Any]] = {}

    for tgt in tqdm(targets, desc="Training models", ncols=100):
        print(f'\n{"="*50}')
        print(f' Start training model for target ---> "{tgt}"')
        print(f'{"="*50}')
        feats = feats_by_tgt[tgt]
        X_df_full = df[feats].copy()
        y_full = df[tgt].to_numpy(dtype=float)
        dtypes_by_tgt[tgt] = X_df_full.dtypes.astype(str).to_dict()

        if tgt in ['bd', 'rho', 'phi_mn']:
            print(f'\n--- Start group training for target "{tgt}", features "{feats}" ---')
            group_best_score, group_best_params = float("inf"), None
                       
            for group_name, sub_df in df.groupby(['f_idx', 'width', 'height']):
                group_id_str = "_".join(map(str, group_name))
                study_name = f"{tgt}_group_{group_id_str}"
                group_name = np.array(group_name).tolist()
                print(f'\n{"="*70}')
                print(f"Tuning for group: {group_name}, target: {tgt}")
                print(f'{"="*70}\n')
                X_group = sub_df[feats].copy()
                y_group = sub_df[tgt].to_numpy(dtype=float)
               
                best_params = run_optuna_study(X_group, y_group, study_name, storage_url, n_trials)
               
                # Evaluate this group's best params on its own data to find the overall best group
                pipe = make_stacking_pipeline(X_group, best_params)
                pipe.fit(X_group, y_group)
                score = mean_absolute_error(y_group, pipe.predict(X_group))
               
                if score < group_best_score:
                    group_best_score = score
                    group_best_params = best_params

            print(f'\n--- Best group for "{tgt}" had MAE {group_best_score:.6f}. Training final model. ---')
            best_params_by_tgt[tgt] = group_best_params
            best_pipe = make_stacking_pipeline(X_df_full, group_best_params)
            best_pipe.fit(X_df_full, y_full)
            models[tgt] = best_pipe
        else:
            study_name = f"{tgt}_study"
            best_params = run_optuna_study(X_df_full, y_full, study_name, storage_url, n_trials)
            best_params_by_tgt[tgt] = best_params           
            best_pipe = make_stacking_pipeline(X_df_full, best_params)
            best_pipe.fit(X_df_full, y_full)
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


def generate_all_visualizations(storage_url: str, out_dir: Path):
    """
    Loads ALL studies directly from storage and generates visualization plots,
    saving them as HTML files. This is robust to script restarts.
    """
    print("\n--- Generating Optuna Visualizations ---")
    
    try:
        # DB에 저장된 모든 study의 요약 정보를 가져옵니다.
        all_studies = optuna.get_all_study_summaries(storage=storage_url)
        if not all_studies:
            print("WARNING: No studies found in storage. Skipping visualization.")
            return
        
        # 요약 정보에서 study 이름만 추출합니다.
        study_names = [s.study_name for s in all_studies]
        print(f"Found {len(study_names)} studies in the database to visualize.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to fetch studies from database at '{storage_url}'. Error: {e}")
        return

    for study_name in tqdm(study_names, desc="Generating Plots", ncols=100):
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
           
            # Create a subdirectory for each study's plots
            study_viz_dir = out_dir / study_name
            study_viz_dir.mkdir(exist_ok=True)

            # 1. Optimization History
            fig = plot_optimization_history(study)
            fig.write_html(study_viz_dir / "optimization_history.html")

            # 2. Parameter Importances
            try:
                fig = plot_param_importances(study)
                fig.write_html(study_viz_dir / "param_importances.html")
            except (ValueError, RuntimeError) as e:
                print(f"Could not plot importances for {study_name}: {e}")

            # 3. Slice Plot
            fig = plot_slice(study)
            fig.write_html(study_viz_dir / "slice.html")

            # 4. Contour Plot for top 2 params
            try:
                # Ensure there are completed trials with at least 2 parameters
                completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and len(t.params) >= 2]
                if completed_trials:
                    top_params = [p for p, _ in optuna.importance.get_param_importances(study).items()][:2]
                    if len(top_params) == 2:
                         fig = plot_contour(study, params=top_params)
                         fig.write_html(study_viz_dir / "contour.html")
            except (ValueError, RuntimeError) as e:
                 print(f"Could not plot contour for {study_name}: {e}")

            # 5. Intermediate Values (Pruning History)
            fig = plot_intermediate_values(study)
            fig.write_html(study_viz_dir / "intermediate_values.html")

        except Exception as e:
            # Changed from KeyError to generic Exception for broader safety
            print(f"ERROR: Failed to generate visualizations for '{study_name}': {e}")
           
    print(f"[OK] All visualizations saved to: {out_dir}")


def main() -> None:
    # 1) Load data
    df = pd.read_csv(DATA_CSV)
    df = df[COL_FEAT]

    # 2) Build features per target and split
    feats_by_tgt = build_feats_by_target(COL_FEAT, COL_TGTS)
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    if USE_OPTUNA:
        # Stream handler of stdout to show the messages (add once)
        _optuna_logger = optuna.logging.get_logger("optuna")
        if not any(isinstance(h, logging.StreamHandler) for h in _optuna_logger.handlers):
            _optuna_logger.addHandler(logging.StreamHandler(sys.stdout))

        print(f"[MODE] Optuna search with persistent storage: {STORAGE_URL}")
        models, dtypes_by_tgt, best_params_by_tgt = train_per_target_with_optuna(
            df_tr, COL_TGTS, feats_by_tgt, STORAGE_URL, n_trials=N_TRIALS
        )
    else:
        print("[MODE] Fixed-parameter training from input JSON")
        if not IN_PARAMS_JSON.exists():
            raise FileNotFoundError(f"Missing parameter file: {IN_PARAMS_JSON}")
        with open(IN_PARAMS_JSON, "r", encoding="utf-8") as f:
            best_params_by_tgt = json.load(f)
        if not all(t in best_params_by_tgt for t in COL_TGTS):
            raise ValueError(f"Input params JSON must include all targets: {COL_TGTS}")

        models, dtypes_by_tgt = {}, {}
        for tgt in tqdm(COL_TGTS, desc="Training fixed-param models", ncols=100):
            feats = feats_by_tgt[tgt]
            X_df, y = df_tr[feats].copy(), df_tr[tgt].to_numpy(dtype=float)
            dtypes_by_tgt[tgt] = X_df.dtypes.astype(str).to_dict()
            pipe = make_stacking_pipeline(X_df, best_params_by_tgt[tgt])
            pipe.fit(X_df, y)
            models[tgt] = pipe

    # 5) Persist best params (if Optuna was used)
    if USE_OPTUNA:
        with open(OUT_PARAMS_JSON, "w", encoding="utf-8") as f:
            json.dump(best_params_by_tgt, f, indent=2)
        print(f"[OK] Best hyperparameters saved to: {OUT_PARAMS_JSON}")

    # 6) Save bundle (API-compatible schema)
    save_bundle(models, feats_by_tgt, dtypes_by_tgt, best_params_by_tgt, OUT_BUNDLE)

    # 7) Report hold-out performance
    generate_and_save_reports(models, feats_by_tgt, df_tr, df_te, OUT_SCORES)

    # 8) Generate and save all visualizations
    if USE_OPTUNA:
        generate_all_visualizations(STORAGE_URL, VIZ_DIR)


if __name__ == "__main__":
    main()