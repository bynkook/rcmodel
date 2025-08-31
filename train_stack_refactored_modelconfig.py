"""
train_stack_refactored.py — DOCSTRING

Purpose
- Train target-wise stacking regressors with scikit-learn’s StackingRegressor.
- Keep API compatibility with existing FastAPI service by persisting the same bundle schema.
- Enable repeatable experiments via TEST_ID-based model presets and Optuna search.
- Allow quick retrain from previously discovered hyperparameters without running Optuna.

Scope and Data
- Features and targets are provided by user constants COL_FEAT / COL_TGTS.
- Leakage guard:
  - Each target’s feature set excludes the target itself.
  - MUT_EXCL enforces mutual exclusion rules (e.g., {'phi_mn': ['as_provided'], ...}).
- Preprocessing is defined in build_preprocessor(X) and is considered immutable per project policy
  (do not alter its internals in this file; keep it consistent with the web API expectations).

Core Design
- Per-target training with one StackingRegressor per target.
- Base estimators fit on full X; final estimator is trained by StackingRegressor using OOF
  (cross_val_predict under the hood) with KFold(n_splits=CV_FOLDS, shuffle=True, random_state).
- Global, reproducible configuration via constants:
  TEST_SIZE, RANDOM_STATE, CV_FOLDS.

TEST_ID and Model Presets
- TEST_ID (global) selects which preset to run.
- MODEL_CONFIGS: dictionary keyed by TEST_ID; each entry defines:
  - base_models: list of (name, estimator_class, fixed_params)
  - final_model: (name, estimator_class, fixed_params)
  - param_space: Optuna search space keyed by '{alias}_{param}'
- Changing TEST_ID switches the entire model combination and its search space without editing code.

Optuna Integration
- USE_OPTUNA=True: runs per-target optimization on the selected preset.
  - Objective: minimize MAE via KFold CV (CV_FOLDS).
  - Each trial prints fold-wise MAE and the mean MAE.
  - Pruning and early stopping are intentionally disabled (project requirement).
- USE_OPTUNA=False: loads saved best hyperparameters (JSON) and trains directly (no search).
- N_TRIALS controls the number of Optuna trials when enabled.

Hyperparameter Persistence (per TEST_ID)
- HYPERPARAMS_FILE = f"best_hyperparameters_{TEST_ID}.json"
  - On USE_OPTUNA=True: best_params_by_target is saved to this JSON.
  - On USE_OPTUNA=False: the same file is loaded to skip search and retrain quickly.

Outputs (per TEST_ID)
- OUT_BUNDLE  = f"stack_bundle_{TEST_ID}.joblib"
- OUT_SCORES  = f"stack_scores_{TEST_ID}.csv"
- HYPERPARAMS = f"best_hyperparameters_{TEST_ID}.json"
- Scores CSV contains per-target metrics on train/test splits: MAE, RMSE, R2.

Bundle Schema (API Compatibility)
- The persisted bundle strictly matches the API expectation:
  {
    "models":            { target: sklearn.Pipeline },      # preprocessor + StackingRegressor
    "features_by_target":{ target: [feature, ...] },
    "dtypes_by_target":  { target: {feature: dtype_str} },
    "targets":           [ target, ... ],
    "best_params_by_target": { target: {param: value, ...} },
    "sklearn_version":   "<x.y.z>"
  }
- api.py can call model.predict(X_df) per target without adjustments.

Training/Evaluation Flow
1) Read CSV and select required columns; drop duplicates.
2) Build features_by_target with leakage and MUT_EXCL rules.
3) Split into train/test by TEST_SIZE.
4) For each target:
   - If USE_OPTUNA=True: run study to get best_params.
   - Else: load best_params from JSON.
   - Build Stacking pipeline from preset + params and fit on full train.
5) Save bundle (joblib), write performance report (CSV) on held-out test.

Extensibility
- Add new presets by inserting entries in MODEL_CONFIGS:
  - Swap base estimators (e.g., RF, GBR, ExtraTrees, HGBR, LightGBM/CatBoost if allowed).
  - Change final estimator (e.g., Ridge/ElasticNet/RF).
  - Adjust param_space keys using the '{alias}_{param}' naming convention.
- Keep preprocessing stable for API alignment; evolve only the estimator side via TEST_ID.

Usage Examples
- Run Optuna search and save best params:
  - Set TEST_ID, USE_OPTUNA=True, then execute the script.
  - Inspect console for per-trial KFold MAE prints and final best parameters.
- Retrain quickly from saved params:
  - Set the same TEST_ID, USE_OPTUNA=False.
  - The script loads best_hyperparameters_{TEST_ID}.json and trains final models.

Notes
- Ensure COL_FEAT, COL_TGTS, MUT_EXCL align with your dataset columns.
- Changing TEST_ID changes both model composition and output filenames, isolating experiments.
- Keep RANDOM_STATE fixed to compare runs fairly.

[사용법]
1. 하이퍼파라미터 탐색 시: USE_OPTUNA = True 설정 후 실행
   - 탐색 완료 후 best_hyperparameters.json 파일이 생성됨.
2. 저장된 파라미터로 학습 시: USE_OPTUNA = False 설정 후 실행
   - best_hyperparameters.json 파일을 읽어와 고정된 파라미터로 빠르게 학습.
"""
from __future__ import annotations
import os
import sys
import json
import warnings
import joblib
import numpy as np
import optuna
import pandas as pd
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn import __version__ as sk_version
from sklearnex import patch_sklearn
patch_sklearn()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", category=UserWarning)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================= 사용자 설정 (CONFIGURATION) =========================
# --- 데이터 및 모델 관련 설정 ---
DATA_CSV   = "batch_analysis_rect_result.csv"
COL_FEAT   = ['f_idx', 'width', 'height', 'Sm', 'bd', 'rho', 'phi_mn']
COL_TGTS   = ['Sm', 'bd', 'rho', 'phi_mn']
MUT_EXCL   = {'as_provided': ['phi_mn'], 'phi_mn': ['as_provided']} # 특정 타겟 학습 시 제외할 피처

# --- 학습 프로세스 제어 ---
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5  # 교차 검증 폴드 수

# --- Optuna 하이퍼파라미터 탐색 제어 ---
USE_OPTUNA = True # True: Optuna 탐색 실행 | False: 저장된 파라미터 사용
N_TRIALS   = 30   # Optuna 탐색 횟수

# --- 실험/모델 구성 설정 ---
TEST_ID = 1  # (요구사항 1) 실험 식별자 (사용자 설정)
# 각 TEST_ID별 모델 조합과 하이퍼파라미터 탐색 공간 정의 (요구사항 2)
# - 새로운 모델 조합을 쉽게 추가/변경할 수 있도록 딕셔너리 구조 사용 (요구사항 5)
MODEL_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "base_models": [
            ("rf", RandomForestRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1}),
            ("gbr", GradientBoostingRegressor, {"random_state": RANDOM_STATE})
        ],
        "final_model": ("ridge", Ridge, {}),
        "param_space": {
            "rf_n_estimators": ("int", 100, 700, 50),
            "rf_max_depth": ("int", 3, 20),
            "rf_min_samples_leaf": ("int", 2, 5),
            "gbr_n_estimators": ("int", 100, 700, 50),
            "gbr_learning_rate": ("float", 1e-2, 0.2, "log"),
            "gbr_max_depth": ("int", 2, 6),
            "ridge_alpha": ("float", 1e-2, 10, "log")
        }
    },
    2: {
        "base_models": [
            ("rf", RandomForestRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1}),
            ("gbr", GradientBoostingRegressor, {"random_state": RANDOM_STATE}),
            ("ridge", Ridge, {})
        ],
        "final_model": ("rf_final", RandomForestRegressor, {"random_state": RANDOM_STATE, "n_jobs": -1}),
        "param_space": {
            "rf_n_estimators": ("int", 100, 700, 50),
            "rf_max_depth": ("int", 3, 20),
            "rf_min_samples_leaf": ("int", 2, 5),
            "gbr_n_estimators": ("int", 100, 700, 50),
            "gbr_learning_rate": ("float", 1e-2, 0.2, "log"),
            "gbr_max_depth": ("int", 2, 6),
            "ridge_alpha": ("float", 1e-2, 10, "log"),
            "rf_final_n_estimators": ("int", 100, 500, 50),
            "rf_final_max_depth": ("int", 3, 15)
        }
    },
}
# 선택한 TEST_ID에 해당하는 구성만 실행하도록 설정 (요구사항 3)
if TEST_ID not in MODEL_CONFIGS:
    print(f"[Error] Undefined TEST_ID: {TEST_ID}. Please check MODEL_CONFIGS.")
    sys.exit(1)
SELECTED_CONFIG = MODEL_CONFIGS[TEST_ID]

# --- 출력 파일 이름 --- (요구사항 4: 기존 이름에 TEST_ID suffix 추가)
OUT_BUNDLE         = f"stack_bundle_{TEST_ID}.joblib"
OUT_SCORES         = f"stack_scores_{TEST_ID}.csv"
HYPERPARAMS_FILE   = f"best_hyperparameters_{TEST_ID}.json"
# =============================================================================


def build_feats_by_target(all_feats: List[str], targets: List[str]) -> Dict[str, List[str]]:
    """타겟별로 상호 배타적인 피처를 제외한 피처 리스트를 생성합니다."""
    feats_map = {}
    for tgt in targets:
        ban = set(MUT_EXCL.get(tgt, []))
        # 타겟 자신과 상호 배타 피처를 제외
        feats = [c for c in all_feats if c != tgt and c not in ban]
        feats_map[tgt] = feats
    return feats_map


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """데이터 타입에 따라 전처리 파이프라인(ColumnTransformer)을 구성합니다."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 데이터 특성에 따른 변환 대상 컬럼 지정
    log_cols = [c for c in num_cols if c.lower() in ['phi_mn']]
    exp_cols = [c for c in num_cols if c.lower() in ['rho']]
    other_num_cols = [c for c in num_cols if c not in log_cols and c not in exp_cols]

    # 파이프라인 정의
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", MinMaxScaler())
    ])
    exp_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("exp_transform", FunctionTransformer(np.exp, validate=False)),
        ("scaler", MinMaxScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if other_num_cols:
        transformers.append(("numeric", num_pipe, other_num_cols))
    if log_cols:
        transformers.append(("log_transformed", log_pipe, log_cols))
    if exp_cols:
        transformers.append(("exp_transformed", exp_pipe, exp_cols))
    if cat_cols:
        transformers.append(("categorical", cat_pipe, cat_cols))
    
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """회귀 모델의 주요 성능 지표(MAE, RMSE, R2)를 계산합니다."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_pipeline_kfold(pipe: Pipeline, X_df: pd.DataFrame, y: np.ndarray, n_splits: int) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    주어진 파이프라인에 대해 K-fold 교차 검증을 수행하고 성능 지표를 반환합니다.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for tr_idx, va_idx in kf.split(X_df):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        p = clone(pipe)
        p.fit(X_tr, y_tr)
        y_hat = p.predict(X_va)
        
        metrics = calculate_regression_metrics(y_va, y_hat)
        fold_scores.append(metrics)
        
    # 폴드별 점수의 평균 계산
    mean_scores = {metric: np.mean([s[metric] for s in fold_scores]) for metric in fold_scores[0]}
    
    return mean_scores, fold_scores


def make_stacking_pipeline(X: pd.DataFrame, params: Dict[str, Any]) -> Pipeline:
    """주어진 하이퍼파라미터로 Stacking 파이프라인을 구성합니다."""
    preprocessor = build_preprocessor(X)

    # Base models (from SELECTED_CONFIG, 요구사항 2,3)
    base_estimators: List[Tuple[str, Any]] = []
    for name, model_class, fixed_params in SELECTED_CONFIG["base_models"]:
        # 각 base model에 대해 고정 파라미터 복사 및 하이퍼파라미터 적용
        model_params = fixed_params.copy()
        for key, value in params.items():
            if key.startswith(f"{name}_"):
                param_name = key[len(name)+1:]
                model_params[param_name] = value
        model_instance = model_class(**model_params)
        base_estimators.append((name, model_instance))
    
    # Final meta-model (from SELECTED_CONFIG, 요구사항 2,3)
    final_name, final_class, final_fixed_params = SELECTED_CONFIG["final_model"]
    final_params = final_fixed_params.copy()
    for key, value in params.items():
        if key.startswith(f"{final_name}_"):
            param_name = key[len(final_name)+1:]
            final_params[param_name] = value
    final_estimator = final_class(**final_params)

    stacking_regressor = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=False
    )
    
    return Pipeline([("preprocessor", preprocessor), ("stacking_regressor", stacking_regressor)])


def optuna_objective(trial: optuna.Trial, X_df: pd.DataFrame, y: np.ndarray) -> float:
    """Optuna 탐색을 위한 목적 함수."""
    params: Dict[str, Any] = {}
    # SELECTED_CONFIG에 정의된 하이퍼파라미터 탐색 공간을 이용하여 파라미터 샘플링 (요구사항 2)
    for param_name, spec in SELECTED_CONFIG["param_space"].items():
        if spec[0] == "int":
            low, high = spec[1], spec[2]
            if len(spec) >= 4 and isinstance(spec[3], int):
                params[param_name] = trial.suggest_int(param_name, low, high, step=spec[3])
            else:
                params[param_name] = trial.suggest_int(param_name, low, high)
        elif spec[0] == "float":
            low, high = spec[1], spec[2]
            log_flag = False
            if len(spec) >= 4 and spec[3] == "log":
                log_flag = True
            params[param_name] = trial.suggest_float(param_name, low, high, log=log_flag)
        # 다른 타입의 하이퍼파라미터도 추가 가능 (예: categorical)

    pipeline = make_stacking_pipeline(X_df, params)
    mean_scores, fold_scores = evaluate_pipeline_kfold(pipeline, X_df, y, n_splits=CV_FOLDS)

    # 콘솔에 상세 성능 출력
    print(f"[Trial {trial.number:03d}] CV Mean Scores: "
          f"MAE={mean_scores['MAE']:.5f}, RMSE={mean_scores['RMSE']:.5f}, R2={mean_scores['R2']:.5f}")
          
    return mean_scores['MAE']  # MAE를 최소화하는 것을 목표로 함


def run_optuna_study(X_df: pd.DataFrame, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
    """특정 타겟에 대해 Optuna 탐색을 실행하고 최적 파라미터를 반환합니다."""
    study = optuna.create_study(direction="minimize")
    objective_func = lambda trial: optuna_objective(trial, X_df, y)
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[Optuna] Best MAE: {study.best_value:.5f}")
    print(f"[Optuna] Best params: {study.best_params}")
    return study.best_params


def load_hyperparameters(filepath: str) -> Dict[str, Any]:
    """JSON 파일에서 하이퍼파라미터를 로드합니다."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] Hyperparameter file not found: {filepath}")
        print("Please run with USE_OPTUNA=True first to generate the file.")
        sys.exit(1)


def save_hyperparameters(params: Dict[str, Any], filepath: str):
    """하이퍼파라미터를 JSON 파일로 저장합니다."""
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"[OK] Best hyperparameters saved to: {filepath}")


def generate_and_save_reports(
    models: Dict[str, Pipeline],
    feats_by_tgt: Dict[str, List[str]],
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame
):
    """학습된 모델들의 학습/테스트 성능을 평가하고 CSV 파일로 저장합니다."""
    report_rows = []
    print("\n--- Final Model Performance Report ---")
    
    for tgt, model in models.items():
        feats = feats_by_tgt[tgt]
        X_tr, y_tr = df_tr[feats], df_tr[tgt].to_numpy(dtype=float)
        X_te, y_te = df_te[feats], df_te[tgt].to_numpy(dtype=float)

        # Train set 성능
        y_tr_pred = model.predict(X_tr)
        train_metrics = calculate_regression_metrics(y_tr, y_tr_pred)
        train_metrics['target'] = tgt
        train_metrics['dataset'] = 'train'
        report_rows.append(train_metrics)
        
        # Test set 성능
        y_te_pred = model.predict(X_te)
        test_metrics = calculate_regression_metrics(y_te, y_te_pred)
        test_metrics['target'] = tgt
        test_metrics['dataset'] = 'test'
        report_rows.append(test_metrics)

        print(f"[{tgt:^10s}] Train | MAE={train_metrics['MAE']:.6f}, RMSE={train_metrics['RMSE']:.6f}, R2={train_metrics['R2']:.6f}")
        print(f"[{tgt:^10s}] Test  | MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.6f}")

    # CSV 파일 저장
    report_df = pd.DataFrame(report_rows)[['target', 'dataset', 'MAE', 'RMSE', 'R2']]
    report_df.to_csv(OUT_SCORES, index=False)
    print(f"\n[OK] Performance scores saved to: {OUT_SCORES}")


def main():
    """메인 실행 함수"""
    df = pd.read_csv(DATA_CSV)
    df = df[COL_FEAT + COL_TGTS] # 필요한 모든 컬럼 선택
    df.drop_duplicates(inplace=True)

    feats_by_tgt = build_feats_by_target(COL_FEAT, COL_TGTS)
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    models: Dict[str, Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_params_by_tgt: Dict[str, Dict[str, Any]] = {}

    if not USE_OPTUNA:
        print(f"--- Skipping Optuna. Loading parameters from {HYPERPARAMS_FILE} ---")
        best_params_by_tgt = load_hyperparameters(HYPERPARAMS_FILE)
        # 로드된 파라미터가 모든 타겟에 대해 존재하는지 확인
        if not all(tgt in best_params_by_tgt for tgt in COL_TGTS):
            print("[Error] The hyperparameter file is missing parameters for some targets.")
            sys.exit(1)

    # 타겟별 모델 학습
    for tgt in tqdm(COL_TGTS, desc="Training models", ncols=100):
        print(f'\n--- Processing target: "{tgt}" ---')
        feats = feats_by_tgt[tgt]
        X_tr_tgt = df_tr[feats].copy()
        y_tr_tgt = df_tr[tgt].to_numpy(dtype=float)
        
        dtypes_by_tgt[tgt] = X_tr_tgt.dtypes.astype(str).to_dict()

        if USE_OPTUNA:
            print(f"[Optuna] Starting hyperparameter search for '{tgt}'...")
            best_params = run_optuna_study(X_tr_tgt, y_tr_tgt, n_trials=N_TRIALS)
            best_params_by_tgt[tgt] = best_params
        else:
            # 파일에서 로드한 파라미터 사용
            best_params = best_params_by_tgt[tgt]
            print(f"[Fixed Params] Using pre-loaded parameters for '{tgt}'.")

        # 최적 파라미터로 최종 모델 학습 (전체 학습 데이터 사용)
        print(f"--- Fitting final model for '{tgt}' with best parameters ---")
        final_pipeline = make_stacking_pipeline(X_tr_tgt, best_params)
        final_pipeline.fit(X_tr_tgt, y_tr_tgt)
        models[tgt] = final_pipeline

    # Optuna를 사용했다면, 찾은 최적 파라미터를 파일에 저장
    if USE_OPTUNA:
        save_hyperparameters(best_params_by_tgt, HYPERPARAMS_FILE)

    # 다른 앱과의 연동을 위한 번들 저장 (요구사항 5: 기존 구조 유지)
    bundle = {
        "models": models,
        "features_by_target": feats_by_tgt,
        "dtypes_by_target": dtypes_by_tgt,
        "targets": list(models.keys()),
        "best_params_by_target": best_params_by_tgt,
        "sklearn_version": sk_version,
    }
    joblib.dump(bundle, OUT_BUNDLE, compress=3)
    print(f"\n[OK] Model bundle saved to: {OUT_BUNDLE}")

    # 최종 성능 리포트 생성 및 저장
    generate_and_save_reports(models, feats_by_tgt, df_tr, df_te)


if __name__ == "__main__":
    main()
