"""
train_stack9.py
Scikit-learn StackingRegressor 기반 스태킹 학습 + Optuna 파라미터 탐색 + 그룹화 적용.
"""
from __future__ import annotations
import os
import sys
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
from sklearnex import patch_sklearn     # https://pypi.org/project/scikit-learn-intelex/
patch_sklearn()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", category=UserWarning)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================= 사용자 설정 =========================
DATA_CSV   = "batch_analysis_rect_result.csv"
COL_FEAT   = ['f_idx', 'width', 'height', 'Sm', 'bd', 'rho', 'phi_mn']
COL_TGTS   = ['Sm', 'bd', 'rho', 'phi_mn']
MUT_EXCL   = {'as_provided': ['phi_mn'], 'phi_mn': ['as_provided']}

TEST_SIZE    = 0.2
RANDOM_STATE = 42
OUT_BUNDLE   = "stack_bundle.joblib"
OUT_SCORES   = "stack_scores.csv"
# =============================================================


def build_feats_by_target(all_feats, targets):
    feats_map = {}
    for tgt in targets:
        ban = set(MUT_EXCL.get(tgt, []))
        feats = [c for c in all_feats if c not in ban and c != tgt]
        feats_map[tgt] = feats
    return feats_map


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    # log 변환
    log_cols = [c for c in num_cols if c.lower() in ['phi_mn']]
    # exp 변환
    exp_cols = [c for c in num_cols if c.lower() in ['rho']]
    other_num_cols = [c for c in num_cols if c not in log_cols and c not in exp_cols]
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        # ("sc", StandardScaler())
        ("sc", MinMaxScaler())
        # ("sc", RobustScaler())
    ])
    log_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        # ("sc", StandardScaler())
        ("sc", MinMaxScaler())
        # ("sc", RobustScaler())
    ])
    exp_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.exp, validate=False)),
        # ("sc", StandardScaler())
        ("sc", MinMaxScaler())
        # ("sc", RobustScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers_ = []
    if other_num_cols:
        transformers_.append(("num", num_pipe, other_num_cols))
    if log_cols:
        transformers_.append(("log", log_pipe, log_cols))
    if exp_cols:
        transformers_.append(("exp", exp_pipe, exp_cols))
    if cat_cols:
        transformers_.append(("cat", cat_pipe, cat_cols))    
    
    pre = ColumnTransformer(
        transformers=transformers_,
        # remainder="drop",
        # sparse_threshold=0.0,
        # verbose_feature_names_out=False,
    )
    return pre


def kfold_mae_for_pipeline(pipe: Pipeline, X_df: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> Tuple[float, List[float]]:
    """
    Pipeline( pre + StackingRegressor )에 대해 수동 KFold로 MAE를 측정.
    - 각 fold에서 fit → predict → MAE
    - fold별 MAE 리스트와 평균 MAE 반환
    - StackingRegressor 내부에서 base는 full X에 맞춰 재적합되며,
      meta는 cv 기반 OOF 입력으로 학습됨(sklearn 구현에 따름).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_df), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # 새 파이프라인으로 클론 후 학습
        # (StackingRegressor는 내부적으로 fold별로 meta를 OOF로 학습)
        # 여기서는 외부에서 파이프라인을 fold 단위로 재학습하여 검증 성능을 본다.
        p = clone(pipe)
        p.fit(X_tr, y_tr)
        y_hat = p.predict(X_va)
        mae = mean_absolute_error(y_va, y_hat)
        maes.append(mae)
    return float(np.mean(maes)), maes


def make_stacking_pipeline(X: pd.DataFrame, params: Dict[str, Any]) -> Pipeline:
    """
    주어진 하이퍼파라미터로 Stacking 파이프라인 구성.
    - base preset은 1세트 고정. params로 일부 값만 치환.
    - final_estimator도 params로 alpha 등 치환 가능.
    """
    pre = build_preprocessor(X)

    # base 복제 및 파라미터 적용
    rf_n_estimators     = params.get("rf_n_estimators", 300)
    rf_max_depth        = params.get("rf_max_depth", None)
    rf_min_samples_leaf = params.get("rf_min_samples_leaf", 2)
    gbr_n_estimators    = params.get("gbr_n_estimators", 300)
    gbr_learning_rate   = params.get("gbr_learning_rate", 0.1)
    gbr_max_depth       = params.get("gbr_max_depth", 3)

    rf  = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    gbr = GradientBoostingRegressor(
        n_estimators=gbr_n_estimators,
        learning_rate=gbr_learning_rate,
        max_depth=gbr_max_depth,
        random_state=RANDOM_STATE
    )

    base_estimators = [("rf", rf), ("gbr", gbr)]
    ridge_alpha = params.get("ridge_alpha", 1.0)
    final_est = Ridge(alpha=ridge_alpha)

    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_est,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=False
    )
    pipe = Pipeline([
        ("pre", pre),
        ("stack", stack)
    ])
    return pipe


def optuna_tune_for_target(X_df: pd.DataFrame, y: np.ndarray, n_trials: int = 30) -> Dict[str, Any]:
    """
    단일 타깃에 대한 Optuna 탐색.
    - 각 trial에서 5-fold MAE의 fold별 점수를 print
    - 반환: best_params
    """
    def objective(trial: optuna.Trial) -> float:
        # 탐색 공간
        params = {
            "rf_n_estimators":     trial.suggest_int("rf_n_estimators", 100, 700, step=50),
            "rf_max_depth":        trial.suggest_int("rf_max_depth",  3,  20),
            "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 2, 5),
            "gbr_n_estimators":    trial.suggest_int("gbr_n_estimators", 100, 700, step=50),
            "gbr_learning_rate":   trial.suggest_float("gbr_learning_rate", 1e-2, 0.2, log=True),
            "gbr_max_depth":       trial.suggest_int("gbr_max_depth", 2, 6),
            "ridge_alpha":         trial.suggest_float("ridge_alpha", 1e-2, 10, log=True),
        }
        pipe = make_stacking_pipeline(X_df, params)
        mean_mae, fold_mae = kfold_mae_for_pipeline(pipe, X_df, y, n_splits=5)
        # ---- 콘솔 출력: fold별 성적 ----
        print(f"[Trial {trial.number:03d}] MAE per fold:", ", ".join(f"{m:.5f}" for m in fold_mae))
        print(f"[Trial {trial.number:03d}] Mean MAE: {mean_mae:.5f}")
        return mean_mae  # minimize
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("[Optuna] Best value:", study.best_value)
    print("[Optuna] Best params:", study.best_params)
    return study.best_params


def train_per_target_with_optuna(
    df: pd.DataFrame,
    targets: List[str],
    feats_by_tgt: Dict[str, List[str]],
    n_trials: int = 30
) -> Tuple[Dict[Tuple[str, str], Pipeline], Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, Any]]]:
    
    """타겟별 Optuna → 최적 파라미터로 full train 파이프라인 학습"""
    models: Dict[Tuple[str, str], Pipeline] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_params_by_tgt: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for tgt in tqdm(targets, desc="Training models", ncols=100):
        print(f'\n--- Start training model for target "{tgt}" ---')
        feats = feats_by_tgt[tgt]
        X_df = df[feats].copy()
        y = df[tgt].to_numpy(dtype=float)
        dtypes_by_tgt[tgt] = X_df.dtypes.astype(str).to_dict()
        
        # 그룹화 적용 (bd, rho, phi_mn에만)
        if tgt in ['bd', 'rho', 'phi_mn']:
            for group_name, sub_df in df.groupby(['f_idx', 'width', 'height']):
                X_group = sub_df[feats].copy()
                y_group = sub_df[tgt].to_numpy(dtype=float)
                # Optuna 탐색
                print(f"  Tuning for group: {group_name}")
                best_params = optuna_tune_for_target(X_group, y_group, n_trials=n_trials if len(sub_df) > 25 else 10)
                best_params_by_tgt[(tgt, str(group_name))] = best_params
                # 최적 파라미터로 full fit
                best_pipe = make_stacking_pipeline(X_group, best_params)
                best_pipe.fit(X_group, y_group)
                models[(tgt, str(group_name))] = best_pipe
        else:
            # Optuna 탐색
            best_params = optuna_tune_for_target(X_df, y, n_trials=n_trials)
            best_params_by_tgt[(tgt, 'all')] = best_params
            # 최적 파라미터로 full fit
            best_pipe = make_stacking_pipeline(X_df, best_params)
            best_pipe.fit(X_df, y)
            models[(tgt, 'all')] = best_pipe
    return models, dtypes_by_tgt, best_params_by_tgt


def main():    
    df = pd.read_csv(DATA_CSV)
    df = df[COL_FEAT]   # preprocessing 전단계에서 학습에 필요한 컬럼 미리 선정(매우 중요하므로 삭제 불가)

    feats_by_tgt = build_feats_by_target(COL_FEAT, COL_TGTS)
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    models, dtypes_by_tgt, best_params_by_tgt = train_per_target_with_optuna(df_tr, COL_TGTS, feats_by_tgt, n_trials=30)

    bundle = {
        "models": models,                      # {(tgt, group): Pipeline}
        "features_by_target": feats_by_tgt,    # {tgt: [feat,...]}
        "dtypes_by_target": dtypes_by_tgt,     # {tgt: {feat: dtype_str}}
        "targets": list(models.keys()),        # [(tgt, group),...]
        "best_params_by_target": best_params_by_tgt,    # {(tgt, group): [parameters,...]}
        "sklearn_version": sk_version,
    }

    joblib.dump(bundle, OUT_BUNDLE, compress=3)
    print(f"[OK] Saved bundle: {OUT_BUNDLE}")

    # 홀드아웃 성능 리포트
    rows = []
    for tgt in set(t[0] for t in models.keys()):  # 타겟별 집계
        feats = feats_by_tgt[tgt]
        X_te = df_te[feats]
        y_true = df_te[tgt].to_numpy(dtype=float)
        # 그룹별 예측
        group_ids = X_te.apply(lambda row: f"{row['f_idx']}_{row['width']}_{row['height']}", axis=1)
        y_pred = np.array([models.get((tgt, str(gid))) for gid in group_ids])
        # 예측 실패 시 예외 발생
        if any(m is None for m in y_pred):            
            print(f"Prediction failed: No trained model for some groups in target '{tgt}'")
            y_pred = None
            break
        y_pred = np.array([m.predict(X_te.loc[[i]])[0] for i, m in enumerate(y_pred) if m is not None])
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        rows.append({"target": tgt, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"[TEST] {tgt:8s}  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
    pd.DataFrame(rows).to_csv(OUT_SCORES, index=False)
    print(f"Scores CSV: {OUT_SCORES}")

if __name__ == "__main__":
    main()