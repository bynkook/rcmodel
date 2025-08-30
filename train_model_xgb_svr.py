# train_model_xgb.py — XGB + SVR 동시 학습/평가/저장
# f_idx, rho, 신규 생성
# f_idx = 27400 / 1e3 = 27.4
# rho = as_provided / (width * d) 
# best_(model)_params_by_target 을 bundle 에 추가함

import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from dataclasses import dataclass
from tqdm.auto import tqdm
from icecream import ic

import joblib
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.svm import SVR
from sklearn import __version__ as sk_version
from sklearnex import patch_sklearn     # https://pypi.org/project/scikit-learn-intelex/
patch_sklearn()
from xgboost import XGBRegressor

# ---------------- config ----------------
DATA_PATH = "batch_analysis_rect_result.csv"
OUT_XGB   = "xgb_bundle.joblib"
OUT_SVR   = "svr_bundle.joblib"
OUT_CSV   = "model_comparison_results.csv"

N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 30
TIMEOUT_SEC = None
N_JOBS = -1

# TODO : 단면 형상('shape') cat_cols 에 추가 예정
# 2025.08.26    findex, agindex, rho 추가
# COL_FEAT = ['f_idx', 'width', 'height', 'Ag_idx', 'd', 'Sm', 'as_provided', 'rho', 'phi_mn']  # 필요한 항목만 학습 대상으로 선택
COL_FEAT = ['f_idx', 'width', 'height', 'Sm', 'bd', 'rho', 'phi_mn']  # 필요한 항목만 학습 대상으로 선택
# COL_TGTS = ['d', 'Sm', 'as_provided', 'phi_mn']     # phi_mn 을 target 에 추가함(Mu > phi_Mn 검토용) 2025.08.24
COL_TGTS = ['Sm', 'bd', 'rho', 'phi_mn']     # for test
# MUT_EXCL = {'as_provided':['phi_mn'], 'phi_mn':['as_provided']}     # 상관성이 높은 feature 을 상호 배제
MIN_UNIQUE_FOR_REG = 3        # 타깃 유효성 최소 고유값 개수

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

ic.configureOutput(prefix='DEBUG :  ')
ic.disable()
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------- helpers ----------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

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
    transformers = []
    if other_num_cols:
        transformers.append(("num", num_pipe, other_num_cols))
    if log_cols:
        transformers.append(("log", log_pipe, log_cols))
    if exp_cols:
        transformers.append(("exp", exp_pipe, exp_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers)

def build_xgb_estimator(X_sample: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_sample)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="exact",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return Pipeline([("pre", pre), ("model", xgb)])

def xgb_search_space() -> Dict[str, Any]:
    return {
        "model__n_estimators":  optuna.distributions.IntDistribution(50, 350, step=10),
        "model__max_depth":     optuna.distributions.IntDistribution(3, 20),
        "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
        "model__subsample":     optuna.distributions.FloatDistribution(0.6, 1.0),
        # "model__colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
        # "model__min_child_weight": optuna.distributions.IntDistribution(1, 10),
        # "model__reg_alpha":     optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
        # "model__reg_lambda":    optuna.distributions.FloatDistribution(1.0, 20.0, log=True),
        # "model__gamma":         optuna.distributions.FloatDistribution(0.0, 1.0),
    }

def build_svr_estimator(X_sample: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_sample)
    svr = SVR(kernel="rbf")
    return Pipeline([("pre", pre), ("model", svr)])

def svr_search_space() -> Dict[str, Any]:
    return {
        "model__C":       optuna.distributions.FloatDistribution(5e-2, 300.0, log=True),
        "model__gamma":   optuna.distributions.FloatDistribution(5e-2, 20.0, log=False),
        "model__epsilon": optuna.distributions.FloatDistribution(5e-2, 3.0, log=False),
    }

# ---------------- main ----------------
def main():
    xgb_models: Dict[str, Any] = {}
    svr_models: Dict[str, Any] = {}
    feats_by_tgt: Dict[str, List[str]] = {}
    dtypes_by_tgt: Dict[str, Dict[str, str]] = {}
    best_xgb_params_by_tgt = {}
    best_svr_params_by_tgt = {}
    rows: List[Dict[str, Any]] = []

    df = pd.read_csv(DATA_PATH)
    df = df[COL_FEAT]  # only selected columns used for model

    # stratify
    # df['strata_bin'] = pd.qcut(df['Ag_idx'], q=df['Ag_idx'].nunique(), labels=False, duplicates='drop')     # 데이터셋의 단면 크기 변화 범위가 클수록 q값은 증가할 것임.

    # target 선택
    candidates = COL_TGTS
    targets = [c for c in candidates if df[c].nunique(dropna=True) >= MIN_UNIQUE_FOR_REG]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for tgt in tqdm(targets, desc="Training targets", mininterval=0.5, ncols=100):
        print(f'\n--- Start training model for target "{tgt}" ---')
        # feats = [x for x in COL_FEAT if x != tgt and x not in MUT_EXCL.get(tgt, [])]   # FEAT 에서 TGT, MUT_EXCL 삭제
        feats = [x for x in COL_FEAT if x != tgt]   # FEAT 에서 TGT 삭제 (철근량은 Mu 휨모멘트 값 없이는 예측이 안됨)
        feats_by_tgt[tgt] = feats   # 저장용

        # for debug
        ic(feats)

        X = df[feats].copy()
        # X = df[feats+['strata_bin']].copy()

        # 예측 시 결측·캐스팅 오류 방지를 위한 dtype 메타 저장
        dtypes_by_tgt[tgt] = X.dtypes.astype(str).to_dict()
        y = df[tgt].copy()

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=X['strata_bin'])

        # drop strata columns
        # X = X.drop(columns=['strata_bin'])
        # X_tr = X_tr.drop(columns=['strata_bin'])
        # X_te = X_te.drop(columns=['strata_bin'])

        # print(X.head())
        # print(X.info())

        # ---------- XGB ----------
        t0 = time.perf_counter()
        xgb_pipe = build_xgb_estimator(X)
        study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        search_xgb = OptunaSearchCV(
            estimator=xgb_pipe,
            param_distributions=xgb_search_space(),
            cv=kf,
            scoring="neg_mean_absolute_error",
            n_trials=N_TRIALS,
            timeout=TIMEOUT_SEC,
            n_jobs=N_JOBS,
            verbose=0,
            refit=True,
            return_train_score=False,
            study=study_xgb,
            random_state=RANDOM_STATE
        )
        search_xgb.fit(X_tr, y_tr)
        best_xgb = search_xgb.best_estimator_
        yhat_xgb = best_xgb.predict(X_te)
        t1 = time.perf_counter()

        rows.append({
            "target": tgt,
            "model": "XGBRegressor",
            "MAE": float(mean_absolute_error(y_te, yhat_xgb)),
            "RMSE": rmse(y_te, yhat_xgb),
            "R2":  float(r2_score(y_te, yhat_xgb)),
            "cv_best_score": float(search_xgb.best_score_),
            "best_params": search_xgb.best_params_,
            "n_trials": len(search_xgb.study_.trials),
            "fit_sec": round(t1 - t0, 3),
            "n_features": len(feats)
        })

        print(f"[{tgt}] model: XGBRegressor, best cv score: {float(search_xgb.best_score_):.6f} | "
              f"MAE: {float(mean_absolute_error(y_te, yhat_xgb)):.6f}, RMSE: {rmse(y_te, yhat_xgb):.6f}, R2: {float(r2_score(y_te, yhat_xgb)):.6f}")

        xgb_models[tgt] = best_xgb
        best_xgb_params_by_tgt[tgt] = search_xgb.best_params_

        # # ---------- SVR ----------
        # t2 = time.perf_counter()
        # svr_pipe = build_svr_estimator(X)
        # study_svr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        # search_svr = OptunaSearchCV(
        #     estimator=svr_pipe,
        #     param_distributions=svr_search_space(),
        #     cv=kf,
        #     scoring="neg_mean_absolute_error",
        #     n_trials=N_TRIALS,
        #     timeout=TIMEOUT_SEC,
        #     n_jobs=N_JOBS,
        #     verbose=0,
        #     refit=True,
        #     return_train_score=False,
        #     study=study_svr,
        #     random_state=RANDOM_STATE
        # )
        # search_svr.fit(X_tr, y_tr)
        # best_svr = search_svr.best_estimator_
        # yhat_svr = best_svr.predict(X_te)
        # t3 = time.perf_counter()

        # rows.append({
        #     "target": tgt,
        #     "model": "SVR(RBF)",
        #     "MAE": float(mean_absolute_error(y_te, yhat_svr)),
        #     "RMSE": rmse(y_te, yhat_svr),
        #     "R2":  float(r2_score(y_te, yhat_svr)),
        #     "cv_best_score": float(search_svr.best_score_),
        #     "best_params": search_svr.best_params_,
        #     "n_trials": len(search_svr.study_.trials),
        #     "fit_sec": round(t3 - t2, 3),
        #     "n_features": len(feats)
        # })

        # print(f"[{tgt}] model: SVR(RBF), best cv score: {float(search_svr.best_score_):.6f} | "
        #       f"MAE: {float(mean_absolute_error(y_te, yhat_svr)):.6f}, RMSE: {rmse(y_te, yhat_svr):.6f}, R2: {float(r2_score(y_te, yhat_svr)):.6f}")

        # svr_models[tgt] = best_svr
        # best_svr_params_by_tgt[tgt] = search_svr.best_params_

    # ---------------- 결과 정리/저장 ----------------
    results_df = pd.DataFrame(rows).sort_values(["target", "model"]).reset_index(drop=True)
    print("\n=== Model comparison ===")
    print(results_df)
    results_df.to_csv(OUT_CSV, index=False)

    xgb_bundle = {
        "models":xgb_models,
        "features_by_target":feats_by_tgt,
        "dtypes_by_target":dtypes_by_tgt,
        "targets":list(xgb_models.keys()),
        "sklearn_version":sk_version,
        "best_xgb_params_by_target": best_xgb_params_by_tgt,
        }
    joblib.dump(xgb_bundle, OUT_XGB)

    # svr_bundle = {
    #     "models":svr_models,
    #     "features_by_target":feats_by_tgt,
    #     "dtypes_by_target":dtypes_by_tgt,
    #     "targets":list(svr_models.keys()),
    #     "sklearn_version":sk_version,
    #     "best_svr_params_by_target": best_svr_params_by_tgt,
    #     }
    # joblib.dump(svr_bundle, OUT_SVR)

    print(f"\nTrained targets: {len(targets)} -> {targets[:10]}{' ...' if len(targets)>10 else ''}")
    print(f"Saved bundles:\n - {OUT_XGB}\n - {OUT_SVR}")
    print(f"Scores CSV: {OUT_CSV}")

if __name__ == "__main__":
    Path(DATA_PATH).exists() or (_ for _ in ()).throw(FileNotFoundError(DATA_PATH))
    main()
