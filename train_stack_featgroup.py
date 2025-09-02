<<<<<<< HEAD
"""
train_stack_featgroup.py
Scikit-learn StackingRegressor 기반 스태킹 학습 + Optuna 파라미터 탐색 + 그룹화 적용.

추가수정요청사항

1. ML 모델 훈련 코드의 Core Design concept 을 review 하고 불일치 하는 부분을 수정한다.
- Per-target training with one StackingRegressor per target.
- Base estimators fit on full X; final estimator is trained by StackingRegressor using OOF
  (cross_val_predict under the hood) with KFold(n_splits=CV_FOLDS, shuffle=True, random_state).
- Global, reproducible configuration via constants:
  TEST_SIZE, RANDOM_STATE, CV_FOLDS.

2. Path, os 모듈로 하위에 output, input 폴더 생성.  파일 저장은 output 폴더에 일관되게 모두 저장함.  input 폴더는 사용자 설정 파일을 불러오는데 사용한다.

3. Optuna Integration method 검토 하고 불일치 하는 부분을 수정한다.
- USE_OPTUNA=True: runs per-target optimization on the selected preset.
  - Objective: minimize MAE via KFold CV (CV_FOLDS).
  - Each trial prints fold-wise MAE and the mean MAE.
  - USE_OPTUNA=False: loads saved best hyperparameters (JSON) and trains directly (no search). 즉, 사용자가 USE_OPTUNA=False 로 설정할 경우, 코드는 input 폴더에 저장된 사용자가 수동으로 설정한 모델훈련파라메터 json 파일을 load 해서 모델 training 에 사용한다.
  - N_TRIALS controls the number of Optuna trials when enabled.

4. optuna.pruners 의 MedianPruner 기능 추가, MAE 최소화 기준에 따른 일관된 pruning 작동 -> K-Fold 진행 중 열세한 trial 을 중도 중단시킴

5. StackingRegressor 모델을 생성할때 passthrough=True/False 를 사용자가 수동 지정할 수 있도록 코드를 수정함.

6. 모델 훈련의 best parameter bundle 에도 저장하고, 별도의 json 파일에도 저장한다.  사용자는 모델의 평가가 완료된 후 최종 모델의 훈련을 위해 이 json 파일을 input 폴더로 복사해서 수동으로 모델 훈련 파라메터들을 수정할 수 있다.

7. 모델 훈련 완료 후 bundle 저장은 api.py, index.html 등 외부 코드와 호환성을 유지하기 위해 bundle 에 저장되는 model의 현재 저장 구조를 변경하지 않으며, 또한 bundle 에 저장되는 변수명과 구조를 변경하지 않는다.

8. 기능추가 / 변경 / 코드 오류 fix 작업이 완료된 후, code 의 전체 refactoring 작업을 실시한다.  refactoring 의 최우선 목표는 main() 함수는 작성된 함수를 작업 순서에 따라 호출하고 반환된 값을 사용하여 다른 함수를 호출하는 목적으로 사용하여 main() 함수 내부의 코드는 최대한 심플하게 정리한다.  예를 들어 현재 main() 내부의 홀드아웃 성능 리포트는 별도의 함수를 추가하여 호출할 수 있다.

9. code refactoring 은 코드 내부의 변수(variables) 사용 방법 등 기본적인 코드 작성 규칙들이 정상적으로 작성되어 있는지 확인하는 작업을 포함한다.

10. 수정된 코드 생성시 현재 작성되어 있는 주석 등 사용자의 의도가 반영된 사항들은 삭제, 수정을 최소화 한다. 함수를 설명하는 주석 등이 반드시 수정이 필요한 경우에는 수정한다.

11. 모든 수정작업이 완료된 후, DOCSTRING 을 생성해서 코드 최상단에 추가한다.  DOCSTRING 은 전체 코드 내용을 comprehensive 하게 요약하여 사용자가 미래에 유지관리를 할때 편리하게 도움을 주는 목적으로 작성한다.  DOCSTRING은 최종 수정된 코드의 structure 에 대한 설명을 포함한다.

12. DOCSTRING은 (Train, Validation or Cross-Validation, Test) 작업 과정에서 데이터셋이 어떻게 분할되고, 사용되는지 설명한다.  또한 StackingRegressor 가 OOB 데이터를 이용해서 어떻게 Training 을 하고있는지도 설명한다.
"""
from __future__ import annotations
import os
import sys
import json
import warnings
import joblib
import numpy as np

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
import optuna
from optuna.pruners import MedianPruner

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
OUT_BUNDLE   = "stack_group_bundle.joblib"
OUT_SCORES   = "stack_group_scores.csv"

# 출력,입력 디렉토리 생성
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
input_dir = Path('./input')
input_dir.mkdir(exist_ok=True)
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
    
    return ColumnTransformer(transformers=transformers_, remainder="passthrough")


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
=======
"""
train_stack_featgroup.py
Scikit-learn StackingRegressor 기반 스태킹 학습 + Optuna 파라미터 탐색 + 그룹화 적용.

추가수정요청사항

1. ML 모델 훈련 코드의 Core Design concept 을 review 하고 불일치 하는 부분을 수정한다.
- Per-target training with one StackingRegressor per target.
- Base estimators fit on full X; final estimator is trained by StackingRegressor using OOF
  (cross_val_predict under the hood) with KFold(n_splits=CV_FOLDS, shuffle=True, random_state).
- Global, reproducible configuration via constants:
  TEST_SIZE, RANDOM_STATE, CV_FOLDS.

2. Path, os 모듈로 하위에 output, input 폴더 생성.  파일 저장은 output 폴더에 일관되게 모두 저장함.  input 폴더는 사용자 설정 파일을 불러오는데 사용한다.

3. Optuna Integration method 검토 하고 불일치 하는 부분을 수정한다.
- USE_OPTUNA=True: runs per-target optimization on the selected preset.
  - Objective: minimize MAE via KFold CV (CV_FOLDS).
  - Each trial prints fold-wise MAE and the mean MAE.
  - USE_OPTUNA=False: loads saved best hyperparameters (JSON) and trains directly (no search). 즉, 사용자가 USE_OPTUNA=False 로 설정할 경우, 코드는 input 폴더에 저장된 사용자가 수동으로 설정한 모델훈련파라메터 json 파일을 load 해서 모델 training 에 사용한다.
  - N_TRIALS controls the number of Optuna trials when enabled.

4. optuna.pruners 의 MedianPruner 기능 추가, MAE 최소화 기준에 따른 일관된 pruning 작동 -> K-Fold 진행 중 열세한 trial 을 중도 중단시킴

5. StackingRegressor 모델을 생성할때 passthrough=True/False 를 사용자가 수동 지정할 수 있도록 코드를 수정함.

6. 모델 훈련의 best parameter bundle 에도 저장하고, 별도의 json 파일에도 저장한다.  사용자는 모델의 평가가 완료된 후 최종 모델의 훈련을 위해 이 json 파일을 input 폴더로 복사해서 수동으로 모델 훈련 파라메터들을 수정할 수 있다.

7. 모델 훈련 완료 후 bundle 저장은 api.py, index.html 등 외부 코드와 호환성을 유지하기 위해 bundle 에 저장되는 model의 현재 저장 구조를 변경하지 않으며, 또한 bundle 에 저장되는 변수명과 구조를 변경하지 않는다.

8. 기능추가 / 변경 / 코드 오류 fix 작업이 완료된 후, code 의 전체 refactoring 작업을 실시한다.  refactoring 의 최우선 목표는 main() 함수는 작성된 함수를 작업 순서에 따라 호출하고 반환된 값을 사용하여 다른 함수를 호출하는 목적으로 사용하여 main() 함수 내부의 코드는 최대한 심플하게 정리한다.  예를 들어 현재 main() 내부의 홀드아웃 성능 리포트는 별도의 함수를 추가하여 호출할 수 있다.

9. code refactoring 은 코드 내부의 변수(variables) 사용 방법 등 기본적인 코드 작성 규칙들이 정상적으로 작성되어 있는지 확인하는 작업을 포함한다.

10. 수정된 코드 생성시 현재 작성되어 있는 주석 등 사용자의 의도가 반영된 사항들은 삭제, 수정을 최소화 한다. 함수를 설명하는 주석 등이 반드시 수정이 필요한 경우에는 수정한다.

11. 모든 수정작업이 완료된 후, DOCSTRING 을 생성해서 코드 최상단에 추가한다.  DOCSTRING 은 전체 코드 내용을 comprehensive 하게 요약하여 사용자가 미래에 유지관리를 할때 편리하게 도움을 주는 목적으로 작성한다.  DOCSTRING은 최종 수정된 코드의 structure 에 대한 설명을 포함한다.

12. DOCSTRING은 (Train, Validation or Cross-Validation, Test) 작업 과정에서 데이터셋이 어떻게 분할되고, 사용되는지 설명한다.  또한 StackingRegressor 가 OOB 데이터를 이용해서 어떻게 Training 을 하고있는지도 설명한다.
"""
from __future__ import annotations
import os
import sys
import json
import warnings
import joblib
import numpy as np

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
import optuna
from optuna.pruners import MedianPruner

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
OUT_BUNDLE   = "stack_group_bundle.joblib"
OUT_SCORES   = "stack_group_scores.csv"

# 출력,입력 디렉토리 생성
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
input_dir = Path('./input')
input_dir.mkdir(exist_ok=True)
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
    
    return ColumnTransformer(transformers=transformers_, remainder="passthrough")


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
>>>>>>> e9f0438d2bed29fdc1037b3c5bdd853ae6f37f72
    main()