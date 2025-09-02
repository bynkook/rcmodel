"""
stacking_ensemble_titanic_refactored.py
=================================

Purpose
-------
A clean, reusable stacking-ensemble classification pipeline with:
- Dataset-specific steps isolated to THREE functions:
  1) load_and_explore_data()
  2) preprocess_titanic_data(df)
  3) create_features(df)
- All other components are dataset-agnostic and reusable across projects
  (model training, stacking, evaluation, ROC/AUC, SHAP, saving outputs).
- A stacking ensemble implemented using scikit-learn's StackingClassifier,
  which trains base estimators and a meta-estimator on out-of-fold predictions (cv=5 by default).

Reusable Components
-------------------
- StackingClassifier (sklearn): ensemble meta-estimator combining multiple base classifiers.
- create_multilabel_target: generic multi-class target creation from an arbitrary risk score.
- evaluate_models: trains direct & stacked models; returns predictions & metrics (no printing).
- plot_roc_auc: saves ROC curve image & returns AUC dictionary (per-class & micro-average).
- analyze_feature_importance: simple feature importance reporting for tree-based direct model.
- analyze_feature_importance_shap: SHAP summary & bar plots for final model (if supported).
- print_result: central place to print ALL classification analysis text to screen AND file.
- save_results: persist predictions and (optional) feature importances to CSV files.
- print_feature_importance: prints top feature importances for a model and logs them to results.
- print_roc_auc: computes ROC AUC metrics via plot_roc_auc, prints them and logs to results.
- perform_shap_analysis: runs SHAP analysis for a model and logs the status (figures saved or skipped).

Variable & Naming Conventions
-----------------------------
- snake_case for all variables/functions (PEP 8).
- X, y: feature matrix and target (standard ML convention).
- X_train, X_test, y_train, y_test: train/test split.
- direct_model: baseline single classifier (default RandomForestClassifier).
- stacked_model: StackingClassifier (with base estimators + meta estimator).
- y_pred_direct, y_pred_stacked: predictions for test set.
- direct_accuracy, stacked_accuracy: accuracy scores.
- class_names / target_names: optional list[str] used in reports/plots.
- output_dir: all artifacts saved under "./output" without sub-directories.

Outputs
-------
- All TEXT outputs are also saved to "output/results.txt".
- All FIGURES (ROC, SHAP) are saved directly under "output/".
- All CSVs (predictions, feature importances) are saved under "output/".

Notes
-----
- The dataset is split into training (75%) and test (25%) sets (stratified by target).
- The StackingClassifier uses 5-fold cross-validation on the training data to generate out-of-fold 
  predictions from base estimators for training the meta-estimator. This prevents information leakage 
  and overfitting in the stacking process. (No separate validation set is needed beyond this.)
- Passthrough of original features to the meta-model can be toggled via STACKING_PASSTHROUGH (default False).
- This script avoids unused imports and keeps main execution minimal by delegating tasks to functions.
- ROC curve plotting and SHAP analysis are performed without showing interactive plots (figures saved to files only).
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")  # ensure no GUI backend
import matplotlib.pyplot as plt

# Configurable Stacking parameters
STACKING_CV: int = 5        # number of CV folds for meta-model training in stacking
STACKING_PASSTHROUGH: bool = False  # whether to include original features in meta-model input

# 출력, 저장 설정 (main()에서 일괄 관리하므로 여기서는 더 이상 정의하지 않음)
warnings.filterwarnings("ignore", category=UserWarning) # 사용자 코드에 의해 생성되는 경고 무시

def load_and_explore_data() -> pd.DataFrame:
    """
    Load the Titanic dataset (seaborn) and perform minimal sanity checks.
    Returns
    -------
    df : pd.DataFrame
        Raw Titanic dataframe with 'survived' target available.
    """
    import seaborn as sns

    df = sns.load_dataset("titanic")
    # Ensure target available and drop rows with missing target
    if "survived" not in df.columns:
        raise ValueError("Expected 'survived' column in Titanic dataset.")
    df = df.dropna(subset=["survived"]).reset_index(drop=True)
    # Basic exploration prints (captured by logger/print_result later if needed)
    print(f"[INFO] Loaded Titanic: shape={df.shape}, columns={list(df.columns)}")
    return df

def preprocess_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Titanic-specific cleaning. Conservative transforms to retain generality.
    - Keep useful columns
    - Fill missing numeric with median; categorical with mode
    - Drop columns with too many missing or duplicative semantics
    """
    # Retain a compact set of commonly used columns
    keep = [
        "survived", "pclass", "sex", "age", "sibsp", "parch",
        "fare", "embarked", "alone"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Fill numeric/categorical NaNs
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != "survived"]

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Titanic-specific feature engineering:
    - family_size, is_alone (cross-check alone)
    - fare_per_person
    - simple one-hot encoding for ('sex','embarked')
    """
    out = df.copy()
    if set(["sibsp", "parch"]).issubset(out.columns):
        out["family_size"] = out["sibsp"] + out["parch"] + 1
    else:
        out["family_size"] = 1

    if "alone" in out.columns:
        out["is_alone"] = out["alone"].astype(int)
    else:
        out["is_alone"] = (out.get("family_size", 1) == 1).astype(int)

    if "fare" in out.columns and "family_size" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["fare_per_person"] = out["fare"] / out["family_size"]
            out["fare_per_person"] = out["fare_per_person"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # One-hot encode simple categoricals
    cat_cols = [c for c in ["sex", "embarked"] if c in out.columns]
    out = pd.get_dummies(out, columns=cat_cols, drop_first=True)

    return out

def create_multilabel_target(
    df: pd.DataFrame,
    risk_features: Optional[List[str]] = None,
    weights: Optional[Iterable[float]] = None,
    n_classes: int = 5,
    new_col: str = "risk_class",
    percentiles: Optional[Iterable[float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create a multi-class target from a generic risk score = dot(X[risk_features], weights).
    If risk_features/weights are None, a reasonable default is attempted (for Titanic).

    Parameters
    ----------
    df : pd.DataFrame
    risk_features : list[str] or None
        Features used in linear risk score. If None, a default set is tried (Titanic-like).
    weights : iterable[float] or None
        Weights for risk_features. If None, equal weights are used.
    n_classes : int
        Number of ordinal classes to derive by percentile binning.
    new_col : str
        Name of the output multi-class target column.
    percentiles : iterable[float] or None
        Custom percentile boundaries (length n_classes-1). If None, uses equally spaced.

    Returns
    -------
    df_out : pd.DataFrame
        df with added multi-class target column `new_col`.
    info : dict
        Dict of thresholds & scores used.
    """
    df_out = df.copy()

    # Default Titanic-like risk features if not provided
    if risk_features is None:
        candidates = [
            "pclass", "age", "sibsp", "parch", "fare", "family_size",
            "is_alone", "fare_per_person",
            # include some one-hot if present
            "sex_male", "embarked_Q", "embarked_S"
        ]
        risk_features = [c for c in candidates if c in df_out.columns]

    if not risk_features:
        raise ValueError("No risk_features available to compute risk score.")

    x = df_out[risk_features].astype(float).values
    d = x.shape[1]
    if weights is None:
        weights = np.ones(d) / float(d)
    else:
        weights = np.asarray(list(weights), dtype=float)
        if weights.shape[0] != d:
            raise ValueError(f"weights length {weights.shape[0]} != num features {d}")

    risk_score = x @ weights
    df_out["_risk_score"] = risk_score

    # Compute percentile thresholds
    if percentiles is None:
        # e.g., for 5 classes -> 4 boundaries at 20, 40, 60, 80
        percentiles = np.linspace(100 / n_classes, 100 - 100 / n_classes, n_classes - 1)

    thresholds = np.percentile(risk_score, percentiles)
    # Digitize into [0..n_classes-1]
    labels = np.digitize(risk_score, thresholds, right=False).astype(int)
    df_out[new_col] = labels

    info = {
        "risk_features": risk_features,
        "weights": weights,
        "percentiles": list(percentiles),
        "thresholds": thresholds.tolist(),
    }
    return df_out, info

def evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    direct_model: Optional[BaseEstimator] = None,
    stacked_model: Optional[StackingClassifier] = None,
    ) -> Dict[str, Any]:
    """
    Train a direct baseline model and a stacked ensemble; compute predictions & accuracies.
    If no model instances are provided, uses RandomForestClassifier as direct model and a 
    StackingClassifier (with default base estimators and LogisticRegression meta-estimator).
    Returns
    -------
    out : dict
        {
          'direct_model', 'stacked_model',
          'y_pred_direct', 'y_pred_stacked',
          'direct_accuracy', 'stacked_accuracy'
        }
    """
    if direct_model is None:
        direct_model = RandomForestClassifier(n_estimators=300, random_state=42)

    if stacked_model is None:
        base_estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=50, random_state=42)),
        ]
        final_estimator = LogisticRegression(max_iter=300, multi_class="auto", n_jobs=None)
        stacked_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=STACKING_CV,
            passthrough=STACKING_PASSTHROUGH,
            n_jobs=None
        )

    direct_model.fit(X_train, y_train)
    y_pred_direct = direct_model.predict(X_test)
    direct_accuracy = accuracy_score(y_test, y_pred_direct)

    stacked_model.fit(X_train, y_train)
    y_pred_stacked = stacked_model.predict(X_test)
    stacked_accuracy = accuracy_score(y_test, y_pred_stacked)

    return {
        "direct_model": direct_model,
        "stacked_model": stacked_model,
        "y_pred_direct": y_pred_direct,
        "y_pred_stacked": y_pred_stacked,
        "direct_accuracy": direct_accuracy,
        "stacked_accuracy": stacked_accuracy,
    }

def plot_roc_auc(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curve",
    savepath: Optional[Path] = None,
    show_plot: bool = False,
    ) -> Dict[str, float]:
    """
    Compute ROC curves per class & micro-average. Save figure if savepath provided.
    Return dict of AUCs (per class + micro).
    """
    # Ensure probabilities are available
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for ROC/AUC.")

    # Binarize y for multi-class one-vs-rest ROC
    classes = np.unique(y)
    y_bin = np.zeros((len(y), len(classes)), dtype=int)
    for i, c in enumerate(classes):
        y_bin[:, i] = (y == c).astype(int)

    proba = model.predict_proba(X)
    # Handle binary meta-proba shape (n, 2) -> use positive class column
    if proba.ndim == 2 and proba.shape[1] == 2 and len(classes) == 2:
        proba = np.column_stack([1 - proba[:, 1], proba[:, 1]])

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    # Per-class ROC
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        fpr_dict[i], tpr_dict[i] = fpr, tpr
        auc_dict[f"class_{c}"] = auc(fpr, tpr)

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), proba.ravel())
    auc_dict["micro"] = auc(fpr_micro, tpr_micro)

    # Plot once (optional), mainly to save to file
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(classes):
        lbl = f"{class_names[i]}" if (class_names and i < len(class_names)) else f"class {c}"
        ax.plot(fpr_dict[i], tpr_dict[i], linewidth=1.5, label=f"{lbl} (AUC={auc_dict[f'class_{c}']:.3f})")
    ax.plot(fpr_micro, tpr_micro, linestyle="--", linewidth=1.5, label=f"micro (AUC={auc_dict['micro']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)

    return auc_dict

def analyze_feature_importance(
    model: BaseEstimator, feature_names: List[str], top_k: int = 20
    ) -> Optional[pd.DataFrame]:
    """
    For tree-based models with feature_importances_, return a DataFrame of top-k importances.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_k]
        df_imp = pd.DataFrame({
            "feature": [feature_names[i] for i in idx],
            "importance": imp[idx],
        })
        return df_imp
    return None

def analyze_feature_importance_shap(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    savepath_prefix: Optional[Path] = None,
    show_plot: bool = False,
    ) -> None:
    """
    Try SHAP summary (beeswarm) & bar plot. Save figures if savepath_prefix provided.
    Only saves files; does not display unless show_plot=True.
    """
    try:
        import shap  # optional dependency

        # Try tree explainer where possible, else Kernel (can be slow)
        explainer = None
        if hasattr(model, "predict_proba"):
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                warnings.warn("TreeExplainer failed; falling back to KernelExplainer (may be slow).")
                # Use a small background to be practical
                bg_idx = np.random.choice(X.shape[0], size=min(100, X.shape[0]), replace=False)
                explainer = shap.KernelExplainer(model.predict_proba, X[bg_idx])

        if explainer is None:
            return

        # SHAP values: for multiclass, returns list per class
        shap_values = explainer.shap_values(X, check_additivity=False)
        fig1 = plt.figure(figsize=(8, 5))
        if isinstance(shap_values, list):
            # Combine absolute values across classes for a global view
            sv_abs = np.sum([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(sv_abs, X, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        if savepath_prefix:
            plt.savefig(f"{savepath_prefix}_beeswarm.png", bbox_inches="tight", dpi=160)
        if show_plot:
            plt.show()
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8, 5))
        if isinstance(shap_values, list):
            sv_abs = np.sum([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(sv_abs, X, feature_names=feature_names, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        if savepath_prefix:
            plt.savefig(f"{savepath_prefix}_bar.png", bbox_inches="tight", dpi=160)
        if show_plot:
            plt.show()
        plt.close(fig2)

    except Exception as e:
        warnings.warn(f"SHAP analysis skipped: {e}")

def print_result(
    y_true: np.ndarray,
    y_pred_direct: np.ndarray,
    y_pred_stacked: np.ndarray,
    direct_accuracy: float,
    stacked_accuracy: float,
    class_names: Optional[List[str]] = None,
    file_path: Optional[Path] = None,
    ) -> None:
    """
    Text-only 결과 출력(간소화 버전):
    - Accuracy 및 개선폭
    - Confusion Matrix (Direct / Stacked)  ← sklearn.metrics.confusion_matrix 사용
    - Classification Report (precision/recall/f1)
    화면에 출력하고, 같은 내용을 file_path에도 append 저장.
    """
    p = Path(file_path) if file_path is not None else None

    # 라벨 순서 고정(보고서/행열 순서 일치)
    labels = np.unique(y_true)
    if class_names and len(class_names) == len(labels):
        label_names = list(class_names)
    else:
        label_names = [str(c) for c in labels]

    # --- Confusion Matrices (간단하게) ---
    cm_direct = confusion_matrix(y_true, y_pred_direct, labels=labels)
    cm_stacked = confusion_matrix(y_true, y_pred_stacked, labels=labels)

    df_cm_direct = pd.DataFrame(cm_direct, index=label_names, columns=label_names)
    df_cm_stacked = pd.DataFrame(cm_stacked, index=label_names, columns=label_names)

    # --- Classification Reports ---
    cr_direct = classification_report(y_true, y_pred_direct,
                                      target_names=label_names if len(label_names) == len(np.unique(y_true)) else None,
                                      digits=4)
    cr_stacked = classification_report(y_true, y_pred_stacked,
                                       target_names=label_names if len(label_names) == len(np.unique(y_true)) else None,
                                       digits=4)

    # --- Assemble text once ---
    lines = []
    lines.append("========== MODEL PERFORMANCE (TEST) ==========")
    lines.append(f"Direct Accuracy : {direct_accuracy:.6f}")
    lines.append(f"Stacked Accuracy: {stacked_accuracy:.6f}")
    lines.append(f"Absolute Improvement: {stacked_accuracy - direct_accuracy:+.6f}\n")

    lines.append("----- Confusion Matrix (Direct) -----")
    lines.append(df_cm_direct.to_string())
    lines.append("")

    lines.append("----- Confusion Matrix (Stacked) -----")
    lines.append(df_cm_stacked.to_string())
    lines.append("")

    lines.append("----- Classification Report (Direct) -----")
    lines.append(cr_direct)
    lines.append("----- Classification Report (Stacked) -----")
    lines.append(cr_stacked)

    text = "\n".join(lines)

    # 화면 출력
    print(text)

    # 파일 저장(append)
    if p:
        with p.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

def save_results(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_direct: np.ndarray,
    y_pred_stacked: np.ndarray,
    feature_importances: Optional[pd.DataFrame] = None,
    pred_csv_path: Optional[Path] = None,
    imp_csv_path: Optional[Path] = None,
    ) -> None:
    """
    Save predictions and optional feature importances to CSV files.
    """
    if pred_csv_path:
        df_pred = pd.DataFrame({
            "y_true": y_test,
            "y_pred_direct": y_pred_direct,
            "y_pred_stacked": y_pred_stacked,
        }, index=X_test.index if isinstance(X_test, pd.DataFrame) else None)
        
        df_pred.to_csv(Path(pred_csv_path), index=True)

    if feature_importances is not None and imp_csv_path:
        feature_importances.to_csv(Path(imp_csv_path), index=False)

def print_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    top_k: int = 20,
    model_name: str = "Direct Model",
    file_path: Optional[Path] = None,
    ) -> Optional[pd.DataFrame]:
    """
    Print top-k feature importances for the given model to console and append to results file.
    Returns the DataFrame of importances (or None).
    """
    imp_df = analyze_feature_importance(model, feature_names, top_k)
    if imp_df is not None:
        header = f"\n----- Top Feature Importances ({model_name}) -----"
        print(header)
        print(imp_df.to_string(index=False))
        if file_path:
            with Path(file_path).open("a", encoding="utf-8") as f:
                f.write(header + "\n")
                f.write(imp_df.to_string(index=False) + "\n")
    return imp_df

def print_roc_auc(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curve (Test - Stacked)",
    savepath: Optional[Path] = None,
    file_path: Optional[Path] = None,
    ) -> None:
    """
    Compute ROC AUC using plot_roc_auc and print AUC results to console and file.
    """
    try:
        auc_map = plot_roc_auc(model=model, X=X, y=y, class_names=class_names, title=title, savepath=savepath, show_plot=False)
        auc_round = {k: round(v, 6) for k, v in auc_map.items()}
        print(f"\n[AUC per class + micro] {auc_round}")
        if file_path:
            with Path(file_path).open("a", encoding="utf-8") as f:
                f.write(f"\n[AUC per class + micro] {auc_round}\n")
    except Exception as e:
        msg = f"[WARN] ROC/AUC skipped: {e}"
        print(msg)
        if file_path:
            with Path(file_path).open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

def perform_shap_analysis(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    savepath_prefix: Optional[Path] = None,
    file_path: Optional[Path] = None,
    ) -> None:
    """
    Run SHAP analysis for the model (using analyze_feature_importance_shap) and log results.
    """
    try:
        analyze_feature_importance_shap(
            model=model,
            X=X,
            feature_names=feature_names,
            savepath_prefix=savepath_prefix,
            show_plot=False,
        )
        info_msg = (
            "[INFO] SHAP plots saved: "
            f"{Path(str(savepath_prefix))}_beeswarm.png, "
            f"{Path(str(savepath_prefix))}_bar.png"
        )
        print(info_msg)
        if file_path:
            with Path(file_path).open("a", encoding="utf-8") as f:
                f.write(info_msg + "\n")
    except Exception as e:
        warn_msg = f"[WARN] SHAP analysis skipped: {e}"
        print(warn_msg)
        if file_path:
            with Path(file_path).open("a", encoding="utf-8") as f:
                f.write(warn_msg + "\n")

# --------------------------------------------------------------------------------------
# Main (project-specific wiring only)
# --------------------------------------------------------------------------------------
def main():

    # 1) 출력 디렉토리 및 RUN_ID 설정
    output_dir = Path("./output_titanic")
    output_dir.mkdir(exist_ok=True)
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2) 모든 저장 경로를 main()에서 일괄 정의
    results_path = output_dir / f"results_{RUN_ID}.txt"
    roc_savepath = output_dir / f"roc_curve_test_{RUN_ID}.png"
    shap_prefix = output_dir / f"shap_test_{RUN_ID}"
    pred_csv = output_dir / f"stacking_predictions_{RUN_ID}.csv"
    imp_csv = output_dir / f"feature_importances_{RUN_ID}.csv"

    # 3) Load & prepare Titanic data (project-specific pieces)
    titanic = load_and_explore_data()
    titanic_processed = preprocess_titanic_data(titanic)
    titanic_features = create_features(titanic_processed)

    # 4) Create a generic multi-class target (reusable function)
    #    You can customize risk_features/weights per project without code changes.
    default_weights = None  # equal weights by default
    titanic_multilabel, info = create_multilabel_target(
        titanic_features,
        risk_features=None,          # will auto-pick from available columns
        weights=default_weights,     # equal weighting
        n_classes=5,
        new_col="risk_class",
        percentiles=None,            # evenly spaced
    )

    # 5) Select features & target
    target_col = "risk_class"  # project-specific choice
    feature_cols = [c for c in titanic_multilabel.columns if c not in {"survived", target_col, "_risk_score"}]
    X = titanic_multilabel[feature_cols].astype(float).values
    y = titanic_multilabel[target_col].values

    # Optional class names (for reports/plots)
    # Provide generic ordinal labels 0..K-1
    classes = np.unique(y)
    class_names = [f"class_{int(c)}" for c in classes]

    # 6) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 7) Evaluate direct & stacked models
    results = evaluate_models(
        X_train, X_test, y_train, y_test,
        direct_model=RandomForestClassifier(n_estimators=400, random_state=42)
        # stacked_model will use default StackingClassifier
    )
    direct_model = results["direct_model"]
    stacked_model = results["stacked_model"]
    y_pred_direct = results["y_pred_direct"]
    y_pred_stacked = results["y_pred_stacked"]
    direct_accuracy = results["direct_accuracy"]
    stacked_accuracy = results["stacked_accuracy"]

    # 8) Print metrics and reports (to console and file)
    print_result(
        y_true=y_test,
        y_pred_direct=y_pred_direct,
        y_pred_stacked=y_pred_stacked,
        direct_accuracy=direct_accuracy,
        stacked_accuracy=stacked_accuracy,
        class_names=class_names,
        file_path=results_path,
    )

    # 9) Feature importance for direct model
    imp_df = print_feature_importance(
        direct_model, feature_cols, top_k=25, model_name="Direct Model",
        file_path=results_path,
    )

    # 10) ROC/AUC for the stacked model (save figure; also print AUC values)
    print_roc_auc(
        model=stacked_model,
        X=X_test,
        y=y_test,
        class_names=class_names,
        savepath=roc_savepath,
        file_path=results_path,
    )

    # 11) SHAP analysis for stacked model (figures saved to files)
    perform_shap_analysis(
        model=stacked_model,
        X=X_test,
        feature_names=feature_cols,
        savepath_prefix=shap_prefix,
        file_path=results_path,
    )

    # 12) Save predictions & importances to CSV under ./output
    save_results(
        X_test=pd.DataFrame(X_test, columns=feature_cols),
        y_test=y_test,
        y_pred_direct=y_pred_direct,
        y_pred_stacked=y_pred_stacked,
        feature_importances=imp_df,
        pred_csv_path=pred_csv,
        imp_csv_path=imp_csv,
    )

    print(f"\n[INFO] All outputs saved under {output_dir.resolve()} (unique RUN_ID={RUN_ID}).")

if __name__ == "__main__":
    main()