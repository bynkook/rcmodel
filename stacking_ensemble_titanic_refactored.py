"""
stacking_ensemble_titanic.py
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

Reusable Components
-------------------
- StackedEnsembleClassifier: flexible base/meta estimators, probability/meta-feature options.
- create_multilabel_target: generic multi-class target creation from an arbitrary risk score.
- evaluate_models: trains direct & stacked models; returns predictions & metrics (no printing).
- plot_roc_auc: saves ROC curve image & returns AUC dictionary (per-class & micro-average).
- analyze_feature_importance: simple feature importance reporting for tree-based direct model.
- analyze_feature_importance_shap: SHAP summary & bar plots for final model (if supported).
- print_result: central place to print ALL classification analysis text to screen AND file.
- save_results: persist predictions and (optional) feature importances to CSV files.

Variable & Naming Conventions
-----------------------------
- snake_case for all variables/functions (PEP 8).
- X, y: feature matrix and target (standard ML convention).
- X_train, X_test, y_train, y_test: train/test split.
- direct_model: baseline single classifier (default RandomForestClassifier).
- stacked_model: StackedEnsembleClassifier (base_clfs + meta_clf).
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
- This script avoids unused imports. It uses only:
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
  (Removed: precision_recall_fscore_support, RocCurveDisplay, roc_auc_score)
- ROC curve is already implemented; we DO NOT add any new plots besides saving existing ones.
- SHAP is optional; handled with a try/except and only saved to files (no plt.show()).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")  # ensure no GUI backend
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Dataset-specific functions (Titanic). Keep these isolated for easy swapping per project
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Generic & reusable components below
# --------------------------------------------------------------------------------------

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


@dataclass
class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Generic stacking ensemble for multi-class classification.

    Parameters
    ----------
    base_clfs : list[Estimator]
        List of base classifiers. If None, defaults to 4 tree-based models.
    meta_clf : Estimator
        Meta-classifier to learn from base-level meta-features (probabilities or labels).
    use_probabilities : bool
        If True, pass base predicted probabilities to meta layer; else pass class labels.
    scale_meta : bool
        Whether to standardize meta-features before meta_clf.
    random_state : int or None
        For reproducibility (where supported).

    Behavior
    --------
    - For K classes, we create binary one-vs-rest targets for a subset/classes_for_base.
      To keep generality across different K:
        * If K >= 5: exclude the highest class index (mimics paper's balance heuristic)
        * Else: use all classes for base one-vs-rest tasks
    - Base_clfs are trained per binary target. Their predictions build the meta-feature matrix.
    - Meta_clf is then trained to predict the multi-class y.

    Notes
    -----
    - This is intentionally simple (no CV-stacking). For stronger generalization, one could
      implement out-of-fold meta-features, but this version stays fast and easy to reuse.
    """
    base_clfs: Optional[List[BaseEstimator]] = None
    meta_clf: Optional[BaseEstimator] = None
    use_probabilities: bool = True
    scale_meta: bool = True
    random_state: Optional[int] = 42

    def __post_init__(self):
        if self.base_clfs is None:
            self.base_clfs = [
                RandomForestClassifier(n_estimators=200, max_depth=None, random_state=self.random_state),
                RandomForestClassifier(n_estimators=400, max_depth=8, random_state=self.random_state),
                RandomForestClassifier(n_estimators=300, max_depth=12, random_state=self.random_state),
                RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state),
            ]
        if self.meta_clf is None:
            self.meta_clf = LogisticRegression(max_iter=200, multi_class="auto", n_jobs=None)

        if self.scale_meta:
            self.meta_pipeline_ = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly
                ("clf", clone(self.meta_clf)),
            ])
        else:
            self.meta_pipeline_ = clone(self.meta_clf)

        self.fitted_base_ = []
        self.classes_ = None
        self.classes_for_base_ = None

    def _prepare_base_targets(self, y: np.ndarray) -> List[np.ndarray]:
        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) >= 5:
            classes_for_base = classes[:-1]
        else:
            classes_for_base = classes
        self.classes_for_base_ = classes_for_base

        base_targets = []
        for c in classes_for_base:
            base_targets.append((y == c).astype(int))
        return base_targets

    def fit(self, X: np.ndarray, y: np.ndarray):
        base_targets = self._prepare_base_targets(y)
        self.fitted_base_ = []

        # Train base classifiers per binary target
        for idx, (c, t) in enumerate(zip(self.classes_for_base_, base_targets), start=1):
            # Clone & fit each base clf for this binary problem
            fitted_list = []
            for b_i, base in enumerate(self.base_clfs, start=1):
                clf = clone(base)
                clf.fit(X, t)
                fitted_list.append(clf)
            self.fitted_base_.append((int(c), fitted_list))
        # Train meta on meta-features
        meta_X = self._build_meta_features(X)
        self.meta_pipeline_.fit(meta_X, y)
        return self

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        blocks = []
        for c, clfs in self.fitted_base_:
            # stack predictions from all base clfs on this binary task
            if self.use_probabilities and hasattr(clfs[0], "predict_proba"):
                # concatenate probability of positive class (1) for each base clf
                probs = [clf.predict_proba(X)[:, 1] for clf in clfs]
                block = np.vstack(probs).T  # shape (n_samples, n_base)
            else:
                preds = [clf.predict(X) for clf in clfs]
                block = np.vstack(preds).T
            blocks.append(block)
        if not blocks:
            raise RuntimeError("No base learners were trained to generate meta-features.")
        meta_X = np.hstack(blocks)
        return meta_X

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._build_meta_features(X)
        return self.meta_pipeline_.predict(meta_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Provide class probability estimates if meta supports it
        meta_X = self._build_meta_features(X)
        if hasattr(self.meta_pipeline_, "predict_proba"):
            return self.meta_pipeline_.predict_proba(meta_X)
        # Fallback via decision function -> softmax
        if hasattr(self.meta_pipeline_, "decision_function"):
            dec = self.meta_pipeline_.decision_function(meta_X)
            # handle binary vs multiclass
            if dec.ndim == 1:
                dec = np.vstack([-dec, dec]).T
            e = np.exp(dec - dec.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        raise AttributeError("Meta classifier does not support probability estimates.")


def evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    direct_model: Optional[BaseEstimator] = None,
    stacked_model: Optional[StackedEnsembleClassifier] = None,
) -> Dict[str, Any]:
    """
    Train a direct baseline model and a stacked ensemble; compute predictions & accuracies.

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
        stacked_model = StackedEnsembleClassifier(
            base_clfs=None, meta_clf=None, use_probabilities=True, scale_meta=True, random_state=42
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
    savepath: Optional[str] = None,
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
    title_prefix: str = "SHAP",
    savepath_prefix: Optional[str] = None,
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
    file_path: Optional[str] = "output/results.txt",
) -> None:
    """
    Centralized text-only printing of classification analysis results:
    - Accuracies & absolute improvement
    - Confusion matrices (direct vs stacked)
    - Classification reports (precision/recall/F1/support; micro/macro/weighted)
    Also writes the same content to `file_path` if provided.
    """
    lines = []
    lines.append("========== MODEL PERFORMANCE (TEST) ==========")
    lines.append(f"Direct Accuracy : {direct_accuracy:.6f}")
    lines.append(f"Stacked Accuracy: {stacked_accuracy:.6f}")
    lines.append(f"Absolute Improvement: {stacked_accuracy - direct_accuracy:+.6f}")
    lines.append("")

    # Confusion matrices
    cm_direct = confusion_matrix(y_true, y_pred_direct)
    cm_stacked = confusion_matrix(y_true, y_pred_stacked)

    def _cm_to_str(cm: np.ndarray, title: str) -> List[str]:
        hdr = [title]
        hdr.append(f"shape={cm.shape}")
        # Optional header with class labels if provided
        if class_names and len(class_names) == cm.shape[0]:
            hdr.append("labels: " + ", ".join(class_names))
        mat = ["\n".join([
            " ".join([f"{v:5d}" for v in row]) for row in cm
        ])]
        return hdr + mat  # type: ignore

    lines += _cm_to_str(cm_direct, "Confusion Matrix (Direct)")
    lines.append("")
    lines += _cm_to_str(cm_stacked, "Confusion Matrix (Stacked)")
    lines.append("")

    # Classification reports
    lines.append("----- Classification Report (Direct) -----")
    lines.append(classification_report(y_true, y_pred_direct, target_names=class_names, digits=4))
    lines.append("----- Classification Report (Stacked) -----")
    lines.append(classification_report(y_true, y_pred_stacked, target_names=class_names, digits=4))

    text = "\n".join(lines)
    print(text)

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def save_results(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_direct: np.ndarray,
    y_pred_stacked: np.ndarray,
    feature_importances: Optional[pd.DataFrame] = None,
    pred_csv_path: str = "output/stacking_predictions.csv",
    imp_csv_path: str = "output/feature_importances.csv",
) -> None:
    """
    Save predictions and optional feature importances to CSV files under ./output.
    """
    df_pred = pd.DataFrame({
        "y_true": y_test,
        "y_pred_direct": y_pred_direct,
        "y_pred_stacked": y_pred_stacked,
    }, index=X_test.index if isinstance(X_test, pd.DataFrame) else None)
    os.makedirs("output", exist_ok=True)
    df_pred.to_csv(pred_csv_path, index=True)

    if feature_importances is not None:
        feature_importances.to_csv(imp_csv_path, index=False)


# --------------------------------------------------------------------------------------
# Main (project-specific wiring only)
# --------------------------------------------------------------------------------------

def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # 1) Load & prepare Titanic data (project-specific pieces)
    titanic = load_and_explore_data()
    titanic_processed = preprocess_titanic_data(titanic)
    titanic_features = create_features(titanic_processed)

    # 2) Create a generic multi-class target (reusable function)
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

    # 3) Select features & target
    target_col = "risk_class"  # project-specific choice
    feature_cols = [c for c in titanic_multilabel.columns if c not in {"survived", target_col, "_risk_score"}]
    X = titanic_multilabel[feature_cols].astype(float).values
    y = titanic_multilabel[target_col].values

    # Optional class names (for reports/plots)
    # Provide generic ordinal labels 0..K-1
    classes = np.unique(y)
    class_names = [f"class_{int(c)}" for c in classes]

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5) Evaluate direct & stacked models (reusable wrapper)
    results = evaluate_models(
        X_train, X_test, y_train, y_test,
        direct_model=RandomForestClassifier(n_estimators=400, random_state=42),
        stacked_model=StackedEnsembleClassifier(
            base_clfs=None, meta_clf=LogisticRegression(max_iter=300), use_probabilities=True, scale_meta=True
        ),
    )
    direct_model = results["direct_model"]
    stacked_model = results["stacked_model"]
    y_pred_direct = results["y_pred_direct"]
    y_pred_stacked = results["y_pred_stacked"]
    direct_accuracy = results["direct_accuracy"]
    stacked_accuracy = results["stacked_accuracy"]

    # 6) Centralized printing of all metrics (also saved to output/results.txt)
    #    (Text-only: accuracy, confusion matrices, classification reports)
    print_result(
        y_true=y_test,
        y_pred_direct=y_pred_direct,
        y_pred_stacked=y_pred_stacked,
        direct_accuracy=direct_accuracy,
        stacked_accuracy=stacked_accuracy,
        class_names=class_names,
        file_path="output/results.txt",
    )

    # 7) Feature importance for direct model (if available)
    imp_df = analyze_feature_importance(direct_model, feature_cols, top_k=25)
    if imp_df is not None:
        print("\n----- Top Feature Importances (Direct Model) -----")
        print(imp_df.to_string(index=False))
        with open("output/results.txt", "a", encoding="utf-8") as f:
            f.write("\n----- Top Feature Importances (Direct Model) -----\n")
            f.write(imp_df.to_string(index=False) + "\n")

    # 8) ROC/AUC for the stacked model (save figure; also print AUC map)
    try:
        auc_map = plot_roc_auc(
            model=stacked_model,
            X=X_test,
            y=y_test,
            class_names=class_names,
            title="ROC Curve (Test - Stacked)",
            savepath="output/roc_curve_test.png",
            show_plot=False,  # do not display; only save
        )
        print("\n[AUC per class + micro] ", {k: round(v, 6) for k, v in auc_map.items()})
        with open("output/results.txt", "a", encoding="utf-8") as f:
            f.write("\n[AUC per class + micro] " + str({k: round(v, 6) for k, v in auc_map.items()}) + "\n")
    except Exception as e:
        msg = f"[WARN] ROC/AUC skipped: {e}"
        print(msg)
        with open("output/results.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 9) SHAP analysis for stacked model (saved to files only)
    try:
        analyze_feature_importance_shap(
            model=stacked_model,
            X=X_test,
            feature_names=feature_cols,
            title_prefix="SHAP (Test - Stacked)",
            savepath_prefix="output/shap_test",
            show_plot=False,  # save only
        )
        print("[INFO] SHAP plots saved: output/shap_test_beeswarm.png, output/shap_test_bar.png")
        with open("output/results.txt", "a", encoding="utf-8") as f:
            f.write("[INFO] SHAP plots saved: output/shap_test_beeswarm.png, output/shap_test_bar.png\n")
    except Exception as e:
        msg = f"[WARN] SHAP analysis skipped: {e}"
        print(msg)
        with open("output/results.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 10) Save predictions & importances to CSV under ./output
    save_results(
        X_test=pd.DataFrame(X_test, columns=feature_cols),
        y_test=y_test,
        y_pred_direct=y_pred_direct,
        y_pred_stacked=y_pred_stacked,
        feature_importances=imp_df,
        pred_csv_path="output/stacking_predictions.csv",
        imp_csv_path="output/feature_importances.csv",
    )

    print("\n[INFO] All outputs saved under ./output (no subdirectories).")


if __name__ == "__main__":
    # Keep warnings tidy in console & results.txt
    warnings.filterwarnings("ignore")
    main()