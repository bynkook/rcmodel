# 스태킹 앙상블 및 설명가능한 AI를 활용한 대금지급지연 예측 - 논문 구현 코드
# 타이타닉 데이터셋 적용

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, RocCurveDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import cycle
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 데이터 로드 및 기본 정보 확인
# ============================================================================

def load_and_explore_data():
    """타이타닉 데이터셋 로드 및 기본 탐색"""
    # 타이타닉 데이터셋 로드
    titanic = sns.load_dataset('titanic')
    
    print("=== 타이타닉 데이터셋 기본 정보 ===")
    print(f"데이터 형태: {titanic.shape}")
    print(f"\n컬럼 정보:")
    print(titanic.info())
    
    print(f"\n결측값 확인:")
    print(titanic.isnull().sum())
    
    print(f"\n생존률 분포:")
    print(titanic['survived'].value_counts(normalize=True))
    
    return titanic

# ============================================================================
# 2. 데이터 전처리 (논문 방식 적용)
# ============================================================================

def preprocess_titanic_data(df):
    """
    논문의 전처리 방식을 타이타닉 데이터에 적용
    - 결측값을 'Missing' 문자열로 대체
    - 범주형 변수 처리
    """
    data = df.copy()
    
    # 범주형 컬럼을 일반 문자열로 변환 후 결측값 처리
    categorical_columns = ['embarked', 'embark_town', 'deck']
    for col in categorical_columns:
        if col in data.columns:
            # 범주형을 문자열로 변환한 후 결측값 처리
            data[col] = data[col].astype(str)
            data[col] = data[col].replace('nan', 'Missing')
            data[col] = data[col].fillna('Missing')
    
    # 수치형 결측값은 평균으로 대체
    data['age'] = data['age'].fillna(data['age'].mean())
    
    # 불필요한 컬럼 제거 (중복 정보)
    drop_cols = ['alive', 'who', 'class']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    return data

# ============================================================================
# 3. 특징 엔지니어링 (논문의 고객/송장 레벨 특징 모방)
# ============================================================================

def create_features(df):
    """
    논문의 특징 엔지니어링 방식을 타이타닉에 적용
    고객 레벨 특징과 개별 레코드 특징을 생성
    """
    data = df.copy()
    
    # 고객 레벨 특징 (pclass를 고객 그룹으로 가정)
    customer_features = data.groupby('pclass').agg({
        'fare': ['mean', 'std', 'count'],
        'survived': ['mean', 'count'],
        'age': ['mean', 'std']
    }).round(3)
    
    customer_features.columns = ['_'.join(col) for col in customer_features.columns]
    customer_features = customer_features.add_prefix('customer_')
    
    # 원본 데이터에 고객 레벨 특징 결합
    data = data.merge(customer_features, left_on='pclass', right_index=True, how='left')
    
    # 추가 파생 변수 생성 (논문의 상호작용 특징 모방)
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    data['is_alone'] = (data['family_size'] == 1).astype(int)
    data['fare_per_person'] = data['fare'] / data['family_size']
    
    # 0으로 나누기 방지
    data['fare_ratio'] = np.where(data['customer_fare_mean'] > 0, 
                                  data['fare'] / data['customer_fare_mean'], 0)
    
    # 범주형 변수 인코딩
    le_dict = {}
    categorical_cols = ['sex', 'embarked', 'embark_town', 'deck']
    for col in categorical_cols:
        if col in data.columns:
            le_dict[col] = LabelEncoder()
            data[f'{col}_encoded'] = le_dict[col].fit_transform(data[col].astype(str))
    
    return data, le_dict

# ============================================================================
# 4. 다중 클래스 타겟 생성 (논문의 5개 지연 구간 모방)
# ============================================================================

def create_multilabel_target(df, target_col='survived'):
    """
    논문처럼 다중 클래스 타겟을 생성
    생존률을 기반으로 5개 구간으로 나누어 다중 클래스 문제로 변환
    """
    data = df.copy()
    
    # 생존 확률을 기반으로 리스크 레벨 생성 (논문의 지연 기간 모방)
    risk_features = ['age', 'fare', 'pclass', 'family_size', 'customer_survived_mean']
    
    # 리스크 스코어 계산 (정규화된 가중합)
    scaler = StandardScaler()
    risk_data = scaler.fit_transform(data[risk_features])
    
    # 가중치 (나이는 높을수록, 요금은 낮을수록, 클래스는 높을수록 위험)
    weights = np.array([0.3, -0.2, 0.3, 0.1, -0.1])  
    risk_score = np.dot(risk_data, weights)
    
    # 리스크 스코어를 5개 구간으로 나누기 (논문의 5개 지연 구간 모방)
    risk_percentiles = np.percentile(risk_score, [20, 40, 60, 80])
    
    def assign_risk_class(score):
        if score <= risk_percentiles[0]:
            return 0  # Very Low Risk (논문의 'paid on time')
        elif score <= risk_percentiles[1]:
            return 1  # Low Risk (논문의 'gt_5_lt_30')
        elif score <= risk_percentiles[2]:
            return 2  # Medium Risk (논문의 'gt_30_lt_60')
        elif score <= risk_percentiles[3]:
            return 3  # High Risk (논문의 'gt_60_lt_90')
        else:
            return 4  # Very High Risk (논문의 'gt_90')
    
    data['risk_class'] = np.array([assign_risk_class(score) for score in risk_score])
    
    return data, risk_percentiles

# ============================================================================
# 5. 스태킹 앙상블 분류기 구현 (논문의 2층 구조)
# ============================================================================

class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    논문에서 제안한 2층 스태킹 앙상블 분류기
    Base Layer: 4개의 One-vs-Rest 이진 분류기
    Meta Layer: Random Forest 최종 분류기
    """
    
    def __init__(self, base_classifiers=None, meta_classifier=None, use_probabilities=True):
        self.base_classifiers = base_classifiers if base_classifiers else [
            RandomForestClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42),
            ExtraTreesClassifier(n_estimators=50, random_state=42),
        ]
        self.meta_classifier = meta_classifier if meta_classifier else RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.use_probabilities = use_probabilities
        self.classes_ = None
        
    def create_base_targets(self, y):
        """
        논문의 4개 이진 분류 타겟 생성 (lt_X_gt_X 방식)
        각 클래스에 대해 One-vs-Rest 방식으로 이진 분류 타겟 생성
        """
        base_targets = {}
        
        # 각 클래스에 대해 One-vs-Rest 이진 분류 타겟 생성
        for class_idx in [0, 1, 2, 3]:  # 0, 1, 2, 3번 클래스 (4번은 제외하여 균형맞춤)
            name = f'class_{class_idx}_vs_rest'
            base_targets[name] = (y == class_idx).astype(int)
            
        return base_targets
    
    def fit(self, X, y):
        """스태킹 앙상블 학습"""
        self.classes_ = np.unique(y)
        
        # Base classifiers를 위한 이진 분류 타겟 생성
        base_targets = self.create_base_targets(y)
        
        # Base classifiers 학습 및 예측
        base_predictions = []
        self.fitted_base_classifiers = []
        
        for i, (name, target) in enumerate(base_targets.items()):
            print(f"Training base classifier {i+1}/4: {name}")
            print(f"  Class distribution: {np.bincount(target)}")
            
            # 모든 샘플이 하나의 클래스에만 속하는 경우 스킵
            if len(np.unique(target)) < 2:
                print(f"  Skipping {name} - only one class present")
                continue
                
            # 각 base classifier 학습
            clf_idx = i % len(self.base_classifiers)  # 분류기 순환 사용
            clf_params = self.base_classifiers[clf_idx].get_params()
            clf = type(self.base_classifiers[clf_idx])(**clf_params)
            clf.fit(X, target)
            self.fitted_base_classifiers.append(clf)
            
            # Cross-validation으로 meta-features 생성
            pred_labels = cross_val_predict(clf, X, target, cv=3, method='predict')
            
            if self.use_probabilities and hasattr(clf, 'predict_proba'):
                pred_proba = cross_val_predict(clf, X, target, cv=3, method='predict_proba')
                # 양성 클래스 확률만 사용
                if pred_proba.shape[1] == 2:
                    base_pred = np.column_stack([pred_labels.reshape(-1, 1), pred_proba[:, 1].reshape(-1, 1)])
                else:
                    base_pred = pred_labels.reshape(-1, 1)
            else:
                base_pred = pred_labels.reshape(-1, 1)
            
            base_predictions.append(base_pred)
        
        # Meta features 생성
        if base_predictions:
            meta_features = np.hstack(base_predictions)
            print(f"Meta features shape: {meta_features.shape}")
            
            # Meta classifier 학습
            self.meta_classifier.fit(meta_features, y)
        else:
            raise ValueError("No valid base classifiers were trained")
        
        return self
    
    def predict(self, X):
        """예측 수행"""
        # Base classifiers로 예측
        base_predictions = []
        
        for clf in self.fitted_base_classifiers:
            pred_labels = clf.predict(X)
            
            if self.use_probabilities and hasattr(clf, 'predict_proba'):
                pred_proba = clf.predict_proba(X)
                if pred_proba.shape[1] == 2:
                    base_pred = np.column_stack([pred_labels.reshape(-1, 1), pred_proba[:, 1].reshape(-1, 1)])
                else:
                    base_pred = pred_labels.reshape(-1, 1)
            else:
                base_pred = pred_labels.reshape(-1, 1)
                
            base_predictions.append(base_pred)
        
        # Meta features 생성
        meta_features = np.hstack(base_predictions)
        
        # Meta classifier로 최종 예측
        return self.meta_classifier.predict(meta_features)

# ============================================================================
# 6. 성능 평가 및 비교
# ============================================================================

def evaluate_models(X_train, X_test, y_train, y_test):
    """
    직접 분류와 스태킹 앙상블 성능 비교
    """
    
    print("=== 모델 학습 및 평가 ===")
    
    # 1. 직접 분류 (Single-layer) - 논문의 baseline
    print("\n1. 직접 분류 (Baseline)")
    direct_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    direct_classifier.fit(X_train, y_train)
    y_pred_direct = direct_classifier.predict(X_test)
    direct_accuracy = accuracy_score(y_test, y_pred_direct)
    
    print(f"직접 분류 정확도: {direct_accuracy:.3f}")
    
    # 2. 스태킹 앙상블 (Two-layer) - 논문의 제안 방법
    print("\n2. 스태킹 앙상블 (제안 방법)")
    stacked_classifier = StackedEnsembleClassifier(use_probabilities=True)
    stacked_classifier.fit(X_train, y_train)
    y_pred_stacked = stacked_classifier.predict(X_test)
    stacked_accuracy = accuracy_score(y_test, y_pred_stacked)
    
    print(f"\n스태킹 앙상블 정확도: {stacked_accuracy:.3f}")
    
    # 성능 비교
    print(f"\n=== 성능 비교 ===")
    print(f"직접 분류 정확도:     {direct_accuracy:.3f}")
    print(f"스태킹 앙상블 정확도: {stacked_accuracy:.3f}")
    print(f"개선 정도:           {(stacked_accuracy - direct_accuracy)*100:+.1f}%p")
    
    # 상세 분류 리포트
    target_names = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    
    print(f"\n=== 직접 분류 상세 결과 ===")
    print(classification_report(y_test, y_pred_direct, target_names=target_names))
    
    print(f"\n=== 스태킹 앙상블 상세 결과 ===")
    print(classification_report(y_test, y_pred_stacked, target_names=target_names))
    
    return direct_classifier, stacked_classifier, y_pred_direct, y_pred_stacked, direct_accuracy, stacked_accuracy

# ============================================================================
# 7. 특성 중요도 분석 (논문의 SHAP 대체)
# ============================================================================

def analyze_feature_importance(model, X, feature_names):
    """
    특성 중요도 분석 (논문의 SHAP 기능 대체)
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        
        # 중요도 순으로 정렬
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("=== 특성 중요도 분석 (논문의 SHAP 대체) ===")
        print("상위 10개 중요 특성:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:20s}: {row['importance']:.3f}")
        
        return importance_df
    else:
        print("해당 모델은 feature_importances_ 속성을 지원하지 않습니다.")
        return None

# ROC, AUC
def plot_roc_auc(model, X, y, class_names=None, title='ROC Curve', savepath=None):
    """
    Binary/Multiclass ROC curve and AUC.
    Parameters
    ----------
    model : fitted estimator with predict_proba or decision_function
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    class_names : list[str] | None
    title : str
    savepath : str | None
    Returns
    -------
    dict : {'micro': auc or None, <class_name>: auc, ...}
    """
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = len(classes)
    if class_names is None:
        class_names = [str(c) for c in classes]

    # score matrix
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if n_classes == 2:
            # ensure shape (n_samples,)
            scores = scores[:, 1] if scores.ndim == 2 else scores.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if n_classes == 2 and scores.ndim > 1:
            scores = scores.ravel()
    else:
        raise ValueError("Estimator must implement predict_proba or decision_function.")

    fig, ax = plt.subplots(figsize=(7, 5))
    auc_map = {}
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y, scores, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        auc_map[class_names[1]] = roc_auc
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
    else:
        y_bin = label_binarize(y, classes=classes)
        if scores.ndim == 1:
            raise ValueError("For multiclass ROC, score must be 2D with class probabilities or decision scores.")
        fpr, tpr = {}, {}
        for i, cname in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], scores[:, i])
            auc_i = auc(fpr[i], tpr[i])
            auc_map[cname] = auc_i
            ax.plot(fpr[i], tpr[i], label=f"{cname} AUC = {auc_i:.4f}")
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), scores.ravel())
        auc_micro = auc(fpr["micro"], tpr["micro"])
        auc_map["micro"] = auc_micro
        ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average AUC = {auc_micro:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
    # ensure nothing is clipped on the canvas
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()  # adjust paddings
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.show()
    return auc_map

# SHAP 분석
def analyze_feature_importance_shap(model, X, feature_names=None, class_idx=None, max_display=20, title_prefix="SHAP", savepath_prefix=None):
    """
    SHAP analysis and plotting without touching existing analyze_feature_importance().
    Parameters
    ----------
    model : fitted estimator
    X : pandas.DataFrame or np.ndarray
    feature_names : list[str] | None
    class_idx : int | None  # for classification, which class to visualize
    max_display : int
    title_prefix : str
    savepath_prefix : str | None  # if given, save figures as f"{savepath_prefix}_beeswarm.png" and f"{savepath_prefix}_bar.png"
    Returns
    -------
    shap_values : shap.Explanation or np.ndarray (implementation-dependent)
    """
    if shap is None:
        raise ImportError("shap is not installed. Please `pip install shap`.")

    # Prepare data and names
    if hasattr(X, "values") and hasattr(X, "columns"):
        data = X
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        data = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(data.shape[1])]

    # Prefer TreeExplainer when possible, else fallback
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)

    sv = explainer(data)

    # Select class for multi-output classification if needed
    # Newer SHAP returns Explanation with potentially 3D values; slice to 2D.
    try:
        values = sv.values
    except Exception:
        values = np.asarray(sv)

    selected = sv
    try:
        if values.ndim == 3:
            k = class_idx if class_idx is not None else (1 if values.shape[2] > 1 else 0)
            base_vals = getattr(sv, "base_values", None)
            base_vals_k = base_vals[:, k] if isinstance(base_vals, np.ndarray) and base_vals.ndim == 2 else base_vals
            selected = shap.Explanation(
                values=values[:, :, k],
                base_values=base_vals_k,
                data=getattr(sv, "data", data),
                feature_names=feature_names,
            )
    except Exception:
        # Fall back silently; plots below will attempt with `sv` as-is.
        selected = sv

    # Beeswarm plot
    try:
        shap.plots.beeswarm(selected, max_display=max_display, show=False)
    except Exception:
        shap.summary_plot(getattr(selected, "values", selected), data, feature_names=feature_names, show=False, max_display=max_display)
    if title_prefix:
        plt.gcf().suptitle(f"{title_prefix} - beeswarm", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if savepath_prefix:
        plt.savefig(f"{savepath_prefix}_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.show()

    # Bar plot (mean |SHAP|)
    try:
        shap.plots.bar(selected, max_display=max_display, show=False)
        if title_prefix:
            plt.gcf().suptitle(f"{title_prefix} - bar", y=0.98)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if savepath_prefix:
            plt.savefig(f"{savepath_prefix}_bar.png", bbox_inches="tight", dpi=150)
        plt.show()
    except Exception:
        # Manual bar as fallback
        vals = getattr(selected, "values", selected)
        if hasattr(vals, "shape") and vals.ndim == 2:
            mean_abs = np.abs(vals).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:max_display]
            fig = plt.figure(figsize=(8, 5))
            plt.bar(range(len(order)), mean_abs[order])
            plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=60, ha="right")
            plt.ylabel("mean(|SHAP value|)")
            plt.title(f"{title_prefix} - bar")
            fig.tight_layout()
            if savepath_prefix:
                plt.savefig(f"{savepath_prefix}_bar.png", bbox_inches="tight", dpi=150)
            plt.show()
    return sv


# ============================================================================
# 8. 결과 저장
# ============================================================================

def save_results(y_test, y_pred_direct, y_pred_stacked, importance_df, direct_accuracy, stacked_accuracy):
    """결과를 CSV 파일로 저장"""
    
    # 1. 예측 결과 저장
    results_df = pd.DataFrame({
        'actual_risk_class': y_test,
        'direct_prediction': y_pred_direct,
        'stacked_prediction': y_pred_stacked,
        'direct_correct': (y_test == y_pred_direct).astype(int),
        'stacked_correct': (y_test == y_pred_stacked).astype(int)
    })
    results_df.to_csv('titanic_stacking_predictions.csv', index=False)
    
    # 2. 특성 중요도 저장
    if importance_df is not None:
        importance_df.to_csv('titanic_feature_importance.csv', index=False)
    
    # 3. 모델 성능 비교 저장
    performance_df = pd.DataFrame({
        'Model': ['Direct Classification', 'Stacked Ensemble'],
        'Accuracy': [direct_accuracy, stacked_accuracy],
        'Improvement_vs_Direct': [0, stacked_accuracy - direct_accuracy]
    })
    performance_df.to_csv('titanic_model_performance.csv', index=False)
    
    print("\n=== CSV 파일 저장 완료 ===")
    print("1. titanic_stacking_predictions.csv - 예측 결과")
    print("2. titanic_feature_importance.csv - 특성 중요도")  
    print("3. titanic_model_performance.csv - 모델 성능 비교")

# ============================================================================
# 9. 메인 실행 함수
# ============================================================================

def main():
    """논문 구현의 전체 파이프라인 실행"""
    
    print("="*80)
    print("스태킹 앙상블 및 설명가능한 AI를 활용한 대금지급지연 예측")
    print("논문 방법론 구현 및 타이타닉 데이터 적용")
    print("="*80)
    
    # 1. 데이터 로드
    titanic = load_and_explore_data()
    
    # 2. 전처리
    print("\n" + "="*50)
    print("데이터 전처리 중...")
    titanic_processed = preprocess_titanic_data(titanic)
    
    # 3. 특징 엔지니어링
    print("특징 엔지니어링 중...")
    titanic_features, label_encoders = create_features(titanic_processed)
    
    # 4. 다중 클래스 타겟 생성
    print("다중 클래스 타겟 생성 중...")
    titanic_multilabel, risk_thresholds = create_multilabel_target(titanic_features)
    
    # 5. 특징 선택 및 데이터 준비
    feature_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone', 
                       'fare_per_person', 'fare_ratio', 'customer_fare_mean', 'customer_fare_std',
                       'customer_survived_mean', 'sex_encoded', 'embarked_encoded', 
                       'embark_town_encoded', 'deck_encoded', 'adult_male']
    
    X = titanic_multilabel[feature_columns].fillna(0)
    y = titanic_multilabel['risk_class']
    
    print(f"\n최종 데이터 형태:")
    print(f"- 특징 수: {X.shape[1]}")
    print(f"- 샘플 수: {X.shape[0]}")
    print(f"- 타겟 클래스: {np.unique(y)}")
    
    # 6. 데이터 분할 (논문처럼 80:20 비율)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n데이터 분할:")
    print(f"- 훈련 데이터: {X_train.shape[0]}개 샘플")
    print(f"- 테스트 데이터: {X_test.shape[0]}개 샘플")
    
    # 7. 모델 학습 및 평가
    print("\n" + "="*50)
    results = evaluate_models(X_train, X_test, y_train, y_test)
    direct_classifier, stacked_classifier, y_pred_direct, y_pred_stacked, direct_accuracy, stacked_accuracy = results
    
    # 8. 특성 중요도 분석
    print("\n" + "="*50)
    importance_df = analyze_feature_importance(direct_classifier, X, feature_columns)

    # === ROC/AUC & SHAP analysis ===    
    FINAL_MODEL_VAR = stacked_classifier
    # 1) locate fitted model variable
    frame = {**globals(), **locals()}
    model = None
    # 1-1) explicit preferred name
    if isinstance(FINAL_MODEL_VAR, str) and FINAL_MODEL_VAR:
        m = frame.get(FINAL_MODEL_VAR)
        if m is not None:
            model = m
    # 1-2) fallback: common names
    if model is None:
        for _name in ["model", "clf", "final_model", "best_model",
                      "stack_clf", "stacking_clf", "stacking_model",
                      "estimator", "pipe", "pipeline"]:
            m = frame.get(_name)
            if m is not None:
                model = m
                break
    # 1-3) last resort: scan any estimator-like fitted object
    if model is None:
        # pick first object that looks fitted: has predict and n_features_in_/classes_
        for name, obj in frame.items():
            if hasattr(obj, "predict") and hasattr(obj, "fit"):
                if any(hasattr(obj, attr) for attr in ("n_features_in_", "classes_", "feature_names_in_")):
                    model = obj
                    break
    if model is None:
        cand = [n for n, o in frame.items() if hasattr(o, "predict") and hasattr(o, "fit")]
        raise RuntimeError(f"Fitted model not found. Set FINAL_MODEL_VAR to one of: {cand}")

    # 2) choose evaluation split without NameError by probing availability
    frame = {**globals(), **locals()}
    candidates = [
        ("X_test", "y_test"),
        ("X_valid", "y_valid"),
        ("X_val", "y_val"),
        ("X_eval", "y_eval"),
        ("X_train", "y_train"),
        ("X", "y"),
    ]
    for xn, yn in candidates:
        if xn in frame and yn in frame:
            X_eval, y_eval = frame[xn], frame[yn]
            break
    else:
        raise RuntimeError(
            "No evaluation split found. Define one of: "
            "(X_test,y_test) | (X_valid,y_valid) | (X_val,y_val) | (X_train,y_train) | (X,y)."
        )

    # 3) ROC/AUC
    class_names = getattr(model, "classes_", None)
    auc_map = plot_roc_auc(model, X_eval, y_eval, class_names=class_names, title="ROC Curve (eval)")
    print("[AUC]", {k: round(v, 6) for k, v in auc_map.items()})

    # 4) SHAP (optional if shap installed)
    feature_names = list(X_eval.columns) if hasattr(X_eval, "columns") else None
    try:
        _ = analyze_feature_importance_shap(
            model,
            X_eval,
            feature_names=feature_names,
            title_prefix="SHAP (eval)"
        )
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    # 9. 결과 저장
    save_results(y_test, y_pred_direct, y_pred_stacked, importance_df, direct_accuracy, stacked_accuracy)
    
    # 10. 최종 요약
    print(f"\n" + "="*80)
    print(f"📋 논문 구현 및 타이타닉 적용 최종 요약")
    print(f"="*80)
    
    print(f"\n🎯 논문 주요 내용:")
    print(f"- 제목: 스태킹 앙상블 및 설명가능한 AI를 활용한 대금지급지연 예측")
    print(f"- 방법론: 2층 스태킹 앙상블 (Base Classifiers + Meta Classifier)")
    print(f"- 핵심 기법: One-vs-Rest 이진 분류 + Random Forest 메타 분류기")
    print(f"- 설명가능성: SHAP을 통한 특성 중요도 분석")
    
    print(f"\n📊 논문 vs 타이타닉 적용 결과:")
    print(f"논문 (대금지급지연 데이터):")
    print(f"  • 데이터: 107,012개 송장, 37개 특성")
    print(f"  • 직접 분류: 81% 정확도")
    print(f"  • 스태킹 앙상블: 93% 정확도 (+12%p)")
    
    print(f"\n타이타닉 적용:")  
    print(f"  • 데이터: 891개 승객, 17개 특성")
    print(f"  • 직접 분류: {direct_accuracy:.1%} 정확도")
    print(f"  • 스태킹 앙상블: {stacked_accuracy:.1%} 정확도 ({(stacked_accuracy-direct_accuracy)*100:+.1f}%p)")
    
    print(f"\n🔍 주요 학습 포인트:")
    print(f"1. 스태킹 앙상블은 대용량 복잡 데이터에서 진가를 발휘")
    print(f"2. 작은 데이터셋에서는 단순한 모델이 때로는 더 효과적")
    print(f"3. 도메인 특성을 반영한 특성 엔지니어링이 핵심")
    print(f"4. 논문의 방법론을 다른 도메인에 적용할 때는 적절한 조정 필요")
    
    print(f"\n✅ 구현 완료!")

# ============================================================================
# 10. 실행
# ============================================================================

if __name__ == "__main__":
    main()