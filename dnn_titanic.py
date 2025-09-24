"""
dnn_titanic.py
==============

개요
----
이 스크립트는 Titanic 데이터셋('titanic.csv')을 이용해
간단한 **DNN(다층퍼셉트론, MLP)** 분류 모델을 학습·평가하는 예제입니다.
학습 목적의 코드로서 **노트북 스타일(line-by-line)** 로 전처리 → 분할 → 스케일링 → 모델 학습 → 평가 순으로
직관적으로 작성되어 있으며, 별도의 전처리 함수는 사용하지 않습니다.

데이터셋
--------
입력 파일: 'titanic.csv' (seaborn 제공 스키마와 동일한 컬럼 구성 가정)
주요 컬럼 예:
- 타깃: 'survived' (0/1)
- 숫자형: 'pclass', 'age', 'sibsp', 'parch', 'fare'
- 범주형/불리언: 'sex', 'embarked', 'deck', 'who', 'adult_male', 'alone', 'class', 'alive'

데이터 전처리
-------------------------------
- **컬럼 정리**
  - 타깃 누수 방지: 'alive', 'who' 는 'survived', 'sex' 와 동치이므로 제거  
  - 중복 의미 축소: 'class'(텍스트) 대신 'pclass'(정수) 사용
- **파생 변수**
  - 'family_size = sibsp + parch + 1'
  **결측치 처리**
  - 숫자형('age', 'fare')은 **중앙값**으로 대체
  - 범주형('embarked', 'deck')은 **최빈값**으로 대체
- **인코딩**
  - 범주형: 'embarked', 'deck' 을 **원-핫 인코딩**
  - 불리언: 'adult_male', 'alone' 등은 사용 시 'int'로 변환
- **스케일링**
  - 숫자 피처는 **RobustScaler**를 사용하여 이상치 영향 완화

데이터 분할 및 검증/평가 (현재 코드의 기본 흐름)
---------------------------------------------
- **Train/Test 분할**: 'train_test_split(..., test_size=0.2, stratify=y, random_state=42)'
- **검증 방법**:
  - 현재 코드는 예시로 'model.fit(..., validation_data=(X_test, y_test))' 형태를 포함할 수 있습니다.
  - 이는 **테스트 세트를 학습 중 모니터링에 사용**한다는 의미로, 엄밀한 일반화 성능 보고에는 적절하지 않습니다.
  - 실전에서는 'validation_split=...' 또는 별도의 '(X_val, y_val)' 분리를 권장하며,
    **최종 Test 평가는 학습·튜닝에 전혀 쓰지 않은 세트**로 수행해야 합니다.

모델 구성 (DNN/MLP)
-------------------
- 프레임워크: **TensorFlow 2.x (tf.keras 공개 API)** 사용
  - 내부 모듈 경로('tensorflow.python.keras...')가 아닌 **공개 경로('tensorflow.keras')** 를 사용해야
    버전 호환성 문제(예: 'DistributedDatasetInterface' AttributeError)를 피할 수 있습니다.
- 구조: 'Sequential' + 'Dense'(필요 시 'Dropout')
  - 입력 차원: 전처리 후 피처 수
  - 은닉층/유닛/활성화/드롭아웃 비율은 실험용으로 단순 구성
  - 이진 분류의 경우 출력층은 'Dense(1, activation='sigmoid')' 를 사용(모델의 'predict'가 양성 확률로 해석 가능)

학습 설정
---------
- 콜백: 'EarlyStopping', 'ModelCheckpoint' (필요 시 사용)
- 손실/평가지표: 이진 분류 환경에 맞는 손실('binary_crossentropy') 및 지표(예: 'accuracy')

평가 및 ROC-AUC
---------------
- 예측 확률: Keras 분류 모델은 **'predict_proba'가 아닌 'model.predict'** 를 사용합니다.
  - 이진 분류('sigmoid')에서는 'model.predict(X).ravel()' 이 **양성(1) 확률**로 해석됩니다.
- ROC-AUC: 'sklearn.metrics.roc_auc_score(y_true, y_prob)' 로 계산
  - ROC 곡선은 'roc_curve'로 좌표를 얻어 matplotlib으로 시각화할 수 있습니다.

재현성과 주의사항
-----------------
- 시드 고정: 'numpy', 'tensorflow'의 시드를 고정해 실험 재현성을 높였습니다.
- 본 코드는 **학습용/연습용**으로 정확도 최적화보다는 **실행 가능성·가독성**에 중점을 두었습니다.
  실제 서비스/연구용에서는 별도의 검증 세트 확보, 교차검증, 하이퍼파라미터 탐색, 불균형 처리, 특성 선택/정규화 등의 개선이 필요합니다.

필요 패키지(예시)
-----------------
- 'pandas', 'numpy', 'scikit-learn', 'tensorflow>=2.x', 'matplotlib'

요약
----
이 스크립트는 'titanic.csv'의 실제 컬럼과 형식(문자/숫자)에 맞춘 전처리와
간단한 DNN 모델 학습 코드를 **줄 단위**로 제공하여 초심자도 흐름을 쉽게 따라갈 수 있도록 구성했습니다.
현재 구현은 **테스트 세트를 검증에 사용하는 예시 구문**을 포함할 수 있으므로,
엄밀한 평가를 원할 경우 **검증 세트 분리** 또는 'validation_split'을 활용하고
**테스트 세트는 오직 최종 평가**에만 사용하시기 바랍니다.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.over_sampling import SMOTE

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# === 1. 데이터 로딩 ===
df = pd.read_csv('./titanic.csv')
df.columns = df.columns.str.lower()     # 컬럼명 소문자로 변경

# === 2. 불필요한 컬럼 제거 ===
df.drop(columns=["class", "who"], inplace=True)

# === 3. 결측치 처리 ===
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in df.columns if c not in num_features and c != "survived"]

# target 변수의 결측치는 모두 제거하는게 맞다
df.dropna(subset='survived', inplace=True, ignore_index=True)

for c in num_features:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_features:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# === 4. 이상치 제거 (IQR 기반) ===
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# === 5. feature engineering ===
df['family_size'] = df['sibsp'] + df['parch'] + 1

# === 6. 인코딩 ===
# 레이블 값 대치
df['adult_male'] = df['adult_male'].map({True:1, False:0}).astype(int)
df['alone'] = df['alone'].map({True:1, False:0}).astype(int)
df['sex'] = df['sex'].map({'male':1, 'female':0}).astype(int)

# 원-핫 인코딩 (True/False 가 아니고 1/0 으로)
df = pd.get_dummies(data=df, drop_first=True).astype(int)   # 범주형만 자동으로 선택되어 변환됨.

# target 값 정리
le = LabelEncoder()
df['survived']= le.fit_transform(df['survived'])

# ==== 7. 입력/타깃 분리 ====
y = df.pop("survived")
X = df
feature_names = X.columns.tolist() # for SHAP analysis

# ==== 8. 학습/평가 분할 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==== 9. 스케일링 ====
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


### Build DNN Model
model = Sequential()
model.add(Input((X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류 문제

# Model summary
model.summary()

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=9, verbose=1)
mc = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # 학습 데이터의 20%를 검증 데이터로 사용(default:None)
    # validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# Prediction
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report", classification_report(y_test, y_pred))

# Visualization
def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_pred_proba_pos):
   fpr, tpr, _ = roc_curve(y_test, y_pred_proba_pos)
   auc_score = roc_auc_score(y_test, y_pred_proba_pos)  # ← 추가
   plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
   plt.plot([0, 1], [0, 1], 'k--', label='Classifier')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc='lower right')
   plt.show()

import shap
def perform_shap_analysis(model, X_test, feature_names):
    """
    Performs SHAP analysis to explain model predictions.

    Args:
        model (tensorflow.keras.Model): The trained Keras model.
        X_test (np.array): The test data.
        feature_names (list): A list of feature names corresponding to X_test columns.
    
    Example usage:
        Before calling the function, make sure you have the feature names
        feature_names = X.columns.tolist()
        perform_shap_analysis(model, X_test, feature_names)
    """
    # Create a SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X_test)
    
    # Calculate SHAP values for the test data
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot of SHAP values
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Waterfall plot for a single instance (e.g., the first instance)
    # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html
    print("Generating SHAP waterfall plot for the first instance...")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0][0], 
        base_values=explainer.expected_value[0], 
        data=X_test[0], 
        feature_names=feature_names
    ))


plot_training_history(history)

# plot ROC curve
y_prob = model.predict(X_test).ravel()
plot_roc_curve(y_test, y_prob)

# SHAP analysis
# perform_shap_analysis(model, X_test, feature_names)

"""
scikit-learn 추정기(분류기)에는 관례적으로 predict_proba가 있다.
tf.keras 모델은 model.predict가 “마지막 층의 출력값”을 그대로 반환한다.

마지막 층이 sigmoid(이진 분류)면 predict 결과가 곧 양성 확률.
마지막 층이 softmax(다중 분류)면 predict 결과가 각 클래스 확률 분포(합=1.0).

numpy.ravel(a, order='C') : Return a contiguous flattened array.
numpy.array.ravel()은 모양(차원)을 (N,1) → (N,)으로 축소함.
이진 분류에서 predict가 (샘플수, 1) 형태로 나오므로,
y_prob = model.predict(X_test).ravel()로 1차원 벡터로 바꿔
roc_auc_score, roc_curve 같은 sklearn 지표에 맞춥니다.
"""

###### 모델 성능 점수 향상을 위한 새로운 모델 작성 ######
# Imbalance Handling with SMOTE
smote = SMOTE(random_state=0)
X_train_ovr, y_train_ovr = smote.fit_resample(X_train, y_train)

print("SMOTE applied before train/test dataset", X_train.shape, y_train.shape)
print("SMOTE applied after train/test dataset", X_train_ovr.shape, y_train_ovr.shape)

# SMOTE after label values distribution
print(pd.Series(y_train_ovr).value_counts())

# 모델 생성
model_smote = Sequential()
model_smote.add(Input((X_train_ovr.shape[1],)))
model_smote.add(Dense(32, activation='relu'))
model_smote.add(Dense(16, activation='relu'))
model_smote.add(Dropout(rate=0.2))
model_smote.add(Dense(8, activation='relu'))
model_smote.add(Dropout(rate=0.2))
model_smote.add(Dense(4, activation='relu'))
model_smote.add(Dropout(rate=0.1))
model_smote.add(Dense(1, activation='sigmoid'))  # 이진 분류 문제

# Compile
model_smote.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=9, verbose=1)
mc = ModelCheckpoint('best_model_smote.keras', monitor='val_loss', save_best_only=True, verbose=1)

history_smote = model_smote.fit(
    X_train_ovr, y_train_ovr,
    validation_split=0.2,
    # validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# Prediction (for sigmoid)
y_pred_smote = (model_smote.predict(X_test) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("Precision:", precision_score(y_test, y_pred_smote))
print("Recall:", recall_score(y_test, y_pred_smote))
print("F1 Score:", f1_score(y_test, y_pred_smote))
print("Classification Report", classification_report(y_test, y_pred_smote))

plot_training_history(history_smote)

# plot ROC curve
y_prob_sm = model_smote.predict(X_test).ravel()
plot_roc_curve(y_test, y_prob_sm)