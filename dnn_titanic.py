"""
dnn_titanic.py
==============

개요
----
이 스크립트는 **seaborn 형식의 Titanic 데이터셋(`titanic.csv`)**을 이용해
간단한 **DNN(다층퍼셉트론, MLP)** 분류 모델을 학습·평가하는 예제입니다.
학습 목적의 코드로서 **노트북 스타일(line-by-line)** 로 전처리 → 분할 → 스케일링 → 모델 학습 → 평가 순으로
직관적으로 작성되어 있으며, 별도의 전처리 함수는 사용하지 않습니다.

데이터셋
--------
입력 파일: `titanic.csv` (seaborn 제공 스키마와 동일한 컬럼 구성 가정)
주요 컬럼 예:
- 타깃: `survived` (0/1)
- 숫자형: `pclass`, `age`, `sibsp`, `parch`, `fare`
- 범주형/불리언: `sex`, `embarked`, `deck`, `who`, `adult_male`, `embark_town`, `alone`, `class`, `alive` 등

데이터 전처리 (현재 코드에 맞춤)
-------------------------------
- **컬럼 정리**
  - 타깃 누수 방지: `alive`(문자형 생존 여부)는 `survived`와 동치이므로 제거
  - 중복 의미 축소: `class`(텍스트) 대신 `pclass`(정수) 사용
- **파생 변수**
  - `family_size = sibsp + parch + 1`
  - `is_alone` = `alone`(bool)을 `int`로 변환(없으면 `family_size==1`로 대체 산출)
  - `fare_per_person = fare / family_size` (무한대/결측은 0 또는 중앙값으로 치환)
- **결측치 처리**
  - 숫자형(`age`, `fare`, `fare_per_person`)은 **중앙값**으로 대체
  - 범주형(`embarked`, `who`, `embark_town`)은 **최빈값**으로 대체
  - `deck`은 결측을 **별도 범주 'M'** 로 두고, 갑판을 그룹화(예: A/B/C→'ABC', D/E→'DE', F/G→'FG', M 유지)
- **인코딩**
  - 범주형: `sex`, `embarked`, `deck`, `who`, `pclass` 등을 **원-핫 인코딩**(`pd.get_dummies`, `drop_first=True`)
  - 불리언: `adult_male`, `alone` 등은 사용 시 `int`로 변환
- **스케일링**
  - 숫자 피처는 **RobustScaler**를 사용하여 이상치 영향 완화

데이터 분할 및 검증/평가 (현재 코드의 기본 흐름)
---------------------------------------------
- **Train/Test 분할**: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- **검증 방법**:
  - 현재 코드는 예시로 `model.fit(..., validation_data=(X_test, y_test))` 형태를 포함할 수 있습니다.
  - 이는 **테스트 세트를 학습 중 모니터링에 사용**한다는 의미로, 엄밀한 일반화 성능 보고에는 적절하지 않습니다.
  - 실전에서는 `validation_split=...` 또는 별도의 `(X_val, y_val)` 분리를 권장하며,
    **최종 Test 평가는 학습·튜닝에 전혀 쓰지 않은 세트**로 수행해야 합니다.

모델 구성 (DNN/MLP)
-------------------
- 프레임워크: **TensorFlow 2.x (tf.keras 공개 API)** 사용
  - 내부 모듈 경로(`tensorflow.python.keras...`)가 아닌 **공개 경로(`tensorflow.keras`)** 를 사용해야
    버전 호환성 문제(예: `DistributedDatasetInterface` AttributeError)를 피할 수 있습니다.
- 구조: `Sequential` + `Dense`(필요 시 `Dropout`)
  - 입력 차원: 전처리 후 피처 수
  - 은닉층/유닛/활성화/드롭아웃 비율은 실험용으로 단순 구성
  - 이진 분류의 경우 출력층은 `Dense(1, activation='sigmoid')` 를 사용(모델의 `predict`가 양성 확률로 해석 가능)

학습 설정
---------
- 콜백: `EarlyStopping`, `ModelCheckpoint` (필요 시 사용)
- 손실/평가지표: 이진 분류 환경에 맞는 손실(`binary_crossentropy`) 및 지표(예: `accuracy`)

평가 및 ROC-AUC
---------------
- 예측 확률: Keras 분류 모델은 **`predict_proba`가 아닌 `model.predict`** 를 사용합니다.
  - 이진 분류(`sigmoid`)에서는 `model.predict(X).ravel()` 이 **양성(1) 확률**로 해석됩니다.
- ROC-AUC: `sklearn.metrics.roc_auc_score(y_true, y_prob)` 로 계산
  - ROC 곡선은 `roc_curve`로 좌표를 얻어 matplotlib으로 시각화할 수 있습니다.

재현성과 주의사항
-----------------
- 시드 고정: `numpy`, `tensorflow`의 시드를 고정해 실험 재현성을 높였습니다.
- 본 코드는 **학습용/연습용**으로 정확도 최적화보다는 **실행 가능성·가독성**에 중점을 두었습니다.
  실제 서비스/연구용에서는 별도의 검증 세트 확보, 교차검증, 하이퍼파라미터 탐색, 불균형 처리, 특성 선택/정규화 등의 개선이 필요합니다.

필요 패키지(예시)
-----------------
- `pandas`, `numpy`, `scikit-learn`, `tensorflow>=2.x`, `matplotlib`

요약
----
이 스크립트는 `titanic.csv`의 실제 컬럼과 형식(문자/숫자)에 맞춘 전처리와
간단한 DNN 모델 학습 코드를 **줄 단위**로 제공하여 초심자도 흐름을 쉽게 따라갈 수 있도록 구성했습니다.
현재 구현은 **테스트 세트를 검증에 사용하는 예시 구문**을 포함할 수 있으므로,
엄밀한 평가를 원할 경우 **검증 세트 분리** 또는 `validation_split`을 활용하고
**테스트 세트는 오직 최종 평가**에만 사용하시기 바랍니다.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras import Sequential, Input                          # type:ignore
from tensorflow.keras.layers import Dense, Dropout                      # type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   # type:ignore
from imblearn.over_sampling import SMOTE

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ==== 1) 데이터 로드: seaborn 형식 titanic.csv ====
df = pd.read_csv("./titanic.csv")
# df.columns -> ['survived','pclass','sex','age','sibsp','parch','fare','embarked',
#                'class','who','adult_male','deck','embark_town','alive','alone']

# ==== 2) 타깃/누수 컬럼/중복 의미 컬럼 정리 ====
# 타깃: survived
# 누수(leakage): 'alive'는 survived를 문자로 표현 → 제거
# 중복/대체가능: 'class'(텍스트)는 'pclass'(정수)와 중복 의미 → pclass만 사용
cols_drop = ["alive", "class"]
df = df.drop(columns=[c for c in cols_drop if c in df.columns])

# ==== 3) 기본 파생 컬럼: family_size, is_alone, fare_per_person ====
df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
# seaborn 데이터에는 'alone'이 bool로 존재 → 정수로 변환(모델 입력에 유리)
df["is_alone"] = df["alone"].astype(int) if "alone" in df.columns else (df["family_size"] == 1).astype(int)
# 0으로 나눗셈 방지
df["fare_per_person"] = (df["fare"] / df["family_size"]).replace([np.inf, -np.inf], np.nan)

# ==== 4) 결측치 처리 ====
# 수치: age, fare, fare_per_person → 중앙값
for c in ["age", "fare", "fare_per_person"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# 범주: embarked, deck, who, embark_town → 최빈값/특정 라벨
if "embarked" in df.columns:
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])
if "deck" in df.columns:
    df["deck"] = df["deck"].fillna("M")  # 결측을 별도 카테고리로
if "who" in df.columns:
    df["who"] = df["who"].fillna(df["who"].mode().iloc[0])
if "embark_town" in df.columns:
    df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode().iloc[0])

# ==== 5) 덱(갑판) 그룹핑(선택) ====
# A/B/C → ABC, D/E → DE, F/G → FG, M(결측)
if "deck" in df.columns:
    df["deck"] = df["deck"].replace({"A":"ABC","B":"ABC","C":"ABC","D":"DE","E":"DE","F":"FG","G":"FG"})

# ==== 6) 사용 컬럼 선택(수치/범주) ====
num_features = ["age", "fare", "sibsp", "parch", "family_size", "fare_per_person", "is_alone"]
cat_features = ["sex", "embarked", "deck", "who", "pclass"]  # pclass를 범주로 더미화

# bool → int (adult_male, alone는 이미 처리 or 제거됨)
if "adult_male" in df.columns:
    df["adult_male"] = df["adult_male"].astype(int)
    num_features.append("adult_male")

# 실제 존재하는 컬럼만 사용하도록 필터링
num_features = [c for c in num_features if c in df.columns]
cat_features = [c for c in cat_features if c in df.columns]
df_model = df[num_features + cat_features + ["survived"]].copy()

# Outlier removal on numerical features
for ft in num_features:
   if ft in df_model.columns:  # Check existence
      Q1 = df_model[ft].quantile(0.25)
      Q3 = df_model[ft].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df_model = df_model[(df_model[ft] >= lower_bound) & (df_model[ft] <= upper_bound)]

# ==== 7) 원-핫 인코딩 ====
df_model = pd.get_dummies(df_model, columns=cat_features, drop_first=True)

# ==== 8) 입력/타깃 분리 ====
y = df_model.pop("survived").astype(int).values
X = df_model.values
feature_names = df_model.columns.tolist()

# ==== 9) 학습/평가 분할 ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==== 10) 스케일링 ====
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


### Build DNN Model
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류 문제

# Model summary
model.summary()

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 다중 클래스 분류(multi-class classification)
#  - categorical_crossentropy : 원핫인코딩 했을 경우
#  - sparse_categorical_crossentropy : 원핫인코딩 안했을 경우
# 출력층이 2개이상인 경우 tf.keras.layers.Dense(n, activation='softmax')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# y_pred = np.argmax(model.predict(X_test), axis=1)는 다중 클래스 분류 결과값
#  - axis=1 로 가장 높은 확률을 가진 레이블의 인덱스를 반환
# 다중 클래스 분류(하나의 클래스 선택)와 달리, 다중 레이블 분류(여러 클래스 동시 선택 가능)에서는
# 소프트맥스 대신 클래스별로 독립적인 시그모이드를 사용하고, binary_crossentropy를 적용

# metrics 기본 옵션 : accruacy, binary accruacy
# tf.keras.metrics  : precision, recall, f1_score, AUC 등 여러개 지정 가능
# 다중 레이블 분류에서는 일반적인 accuracy가 성능을 과소평가 할 수 있음
# binary_accuracy: 레이블별 정확도를 평가
# precision, recall, f1_score: 클래스 불균형이나 특정 클래스(양성)에 초점을 맞출 때 유용
# auc: 클래스 분리 능력을 평가

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    # validation_split=0.2,  # 학습 데이터의 20%를 검증 데이터로 사용(default:None)
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[es, mc],
    verbose=0
)

# Prediction
y_pred = (model.predict(X_test) > 0.5).astype(int)
# 시그모이드 출력층과 binary_crossentropy 손실 함수를 사용하는 이진 분류 DNN 모델
# model.predict(X_test) > 0.5는 각 샘플의 예측 확률이 0.5보다 크면 True
# .astype(int)는 boolean 값을 0 or 1 로 변환
# softmax 함수로 2개 output 출력되었을 경우,
# y_pred = np.argmax(model.predict(X_test, axis=1)) 로 2중에 큰값을 저장한다.
# axis = 1 은 컬럼방향 최대값 선택의 의미
# np.argmax : 행(axis = 0) 또는 열(axis = 1)을 따라 가장 큰 값의 index 반환

# Metrics

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report", classification_report(y_test, y_pred))

# Visualization
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
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
    ##### 다른 결과값 출력시
    # plt.plot(history.history['precision'], label='Training Precision')
    # plt.plot(history.history['val_precision'], label='Validation Precision')
    # plt.plot(history.history['recall'])
    # plt.plot(history.history['val_recall'])
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_pred_proba_pos):
   fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pos)
   auc_score = roc_auc_score(y_test, y_pred_proba_pos)  # ← 추가
   plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
   plt.plot([0, 1], [0, 1], 'k--', label='Classifier')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc='lower right')
   plt.show()

plot_training_history(history)

# plot ROC curve
y_prob = model.predict(X_test).ravel()
plot_roc_curve(y_test, y_prob)
"""
scikit-learn 추정기(분류기)에는 관례적으로 predict_proba가 있다.
tf.keras 모델은 model.predict가 “마지막 층의 출력값”을 그대로 반환한다.

마지막 층이 sigmoid(이진 분류)면 predict 결과가 곧 양성 확률.
마지막 층이 softmax(다중 분류)면 predict 결과가 각 클래스 확률 분포(합=1.0).

ravel()은 모양(차원)을 (N,1) → (N,)으로 펴 주는 것뿐입니다.
이진 분류에서 predict가 (샘플수, 1) 형태로 나오므로,
y_prob = model.predict(X_test).ravel()로 1차원 벡터로 바꿔
roc_auc_score, roc_curve 같은 sklearn 지표에 맞춥니다.

"""

# print AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"AUC Score: {auc_score:.2f}")


###### 모델 성능 점수 향상을 위한 새로운 모델 작성 ######
### Imbalance Handling with SMOTE
smote = SMOTE(random_state=0)
X_train_ovr, y_train_ovr = smote.fit_resample(X_train, y_train)

print("SMOTE applied before train/test dataset", X_train.shape, y_train.shape)
print("SMOTE applied after train/test dataset", X_train_ovr.shape, y_train_ovr.shape)

# SMOTE after label values distribution
print(pd.Series(y_train_ovr).value_counts())

# 모델 생성
model_smote = Sequential()

# 모델 구성
model_smote.add(Dense(32, activation='relu', input_shape=(X_train_ovr.shape[1],)))
model_smote.add(Dense(16, activation='relu'))
model_smote.add(Dropout(rate=0.2))
model_smote.add(Dense(16, activation='relu'))
model_smote.add(Dropout(rate=0.2))
model_smote.add(Dense(8, activation='relu'))
model_smote.add(Dropout(rate=0.2))
model_smote.add(Dense(1, activation='sigmoid'))  # 이진 분류 문제

# Compile
model_smote.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = ModelCheckpoint('best_model_smote.h5', monitor='val_loss', save_best_only=True, verbose=1)

history_smote = model_smote.fit(
    X_train_ovr, y_train_ovr,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[es, mc],
    verbose=0
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

# print AUC score
auc_score = roc_auc_score(y_test, y_prob_sm)
print(f"AUC Score: {auc_score:.2f}")
