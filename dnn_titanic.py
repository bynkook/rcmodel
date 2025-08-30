import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import random

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

### EDA
df = pd.read_csv("./titanic.csv")
df = df.dropna()

##### Feature Engineering and Data Preprocessing #####
num_features = [...]
cat_features = [...]

# 범주형 변수는 원-핫 인코딩(one-hot encoded)
df = pd.get_dummies(df, columns=cat_features, drop_first=True)

# target 0, 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Survived'] = le.fit_transform('Survived')

# 이상치 제거 removing outliers
for feature in num_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature]>=lower_bound)&(df[feature]<=upper_bound)]

y = df.pop('Survived')
X = df

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 데이터셋 분리 후 스케일링
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    # 출력층 시그모이드 함수 사용(이진분류문제)
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

'''
# 다른 방법
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout #type: ignore
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train[1],))),
model.add(Dropout(rate=0.2)),
'''

# 모델 요약
model.summary()

# 모델 컴파일
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


model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
# 모델 학습
# 만약 history 로 저장하려면,

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #type: ignore
# point 는 소문자임에 주의

es = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

history = model.fit(
    X_train,                 # 학습 데이터 (입력)
    y_train,                 # 학습 데이터 (타겟)
    # validation_split=0.2,  # 학습 데이터의 20%를 검증 데이터로 사용(default:None)
    validation_data=(X_test, y_test),
    epochs=20,               # 최대 학습 반복 횟수
    batch_size=16,           # 배치 크기
    callbacks=[es, mc]       # 콜백 추가
)      

y_pred = (model.predict(X_test) > 0.5).astype(int)
# 시그모이드 출력층과 binary_crossentropy 손실 함수를 사용하는 이진 분류 DNN 모델
# model.predict(X_test) > 0.5는 각 샘플의 예측 확률이 0.5보다 크면 True
# .astype(int)는 boolean 값을 0 or 1 로 변환
# softmax 함수로 2개 output 출력되었을 경우, 
# y_pred = np.argmax(model.predict(X_test, axis=1)) 로 2중에 큰값을 저장한다.
# axis = 1 은 컬럼방향 최대값 선택의 의미
# np.argmax : 행(axis = 0) 또는 열(axis = 1)을 따라 가장 큰 값의 index 반환


# 점수 출력
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report", classification_report(y_test, y_pred)) # 모아서 출력

# 모델 성능 시각화
import matplotlib.pyplot as plt
def plot_training_history(history):
    # 손실 그래프
    plt.figure(figsize=(12, 5))
    
    # 손실 값 (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 정확도 값 (Accuracy)
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
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
    
    # 그래프 출력
    plt.tight_layout()
    plt.show()

# 함수 호출: 모델 학습 결과 시각화
plot_training_history(history)


###### 모델 성능 점수 향상을 위한 새로운 모델 작성 ######
# pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_ovr, y_train_ovr = smote.fit(X_train, y_train)

print("SMOTE 적용 전 train/test 데이터셋",  X_train.shape, y_train.shape)
print("SMOTE 적용 후 train/test 데이터셋",  X_train_ovr.shape, y_train_ovr.shape)

# SMOTE 적용 후 레이블값 분포 확인
pd.Series(y_train_ovr).value_counts()

# build DNN model
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout #type: ignore
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train[1],))),
model.add(Dense(32, activation='relu')),
model.add(Dropout(rate=0.2)),
model.add(Dense(16, activation='relu')),
model.add(Dropout(rate=0.2)),
model.add(Dense(8, activation='relu')),
model.add(Dropout(rate=0.2)),
model.add(Dense(2, activation='softmax'))

# model compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
mc = ModelCheckpoint('best_model_smote.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

history = model.fit(
    X_train_ovr,              # 학습 데이터 (입력)
    y_train_ovr,               # 학습 데이터 (타겟)
    # validation_split=0.2,   # 학습 데이터의 20%를 검증 데이터로 사용(default:None)
    validation_data=(X_test, y_test),
    epochs=20,                # 최대 학습 반복 횟수
    batch_size=16,            # 배치 크기
    callbacks=[es, mc],       # 콜백 추가
)

plot_training_history(history)

y_pred = np.argmax(model.predict(X_test, axis=1))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report", classification_report(y_test, y_pred)) # 모아서 출력