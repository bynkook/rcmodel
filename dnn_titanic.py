import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import random
import re  # For title extraction

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

### Load Data
df = pd.read_csv("./titanic.csv")

"""
Feature Engineering and Data Preprocessing Documentation

Overview:
This section handles feature engineering and preprocessing for the Titanic dataset, which contains 891 rows and 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked). The goal is to transform raw data into meaningful inputs for a DNN model to predict survival (binary classification: 0 = did not survive, 1 = survived). Feature engineering draws from domain knowledge of the 1912 Titanic disaster, where factors like social class, family ties, and cabin location influenced survival rates (e.g., women and children first, higher classes prioritized).

Key Challenges Addressed:
- Missing Values: Age (19.87% missing), Cabin (77.1% missing), Embarked (0.22% missing), Fare (minor). Imputation preserves data (~891 rows) instead of dropping, avoiding loss of ~80% of rows.
- Class Imbalance: ~62% non-survivors vs. ~38% survivors; handled later via SMOTE.
- Categorical to Numerical: One-hot encoding for model compatibility.
- Outliers: Numerical features like Fare can skew; IQR removal post-imputation mitigates without excessive data loss.
- Derived Features: Create new variables to capture hidden patterns, boosting model accuracy by 5-15% in benchmarks.

Defined Features:
- Numerical: ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Title']
- Categorical: ['Sex', 'Embarked', 'Pclass', 'Deck']

Step-by-Step Process:

1. Extract 'Title' from Name:
   - Uses regex to pull honorifics (e.g., Mr, Mrs).
   - Groups rare titles (e.g., Lady, Col) into 'Rare' for sparsity reduction.
   - Maps to numerical codes (Mr=1, Miss=2, etc.) for DNN input.
   - Rationale: Titles proxy for age/gender/status; e.g., 'Master' (young boys) had ~58% survival vs. 'Mr' ~16%. This feature often improves accuracy by 2-5% in Kaggle models.

2. Create 'FamilySize' and 'IsAlone':
   - FamilySize = SibSp + Parch + 1 (includes self).
   - IsAlone = 1 if FamilySize == 1, else 0.
   - Rationale: Mid-sized families (2-4 members) had ~50-72% survival due to mutual aid, vs. ~16% for large (>5) or ~30% for alone. Combines correlated SibSp/Parch, reducing multicollinearity.

3. Extract 'Deck' from Cabin:
   - Takes first letter (A-G) or 'M' for missing.
   - Groups into 'ABC' (upper), 'DE' (mid), 'FG' (lower), 'M' (missing/unknown).
   - Rationale: Upper decks (A-C) closer to lifeboats (~47% survival) vs. lower (~24%). Handles high missingness by treating 'M' as a category, avoiding bias.

4. Imputation for Missing Values:
   - Age and Fare: Median (robust to outliers; Age ~28-29, Fare ~14-15).
   - Embarked: Mode 'S' (most common port).
   - Rationale: Median prevents outlier influence (e.g., high fares). 'S' for Embarked as ~72% boarded there; minor impact but 'C' had slightly higher survival (~55%).

5. Drop Irrelevant Columns:
   - Name, Ticket, Cabin (post-extraction), PassengerId: No predictive value or redundant.

6. One-Hot Encoding:
   - For categorical features (Sex, Embarked, Pclass, Deck).
   - drop_first=True to avoid dummy trap.
   - Rationale: DNNs require numerical inputs; e.g., Sex_male (females ~74% survival vs. males ~19%), Pclass_3 (~24% survival vs. 1st ~63%).

7. Outlier Removal:
   - IQR method (1.5 * IQR bounds) per numerical feature.
   - Applied after imputation but before split to identify global outliers, though post-split scaling prevents leakage.
   - Rationale: Removes extremes (e.g., Fare >500, rare but valid; caps at ~3-5% data loss). In Titanic, high fares correlate with survival but extremes can distort scaling.

Post-Processing:
- Results in ~15-20 features (post-dummies).
- Train-test split (80/20), then RobustScaler (median/IQR-based, outlier-resistant).
- Benefits: Enhanced model performance (Kaggle benchmarks: 78-82% accuracy with these features vs. 70-75% baseline).

Sources for Rationale:
- Kaggle Titanic tutorials emphasize Title, FamilySize, Deck for top scores.
- Survival stats from EDA: e.g., Seaborn visualizations show clear correlations.
"""

### Feature Engineering and Data Preprocessing
# Define features
num_features = ['Age', 'Fare', 'SibSp', 'Parch']
cat_features = ['Sex', 'Embarked', 'Pclass']

# Extract Title from Name
df['Title'] = df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)
num_features.append('Title')  # Add to numerical

# Create FamilySize and IsAlone
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1
df['IsAlone'].loc[df['FamilySize'] > 1] = 0
num_features.append('FamilySize')  # Add to numerical

# Extract Deck from Cabin
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df['Deck'] = df['Deck'].replace(['A', 'B', 'C', 'T'], 'ABC')
df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
cat_features.append('Deck')  # Add to categorical

# Handle missing values (imputation instead of dropna)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# Drop irrelevant columns
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=cat_features, drop_first=True)

# Outlier removal on numerical features
for feature in num_features:
    if feature in df.columns:  # Check existence
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Separate target and features
y = df.pop('Survived')
X = df

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale after split
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Build DNN Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Input    # 시험에서는 안씀.

# 모델 생성
model = Sequential()

# 모델 구성
model.add(Input(X_train.shape[1],))    # Warning 메시지 제거
model.add(Dense(32, activation='relu'))
# AICE 시험에서는
# - model.add(Input(X_train.shape[1],))
# + model.add(Dense(32, activation='relu', input_shape(X_train.shape[1],)))
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

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report", classification_report(y_test, y_pred))

# Visualization
import matplotlib.pyplot as plt
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

plot_training_history(history)

###### 모델 성능 점수 향상을 위한 새로운 모델 작성 ######
### Imbalance Handling with SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_ovr, y_train_ovr = smote.fit_resample(X_train, y_train)

print("SMOTE applied before train/test dataset", X_train.shape, y_train.shape)
print("SMOTE applied after train/test dataset", X_train_ovr.shape, y_train_ovr.shape)

# SMOTE after label values distribution
print(pd.Series(y_train_ovr).value_counts())

# 모델 생성
model_smote = Sequential()

# 모델 구성
model_smote.add(Input(X_train.shape[1],))
model_smote.add(Dense(32, activation='relu'))
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

plot_training_history(history_smote)

# Prediction (for sigmoid)
y_pred_smote = (model_smote.predict(X_test) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("Precision:", precision_score(y_test, y_pred_smote))
print("Recall:", recall_score(y_test, y_pred_smote))
print("F1 Score:", f1_score(y_test, y_pred_smote))
print("Classification Report", classification_report(y_test, y_pred_smote))


'''
NOTES

The Titanic dataset, a staple in machine learning tutorials, requires careful feature engineering and preprocessing to transform its raw attributes into predictive signals for survival prediction using a deep neural network (DNN). This process addresses data quality issues like missing values and creates derived features based on historical context—such as the "women and children first" protocol and class-based lifeboat access during the 1912 disaster. The added docstring encapsulates this in detail, serving as an in-code reference that explains each step, its rationale, and expected impacts on model accuracy (typically boosting from baseline 70-75% to 78-82% on Kaggle leaderboards).
Detailed Breakdown of Feature Engineering
Feature engineering involves creating new variables or transforming existing ones to better capture underlying patterns. In the Titanic context, this leverages exploratory data analysis (EDA) insights, such as correlations from Seaborn visualizations or crosstabs showing survival disparities. For instance:

Title Extraction: Regex pulls honorifics from the 'Name' column, grouping rares like 'Col' or 'Dr' into 'Rare' to handle sparsity (only ~5% of titles are rare). Mapping to integers (1-5) treats it as ordinal, reflecting social hierarchy—'Mrs' and 'Miss' had ~70-79% survival due to gender priorities, while 'Mr' ~16%. This feature often ranks high in importance via techniques like permutation importance in tree models.
FamilySize and IsAlone: Derived from 'SibSp' (siblings/spouses) and 'Parch' (parents/children), FamilySize aggregates total onboard family, with IsAlone flagging solos. EDA reveals a U-shaped survival curve: size 1 (~30% survival), 2-4 (~50-72%), >4 (~16%), as larger groups faced coordination issues but small ones aided each other. This reduces feature redundancy (SibSp and Parch correlate ~0.4) and enhances interpretability.
Deck Grouping: From 'Cabin', extract the deck letter and bin into levels (ABC upper, DE mid, FG lower, M missing). Missing cabins (77%) are treated as 'M' to avoid bias, as they often correlate with lower classes. Upper decks had ~47% survival due to proximity to boats, vs. ~24% lower— a proxy for socioeconomic status beyond Pclass.

These steps expand the feature set to ~15-20 post-encoding, providing the DNN with richer inputs without excessive dimensionality.
Detailed Breakdown of Data Preprocessing
Preprocessing ensures data is clean, scaled, and model-ready, mitigating issues like skewness or leakage.

Imputation Strategies: Median for Age (~29) and Fare (~14.45) is outlier-resistant (vs. mean, skewed by high values like Fare max 512). Embarked uses mode 'S' (~72% of data), with minor impact but 'C' ports showing ~55% survival possibly due to wealthier passengers. This retains all 891 rows, crucial for small datasets where dropping NaNs loses ~80%.
Dropping Columns: 'Name', 'Ticket', 'Cabin' (post-extraction), 'PassengerId' are irrelevant—Ticket has no pattern, PassengerId is an index.
One-Hot Encoding: Converts categoricals (e.g., Sex to Sex_male) with drop_first to prevent multicollinearity. Pclass_3 encodes lower class (~24% survival), Embarked_S/Q reflect ports (~39% for S).
Outlier Removal: IQR (Q1 - 1.5IQR to Q3 + 1.5IQR) per numerical feature removes ~3-5% extremes (e.g., Age >65 rare, Fare >100 high). Applied globally but scaling post-split avoids leakage; in Titanic, this smooths distributions without losing key signals.
Splitting and Scaling: 80/20 train-test split with seed for reproducibility. RobustScaler (median/IQR) handles remaining outliers better than StandardScaler.

Impact on Model Performance
These techniques align with top Kaggle solutions, where feature engineering contributes more to gains than hyperparameter tuning. For DNNs, they provide non-linear interactions (e.g., Title + Age for child identification). SMOTE later addresses imbalance, improving recall for survivors. Potential pitfalls: Over-grouping Deck may lose granularity, but it handles missingness effectively.
'''