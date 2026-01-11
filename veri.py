# ===============================
# 1. GEREKLİ KÜTÜPHANELER
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve
)

# ===============================
# 2. VERİYİ OKUMA
# ===============================
df = pd.read_csv("C:/Users/ASUS/Desktop/veri/veriseti/logistic_regression.csv")
print(df.head())

# ===============================
# 3. EDA (VERİ KEŞFİ)
# ===============================
print(df.info())
print(df.describe())

print("\nEksik Değerler:\n")
print(df.isnull().sum())

# Loan Status dağılımı
sns.countplot(x="loan_status", data=df)
plt.title("Loan Status Dağılımı")
plt.show()   # ✅ KALICI

# ===============================
# 4. ÖN İŞLEME
# ===============================

cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(exclude="object").columns

# Eksik değer doldurma
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

# Label Encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ===============================
# 5. BAĞIMSIZ / BAĞIMLI
# ===============================
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ===============================
# 6. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# 7. TRAIN - TEST
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 8. MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# 9. TAHMİN
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# 10. METRİKLER
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ===============================
# 11. GÖRSELLER
# ===============================

# Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()   # ✅ KALICI

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()   # ✅ KALICI
