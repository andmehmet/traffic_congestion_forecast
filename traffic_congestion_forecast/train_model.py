import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. Veri setini oku
df = pd.read_csv("trafik.csv")

# 2. Girdi ve çıktıyı ayır
X = df[["saat", "gun", "hava"]]
y = df["etiket"]

# 3. Etiketi sayıya çevir
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeli eğit
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# 7. Performans sonuçlarını göster
print("=== MODEL PERFORMANS SONUÇLARI ===")
print(f"Accuracy (Doğruluk): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetaylı Rapor:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Modeli kaydet
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel başarıyla kaydedildi!")
