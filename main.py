import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# Feature Extraction
# =====================================

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return np.mean(combined.T, axis=0)

# =====================================
# Load Dataset
# =====================================

X = []
y = []

real_path = "real"
fake_path = "fake"

real_files = [os.path.join(real_path, f) for f in os.listdir(real_path)]
fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path)]

for file in real_files:
    X.append(extract_features(file))
    y.append(0)

for file in fake_files:
    X.append(extract_features(file))
    y.append(1)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Real samples:", sum(y == 0))
print("Fake samples:", sum(y == 1))

# =====================================
# Train/Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================
# Train Model
# =====================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =====================================
# Evaluate
# =====================================

predictions = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# =====================================
# Cross Validation
# =====================================

cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# =====================================
# Save Model
# =====================================

joblib.dump(model, "voice_model.pkl")
print("\nModel saved as voice_model.pkl")
