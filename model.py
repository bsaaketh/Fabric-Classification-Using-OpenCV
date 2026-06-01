import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern
import joblib

# LBP parameters — must match app.py exactly
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS  # 24
LBP_METHOD = 'uniform'
LBP_BINS = LBP_POINTS + 2     # 26

IMG_SIZE = 128  # faster training, still accurate


def extract_features(image):
    """Extract LBP texture features from a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def load_dataset(dataset_path, classes=None):
    """Load images from class-named subdirectories, optionally filtering by class list."""
    data, labels = [], []
    folders = sorted(os.listdir(dataset_path))
    for folder in folders:
        if classes and folder not in classes:
            continue
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        loaded = 0
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            image = cv2.imread(file_path)
            if image is None:
                continue
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(extract_features(image))
            labels.append(folder)
            loaded += 1
        print(f"  {folder}: {loaded} images")
    return np.array(data), np.array(labels)


# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_path = os.path.join(os.path.dirname(__file__), "datasets")
TARGET_CLASSES = ["Cotton", "Denim", "Leather", "Nylon", "Wool"]

print("Loading dataset...")
X, y = load_dataset(dataset_path, classes=TARGET_CLASSES)
X, y = shuffle(X, y, random_state=42)
print(f"Total samples: {len(X)}")

# ── Label encoding ────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── SVM ───────────────────────────────────────────────────────────────────────
print("\nTraining SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_acc:.4f}")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# ── Random Forest ─────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ── Save best model ───────────────────────────────────────────────────────────
best_model = svm if svm_acc >= rf_acc else rf
best_name = "SVM" if svm_acc >= rf_acc else "Random Forest"
print(f"\nBest model: {best_name} ({max(svm_acc, rf_acc):.4f})")

joblib.dump(best_model, "fabric_classifier_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(rf, "fabric_rf_model.pkl")
joblib.dump(svm, "fabric_svm_model.pkl")
print("Models and label encoder saved.")
