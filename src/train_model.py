import os
import json
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import argparse


def train_svm_model(
    features_path="data/processed/features.npy",
    labels_path="data/processed/labels.npy",
    model_path="models/svm_model.pkl",
    scaler_path="models/scaler.pkl",
    label_map_path="models/label_map.json",
    test_size=0.3,
    random_state=42
):
    print("📥 Loading features and labels...")
    X = np.load(features_path)
    y = np.load(labels_path)
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each.")

    # Save label map (sorted to ensure consistent ordering)
    unique_labels = sorted(set(y))
    label_map = OrderedDict((label, i) for i, label in enumerate(unique_labels))

    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"✅ Label map saved to: {label_map_path}")

    print("\n⚖️  Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Feature scaling complete.")

    print("\n🔀 Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"✅ Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    print("\n🚂 Training SVM classifier...")
    model = SVC(kernel='rbf', class_weight='balanced', probability=True)
    model.fit(X_train, y_train)
    print("✅ Training complete.")

    print("\n🧪 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels=unique_labels)
    report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=0)

    print(f"\n🎯 Accuracy: {accuracy:.2%}")
    print("\n📊 Confusion Matrix:")
    print(conf_mat)
    print("\n📋 Classification Report:")
    print(report)

    print("\n💾 Saving model and scaler...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM model on audio features.")
    parser.add_argument("--features", default="data/processed/features.npy",
                        help="Path to features.npy")
    parser.add_argument("--labels", default="data/processed/labels.npy", help="Path to labels.npy")
    parser.add_argument("--model", default="models/svm_model.pkl",
                        help="Path to save the SVM model")
    parser.add_argument("--scaler", default="models/scaler.pkl", help="Path to save the scaler")
    parser.add_argument("--label_map", default="models/label_map.json",
                        help="Path to save the label map")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test set proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_svm_model(
        features_path=args.features,
        labels_path=args.labels,
        model_path=args.model,
        scaler_path=args.scaler,
        label_map_path=args.label_map,
        test_size=args.test_size,
        random_state=args.random_state
    )
