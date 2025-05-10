import os
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict
import argparse


def extract_features_from_file(file_path, frame_length=1024, hop_length=512, n_mfcc=13):
    try:
        signal, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return None

    features = []

    # ----- Time-domain features -----
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T
    td_feats = []
    for frame in frames:
        max_val = np.max(frame)
        min_val = np.min(frame)
        rms_val = np.sqrt(np.mean(frame ** 2))
        avg_val = np.mean(frame)
        zero_crossings = np.mean(librosa.zero_crossings(frame, pad=False))
        td_feats.append([max_val, min_val, rms_val, avg_val, zero_crossings])
    td_feats = np.array(td_feats)
    td_mean = np.mean(td_feats, axis=0)
    td_std = np.std(td_feats, axis=0)
    features.extend(td_mean)
    features.extend(td_std)

    # ----- MFCCs -----
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    features.extend(mfccs_mean)
    features.extend(mfccs_std)

    # ----- Spectral Features -----
    spec_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]

    for feat in [spec_centroid, spec_bw, spec_rolloff, zcr]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    return np.array(features)


def extract_all_features(data_dir="data/raw", save_dir="data/processed"):
    features_list = []
    labels_list = []
    skipped_files = 0
    label_counts = defaultdict(int)
    skip_counts = defaultdict(int)

    data_path = Path(data_dir)

    print("🔍 Extracting features from audio files...\n")

    for label_dir in sorted(data_path.iterdir()):
        if label_dir.is_dir():
            label = label_dir.name
            print(f"📁 Processing label: {label}")
            count = 0
            for wav_path in sorted(label_dir.glob("*.wav")):
                feat = extract_features_from_file(str(wav_path))
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(label)
                    count += 1
                    label_counts[label] += 1
                else:
                    skipped_files += 1
                    skip_counts[label] += 1
            print(f"   ➤ {count} files processed, {skip_counts[label]} skipped.\n")

    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "features.npy"), X)
    np.save(os.path.join(save_dir, "labels.npy"), y)

    # Summary
    print("\n✅ Feature extraction complete.")
    print("📦 Saved features to:", os.path.join(save_dir, "features.npy"))
    print("📦 Saved labels to:  ", os.path.join(save_dir, "labels.npy"))
    print(f"\n📊 Total files processed: {len(features_list)}")
    print(f"⚠️  Total files skipped:   {skipped_files}")
    print("\n🔢 Label distribution:")
    for label in sorted(label_counts.keys()):
        print(f"   {label}: {label_counts[label]} samples (skipped: {skip_counts[label]})")

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio dataset.")
    parser.add_argument("--data_dir", default="data/raw", help="Directory with raw .wav files")
    parser.add_argument("--save_dir", default="data/processed",
                        help="Directory to save features and labels")

    args = parser.parse_args()

    extract_all_features(data_dir=args.data_dir, save_dir=args.save_dir)
