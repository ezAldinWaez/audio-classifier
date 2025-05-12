# 🎧 Arabic Letter Audio Classifier

A machine learning project to classify spoken Arabic letters from audio recordings using Python. It features both a training pipeline and a modern GUI for real-time prediction.

## 🚀 Features

- 🎙 Record your voice and predict the letter you spoke
- 🔊 Browse and classify `.wav` files
- 📈 Extracts robust audio features (MFCCs, spectral features)
- 🧠 Trained with an SVM classifier
- 💻 Built with a modern Tkinter GUI (ttkbootstrap)
- 📊 Visualization of prediction confidence

## ⚙️ Installation

```bash
git clone https://github.com/ezAldinWaez/audio-classifier.git
cd audio-classifier
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🧪 Step-by-Step Usage

### 1. 🔧 Extract Features

```bash
python src/feature_extraction.py
```

### 2. 🧠 Train the Model

```bash
python src/train_model.py
```

### 3. 🖼 Launch the GUI

```bash
python predict_gui.py
```

## 🎮 GUI Functionality

- **📂 Browse WAV** – Select any `.wav` file for prediction.
- **🎙 Record** – Capture 5 seconds of your voice and classify it.
- **🔊 Play** – Replay the most recent audio used.
- **Confidence Meter** – Visual feedback on prediction certainty.

## 📦 Dependencies

Install via:

```bash
pip install -r requirements.txt
```

Key packages:

- `librosa`
- `scikit-learn`
- `ttkbootstrap`
- `pygame`
- `sounddevice`, `soundfile`
- `joblib`, `numpy`

## 🧾 License

This project is licensed under the MIT License.
