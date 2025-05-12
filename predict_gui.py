import os
import json
import joblib
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from ttkbootstrap import Window, Style
from ttkbootstrap import ttk
from ttkbootstrap.widgets import Meter
from src.feature_extraction import extract_features_from_file
import platform

# Load model, scaler, and labels
MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_MAP_PATH = "models/label_map.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
labels = list(label_map.keys())

# Last used audio file path
last_audio_file = [None]


# --- Prediction ---


def predict(file_path):
    try:
        features = extract_features_from_file(file_path)
        if features is None:
            return "Invalid audio", 0
        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_index = np.argmax(probabilities)
        predicted_label = model.classes_[predicted_index]
        confidence = probabilities[predicted_index] * 100
        return predicted_label, confidence
    except Exception as e:
        return f"Prediction failed: {e}", 0

# --- GUI Functions ---


def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select a WAV file",
        filetypes=[("WAV files", "*.wav")]
    )
    if file_path:
        last_audio_file[0] = file_path
        predicted_label, confidence = predict(file_path)
        result_var.set(f"Predicted Letter: {predicted_label}")
        confidence_meter.configure(amountused=confidence)
        status_var.set(f"Processed: {os.path.basename(file_path)}")


def play_last_audio():
    if last_audio_file[0] and Path(last_audio_file[0]).is_file():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(last_audio_file[0])
            pygame.mixer.music.play()
            status_var.set(f"Playing: {os.path.basename(last_audio_file[0])}")
        except Exception as e:
            status_var.set(f"Failed to play: {str(e)}")
    else:
        status_var.set("No audio file loaded yet.")


def record_audio():
    try:
        fs = 22050
        duration = 1  # seconds
        status_var.set("🎙️ Recording 1 seconds...")
        app.update()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = tmpfile.name
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        sf.write(audio_path, recording, fs)
        last_audio_file[0] = audio_path
        status_var.set(f"Recorded to: {audio_path}")
        predicted_label, confidence = predict(audio_path)
        result_var.set(f"Predicted Letter: {predicted_label}")
        confidence_meter.configure(amountused=confidence)
    except Exception as e:
        status_var.set(f"Recording failed: {e}")


# --- GUI Setup ---


app = Window(themename="superhero")
app.title("Arabic Letter Classifier")

# Maximize window cross-platform
if platform.system() == "Windows":
    app.state("zoomed")
elif platform.system() == "Linux":
    app.attributes("-zoomed", True)
else:
    app.attributes("-fullscreen", True)

style = Style()
style.configure("TButton", font=("Helvetica", 16))
style.configure("TLabel", font=("Helvetica", 18))

main_frame = ttk.Frame(app, padding=20)
main_frame.pack(expand=True, fill="both")

# Title
title_label = ttk.Label(
    main_frame,
    text="Arabic Letter Classifier",
    font=("Helvetica", 24, "bold"),
    anchor="center"
)
title_label.pack(pady=10)

# Browse button
browse_button = ttk.Button(
    main_frame,
    text="Browse WAV File",
    command=browse_file,
    width=30,
    bootstyle="primary"
)
browse_button.pack(pady=10)

# Record button
record_button = ttk.Button(
    main_frame,
    text="Record 1 Seconds",
    command=record_audio,
    width=30,
    bootstyle="info"
)
record_button.pack(pady=10)

# Play button
play_button = ttk.Button(
    main_frame,
    text="Play Last Audio",
    command=play_last_audio,
    width=30,
    bootstyle="secondary"
)
play_button.pack(pady=10)

# Result label
result_var = tk.StringVar(value="Predicted Letter: ")
result_label = ttk.Label(
    main_frame,
    textvariable=result_var,
    font=("Helvetica", 20),
    anchor="center"
)
result_label.pack(pady=10)

# Confidence meter
confidence_meter = Meter(
    main_frame,
    bootstyle="success",
    subtext="Confidence",
    interactive=False,
    amountused=0,
    amounttotal=100,
    meterthickness=20,
    metersize=200,
    textright="%",
    textfont=("Helvetica", 16, "bold")
)
confidence_meter.pack(pady=20)

# Status bar
status_var = tk.StringVar(value="Ready")
status_bar = ttk.Label(
    app,
    textvariable=status_var,
    anchor="w",
    bootstyle="secondary"
)
status_bar.pack(side="bottom", fill="x")

app.mainloop()
