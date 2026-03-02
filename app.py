import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

# ==========================
# Page Config
# ==========================

st.set_page_config(page_title="Voice Anti-Spoofing", layout="wide")

# ==========================
# Load Model
# ==========================

model = joblib.load("voice_model.pkl")

# ==========================
# Feature Extraction
# ==========================

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return np.mean(combined.T, axis=0)

# ==========================
# Segment Detection
# ==========================

def segment_level_detection(file_path, segment_duration=1):
    audio, sr = librosa.load(file_path, sr=16000)
    segment_length = segment_duration * sr

    predictions = []
    timestamps = []

    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]

        if len(segment) < segment_length:
            continue

        segment = librosa.util.normalize(segment)

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        features = np.mean(combined.T, axis=0)

        prediction = model.predict([features])[0]

        predictions.append(prediction)
        timestamps.append(start / sr)

    return timestamps, predictions

# ==========================
# UI
# ==========================

st.title("🎙 Voice Anti-Spoofing System")

option = st.radio("Choose Input:", ["Upload Audio", "Record Live"])

file_path = None

# ==========================
# Upload Option
# ==========================

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload WAV/FLAC/MP3", type=["wav", "flac", "mp3"])
    if uploaded_file:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        file_path = "temp.wav"
        st.audio(file_path)

# ==========================
# Record Option
# ==========================

elif option == "Record Live":
    duration = st.slider("Recording Duration", 2, 10, 5)
    if st.button("Start Recording"):
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("live_recording.wav", fs, recording)
        file_path = "live_recording.wav"
        st.audio(file_path)

# ==========================
# Detection Section
# ==========================

if file_path:

    # Load audio for visualization
    audio, sr = librosa.load(file_path, sr=16000)

    # Prediction
    features = extract_features(file_path)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]

    confidence = probabilities[0] if prediction == 0 else probabilities[1]

    if prediction == 0:
        st.success("Prediction: REAL VOICE")
    else:
        st.error("Prediction: SYNTHETIC VOICE")

    st.write(f"Confidence: {round(confidence*100,2)}%")

    # ======================
    # Probability Bar Graph
    # ======================

    st.subheader("Prediction Probability")

    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(["Real", "Synthetic"],
                [probabilities[0]*100, probabilities[1]*100])
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_ylim([0, 100])
    st.pyplot(fig_prob)

    # ======================
    # Waveform
    # ======================

    st.subheader("Waveform")

    fig_wave, ax_wave = plt.subplots(figsize=(10,3))
    librosa.display.waveshow(audio, sr=sr, ax=ax_wave)
    ax_wave.set_title("Audio Waveform")
    st.pyplot(fig_wave)

    # ======================
    # Spectrogram
    # ======================

    st.subheader("Spectrogram")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    fig_spec, ax_spec = plt.subplots(figsize=(10,4))
    img = librosa.display.specshow(D, sr=sr,
                                   x_axis='time',
                                   y_axis='log',
                                   ax=ax_spec)
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title("Log Spectrogram")
    st.pyplot(fig_spec)

    # ======================
    # Segment-Level Analysis
    # ======================

    st.subheader("Segment-Level Analysis")

    timestamps, segment_preds = segment_level_detection(file_path)

    total_segments = len(segment_preds)
    synthetic_segments = sum(segment_preds)
    synthetic_percentage = (synthetic_segments / total_segments) * 100 if total_segments > 0 else 0

    st.metric("Synthetic Content Percentage",
              f"{round(synthetic_percentage,2)}%")

    if synthetic_percentage > 20:
        st.error("⚠️ ALERT: Significant Synthetic Content Detected")
    elif synthetic_percentage > 5:
        st.warning("⚠️ Partial Synthetic Content Detected")
    else:
        st.success("Mostly Real Audio")

    colors = ["green" if p == 0 else "red" for p in segment_preds]

    fig_seg, ax_seg = plt.subplots(figsize=(12,2))
    ax_seg.bar(timestamps,
               [1]*len(segment_preds),
               width=0.8,
               color=colors)

    ax_seg.set_yticks([])
    ax_seg.set_xlabel("Time (seconds)")
    ax_seg.set_title("Green = Real | Red = Synthetic")
    st.pyplot(fig_seg)
