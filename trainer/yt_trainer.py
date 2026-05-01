import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import pipeline


def parse_urls(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    return re.findall(r"https?://[^\s\"']+", content)


def download_videos(urls, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    if not urls:
        return
    if shutil.which("yt-dlp") is None:
        logging.warning("yt-dlp is not installed. Skipping YouTube downloads.")
        return
    for url in urls:
        logging.info("Downloading %s", url)
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", os.path.join(download_dir, "%(title)s.%(ext)s"), url],
            check=False
        )


def extract_audio(video_path, audio_path):
    if shutil.which("ffmpeg") is None:
        logging.warning("ffmpeg is not installed. Skipping audio extraction for %s", video_path)
        return False
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and os.path.exists(audio_path)


def load_wav(audio_path):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def transcribe_audio(video_path, asr_pipeline):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")
        if not extract_audio(video_path, audio_path):
            return ""
        audio, sample_rate = load_wav(audio_path)
        if audio.size == 0:
            return ""
        result = asr_pipeline(audio, sampling_rate=sample_rate)
        return result["text"] if isinstance(result, dict) else str(result)


def extract_frame_features(video_path, frame_stride=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning("Unable to open video file: %s", video_path)
        return {"frame_mean": 0.0, "frame_std": 0.0, "frame_count": 0}

    means = []
    stds = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(np.mean(gray)))
            stds.append(float(np.std(gray)))
        frame_index += 1

    cap.release()
    if not means:
        return {"frame_mean": 0.0, "frame_std": 0.0, "frame_count": 0}
    return {
        "frame_mean": float(np.mean(means)),
        "frame_std": float(np.mean(stds)),
        "frame_count": len(means)
    }


def get_video_label(video_path, video_dir):
    parent_label = os.path.basename(os.path.dirname(video_path))
    base_label = os.path.splitext(os.path.basename(video_path))[0]
    video_dir_name = os.path.basename(os.path.abspath(video_dir))
    if parent_label and parent_label != video_dir_name:
        return parent_label
    return base_label


def collect_video_samples(video_dir, asr_model, frame_stride):
    video_files = [
        os.path.join(video_dir, file)
        for file in os.listdir(video_dir)
        if file.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
    ]
    if not video_files:
        logging.warning("No video files found in %s", video_dir)
        return pd.DataFrame(), []

    asr_pipeline = pipeline("automatic-speech-recognition", model=asr_model)
    rows = []
    labels = []
    for video_path in video_files:
        transcript_path = f"{os.path.splitext(video_path)[0]}.txt"
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as file:
                transcript = file.read().strip()
        else:
            transcript = transcribe_audio(video_path, asr_pipeline)

        frame_features = extract_frame_features(video_path, frame_stride=frame_stride)
        label = get_video_label(video_path, video_dir)

        rows.append({
            "transcript": transcript,
            "frame_mean": frame_features["frame_mean"],
            "frame_std": frame_features["frame_std"],
            "frame_count": frame_features["frame_count"]
        })
        labels.append(label)

    return pd.DataFrame(rows), labels


def train_and_save_model(df, labels, output_path):
    if df.empty:
        logging.warning("No samples available to train.")
        return
    if len(set(labels)) < 2:
        logging.warning("Not enough classes to train a classifier.")
        return

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(), "transcript"),
            ("numeric", StandardScaler(), ["frame_mean", "frame_std", "frame_count"])
        ]
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info("Training completed with accuracy: %.4f", accuracy)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Model saved to %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_video_dir = os.path.join(repo_root, "Data source", "youtube_videos")
    default_urls_file = os.path.join(repo_root, "Data source", "YS.md")
    default_output_model = os.path.join(repo_root, "pre-trained model", "yt_model.pkl")

    parser = argparse.ArgumentParser(description="Train a YouTube model using video frames and audio transcripts.")
    parser.add_argument("--video-dir", default=default_video_dir, help="Directory containing downloaded YouTube videos")
    parser.add_argument("--urls-file", default=default_urls_file, help="Path to YS.md or ruf.text URL list")
    parser.add_argument("--output-model", default=default_output_model, help="Path to save the trained model")
    parser.add_argument("--frame-stride", type=int, default=30, help="Frame sampling stride")
    parser.add_argument("--asr-model", default="openai/whisper-small", help="ASR model for speech recognition")
    args = parser.parse_args()

    urls = parse_urls(args.urls_file)
    download_videos(urls, args.video_dir)

    df, labels = collect_video_samples(args.video_dir, args.asr_model, args.frame_stride)
    train_and_save_model(df, labels, args.output_model)


if __name__ == "__main__":
    main()
