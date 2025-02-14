import streamlit as st
import whisper
import torch
import librosa
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import pipeline
from sentence_transformers import SentenceTransformer

model = whisper.load_model("tiny")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
topic_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def transcribe_audio(uploaded_file):
    audio, sr = librosa.load(uploaded_file, sr=16000)
    sf.write("temp.wav", audio, sr)
    result = model.transcribe("temp.wav", language=None)
    os.remove("temp.wav")
    return result["text"], result["language"]

def analyze_emotion(text):
    emotions = emotion_classifier(text[:512])  
    return [(e["label"], e["score"]) for e in emotions[0]]

def classify_topic(text):
    topics = ["news", "casual talk", "complaint", "storytelling", "business", "technical"]
    embeddings = topic_model.encode([text] + topics)
    similarities = np.dot(embeddings[0], np.transpose(embeddings[1:]))
    return topics[np.argmax(similarities)]

def summarize_text(text):
    summary = summarizer(text[:1024], max_length=200, min_length=100, do_sample=False)
    return summary[0]["summary_text"]

st.set_page_config(page_title="Silent Interpreter", layout="wide")

st.title("🗣️ Silent Interpreter: Speech-to-Text with Emotion & Context Analysis")
st.write("Upload an audio file.")

uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Transcribing..."):
        transcript, detected_lang = transcribe_audio(uploaded_file)

    st.subheader("Transcription:")
    st.write(transcript)
    st.write(f"🌍 Detected Language: **{detected_lang.upper()}**")

    with st.spinner("Analyzing Emotion..."):
        emotions = analyze_emotion(transcript)
    
    st.subheader("Emotion Analysis:")
    st.write(f"🎭 Most Likely Emotion: **{emotions[0][0]}**")
    
    emotion_labels = [e[0] for e in emotions]
    emotion_scores = [e[1] for e in emotions]
    fig, ax = plt.subplots()
    ax.bar(emotion_labels, emotion_scores)
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Emotion")
    st.pyplot(fig)

    with st.spinner("Detecting Context..."):
        detected_topic = classify_topic(transcript)

    st.subheader("📝 Context Category:")
    st.write(f"🔍 {detected_topic.capitalize()}")

    with st.spinner("Summarizing..."):
        summary = summarize_text(transcript)

    st.subheader("Summary:")
    st.write(summary)
