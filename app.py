#   EMOTION DETECTION APP
#   Detects emotions from: Text | Speech | Facial Expressions
#   Built with Streamlit, Transformers, SpeechRecognition, FER
# =============================================================
# """

import streamlit as st
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    .hero-header h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.3rem; }
    .hero-header p  { font-size: 1.05rem; opacity: 0.9; margin: 0; }

    /* Mode cards */
    .mode-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .mode-card:hover { box-shadow: 0 8px 24px rgba(102,126,234,0.2); }

    /* Emotion result box */
    .emotion-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(245, 87, 108, 0.3);
    }
    .emotion-result .emotion-label {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .emotion-result .emotion-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }

    /* Confidence bar container */
    .confidence-section {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #e8ecff;
    }
    .confidence-label {
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }

    /* Emotion bar */
    .emotion-bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
        gap: 0.6rem;
    }
    .emotion-bar-name {
        width: 80px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #4a5568;
    }
    .emotion-bar-bg {
        flex: 1;
        height: 10px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }
    .emotion-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .emotion-bar-pct {
        width: 45px;
        text-align: right;
        font-size: 0.82rem;
        font-weight: 600;
        color: #718096;
    }

    /* Section header */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }

    /* Info banner */
    .info-banner {
        background: #ebf8ff;
        border-left: 4px solid #4299e1;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        color: #2c5282;
        font-size: 0.9rem;
    }

    /* Stframe image override */
    img { border-radius: 10px; }

    /* Sidebar */
    .css-1d391kg { background: #f7f8ff; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2ff 0%, #f8f8ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EMOTION UTILS
# ─────────────────────────────────────────────

EMOTION_COLORS = {
    "joy":        "#f6c90e",
    "happy":      "#f6c90e",
    "happiness":  "#f6c90e",
    "sadness":    "#4299e1",
    "sad":        "#4299e1",
    "anger":      "#fc4444",
    "angry":      "#fc4444",
    "disgust":    "#48bb78",
    "fear":       "#9f7aea",
    "surprise":   "#ed8936",
    "neutral":    "#a0aec0",
    "contempt":   "#718096",
}

EMOTION_EMOJI = {
    "joy":       "😄",
    "happy":     "😄",
    "happiness": "😄",
    "sadness":   "😢",
    "sad":       "😢",
    "anger":     "😠",
    "angry":     "😠",
    "disgust":   "🤢",
    "fear":      "😨",
    "surprise":  "😲",
    "neutral":   "😐",
    "contempt":  "😒",
}

def get_emotion_color(emotion: str) -> str:
    return EMOTION_COLORS.get(emotion.lower(), "#667eea")

def get_emotion_emoji(emotion: str) -> str:
    return EMOTION_EMOJI.get(emotion.lower(), "🤔")

def render_emotion_result(emotion: str, source_label: str = "Detected Emotion"):
    """Render a large colourful result card."""
    emoji  = get_emotion_emoji(emotion)
    st.markdown(f"""
    <div class="emotion-result">
        <div style="font-size:3.5rem; margin-bottom:0.3rem;">{emoji}</div>
        <div class="emotion-label">{emotion.upper()}</div>
        <div class="emotion-subtitle">{source_label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_confidence_bars(scores: dict, title: str = "Confidence Scores"):

    st.subheader(title)

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for emo, score in sorted_scores:

        pct = round(score * 100, 1)

        st.write(f"{emo.capitalize()} : {pct}%")

        st.progress(float(score))
    


# ─────────────────────────────────────────────
# TEXT EMOTION DETECTION
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_text_model():
    """Load the text emotion classifier (HuggingFace)."""
    try:
        from transformers import pipeline
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1   # CPU
        )
        return classifier, "transformer"
    except Exception:
        pass
    # Fallback: simple keyword approach
    return None, "keyword"

KEYWORD_EMOTIONS = {
    "joy":     ["happy", "joy", "joyful", "delighted", "elated", "ecstatic", "cheerful",
                "glad", "pleased", "wonderful", "amazing", "love", "great", "excited", "thrilled"],
    "sadness": ["sad", "unhappy", "depressed", "cry", "crying", "tears", "heartbroken",
                "lonely", "miserable", "gloomy", "sorrow", "grief", "disappointed", "down"],
    "anger":   ["angry", "furious", "rage", "mad", "annoyed", "irritated", "outraged",
                "frustrated", "hate", "infuriated", "livid", "enraged"],
    "fear":    ["scared", "afraid", "fearful", "terrified", "anxious", "nervous", "worried",
                "panic", "dread", "horrified", "frightened", "terror"],
    "surprise":["surprised", "shocked", "astonished", "amazed", "stunned", "unexpected",
                "wow", "omg", "unbelievable", "incredible"],
    "disgust": ["disgusting", "disgusted", "gross", "revolting", "nauseating", "yuck",
                "awful", "horrible", "repulsed"],
    "neutral": ["okay", "ok", "fine", "alright", "normal", "regular", "average", "usual"],
}

def keyword_emotion(text: str) -> dict:
    """Simple keyword fallback when transformers unavailable."""
    text_lower = text.lower()
    scores = {emo: 0.0 for emo in KEYWORD_EMOTIONS}
    total  = 0
    for emo, words in KEYWORD_EMOTIONS.items():
        for w in words:
            if w in text_lower:
                scores[emo] += 1
                total += 1
    if total == 0:
        scores["neutral"] = 1.0
        return scores
    for emo in scores:
        scores[emo] = round(scores[emo] / total, 4)
    return scores

def detect_text_emotion(text: str):
    """Return (top_emotion, scores_dict)."""
    classifier, mode = load_text_model()
    if mode == "transformer" and classifier is not None:
        result = classifier(text)[0]
        scores = {item["label"].lower(): round(item["score"], 4) for item in result}
        top    = max(scores, key=scores.get)
        return top, scores
    else:
        scores = keyword_emotion(text)
        top    = max(scores, key=scores.get)
        return top, scores


# ─────────────────────────────────────────────
# SPEECH EMOTION DETECTION
# ─────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Use SpeechRecognition (Google) to transcribe audio."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    # Write to a temp file
    suffix = os.path.splitext(filename)[-1] if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.warning(f"Speech recognition service error: {e}")
        return ""
    finally:
        os.unlink(tmp_path)

def detect_speech_emotion(audio_bytes: bytes, filename: str = "audio.wav"):
    """Transcribe → run text emotion. Returns (transcript, top_emotion, scores)."""
    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(audio_bytes, filename)
    if not transcript:
        return None, None, None
    with st.spinner("Analysing emotions in transcript..."):
        top, scores = detect_text_emotion(transcript)
    return transcript, top, scores


# ─────────────────────────────────────────────
# FACIAL EMOTION DETECTION
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_face_detector():
    """Load OpenCV Haar Cascade for face detection."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def detect_faces_opencv(img_array: np.ndarray):
    """Return list of (x, y, w, h) bounding boxes."""
    face_cascade = load_face_detector()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return faces if len(faces) > 0 else []

def detect_facial_emotion_fer(img_array: np.ndarray):
    """
    Use the 'fer' library (Facial Expression Recognition) for emotion detection.
    Returns (annotated_image, face_results_list).
    face_results_list: [{"box": ..., "emotions": {...}, "dominant_emotion": ...}, ...]
    """
    try:
        from fer import FER
        detector = FER(mtcnn=False)  # mtcnn=False is faster and avoids TF overhead
        results  = detector.detect_emotions(img_array)
    except ImportError:
        # FER not available – fallback to OpenCV only
        results = []

    annotated = img_array.copy()
    face_results = []

    for face in results:
        box    = face["box"]          # [x, y, w, h]
        emos   = face["emotions"]     # {"angry": 0.02, "disgust": ..., ...}
        top_emo = max(emos, key=emos.get)
        x, y, w, h = box

        # Draw bounding box
        color = tuple(int(c) for c in bytes.fromhex(
            get_emotion_color(top_emo).lstrip("#")
        ))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        # Draw label background
        label      = f"{get_emotion_emoji(top_emo)} {top_emo.upper()} ({round(emos[top_emo]*100)}%)"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
        cv2.putText(annotated, label, (x + 4, y - 5), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        face_results.append({
            "box":              box,
            "emotions":         emos,
            "dominant_emotion": top_emo
        })

    # If no FER results, still try OpenCV face detection boxes
    if not face_results:
        faces = detect_faces_opencv(img_array)
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (102, 126, 234), 3)
            cv2.putText(annotated, "Face detected", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated, face_results


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.2rem;font-weight:700;color:#4a5568;">Emotion Detector</div>
        <div style="font-size:0.8rem;color:#a0aec0;margin-top:0.2rem;">AI-Powered Analysis</div>
    </div>
    <hr style="border-color:#e2e8f0;margin:1rem 0;">
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Detection Mode",
        ["💬 Text","🎙️ Speech"],
        index=0,
    )

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>🧠 Emotion Detection App</h1>
    <p>Analyse emotions from text, speech, and facial expressions using AI</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODE: TEXT
# ─────────────────────────────────────────────

if "💬 Text" in mode:
    st.markdown('<div class="section-header">💬 Text Emotion Detection</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        Type or paste any text below. The AI model will analyse the emotional content
        and return a confidence score for each emotion category.
    </div>
    """, unsafe_allow_html=True)

    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("*Try an example:*")
        examples = {
            "😄 Happy":    "I just got promoted at work! I'm so incredibly happy and excited about what the future holds!",
            "😢 Sad":      "I miss my old friend so much. It's been months and I still feel this deep sadness every single day.",
            "😠 Angry":    "I can't believe how unfair this situation is. I am absolutely furious and I won't stand for this anymore.",
            "😨 Fear":     "There was a strange noise outside and I was completely terrified. My heart was racing all night.",
            "😲 Surprise": "I had no idea they were planning a party! I was completely shocked and overwhelmed when everyone jumped out.",
            "🤢 Disgust":  "The food was revolting and the whole place smelled terrible. I was disgusted and had to leave immediately.",
        }
        chosen = st.selectbox("", list(examples.keys()), label_visibility="collapsed")
        if st.button("Use this example", use_container_width=True):
            st.session_state["text_input"] = examples[chosen]

    with col_input:
        text_value = st.session_state.get("text_input", "")
        user_text  = st.text_area(
            "Enter text to analyse:",
            value=text_value,
            height=140,
            placeholder="Type something like: 'I am so happy today, life feels wonderful!'",
            label_visibility="collapsed"
        )

    analyze_btn = st.button("🔍 Detect Emotion", type="primary", use_container_width=False)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analysing text emotions..."):
                top_emotion, scores = detect_text_emotion(user_text.strip())

            render_emotion_result(top_emotion, "Detected from Text")
            render_confidence_bars(scores, "Confidence Scores per Emotion")
            
# ─────────────────────────────────────────────
# MODE: SPEECH
# ─────────────────────────────────────────────

if "🎙️ Speech" in mode:

    st.markdown(
        '<div class="section-header">🎙️ Speech Emotion Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-banner">
        Upload an audio file (.wav recommended).
        The system will convert speech to text and detect emotion.
    </div>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3"]
    )

    if audio_file is not None:

        audio_bytes = audio_file.read()

        with st.spinner("Processing speech..."):

            transcript, top_emotion, scores = detect_speech_emotion(
                audio_bytes,
                audio_file.name
            )

        if transcript:

            st.subheader("Transcribed Text")
            st.write(transcript)

            render_emotion_result(
                top_emotion,
                "Detected from Speech"
            )

            render_confidence_bars(
                scores,
                "Confidence Scores per Emotion"
            )

        else:

            st.error(
                "Could not transcribe audio. Try a clearer recording."
            )
            # Show which model was used
            _, mode_label = load_text_model()
            model_note = (
                "🤖 Powered by DistilRoBERTa emotion classifier"
                if mode_label == "transformer"
                else "📝 Using keyword-based fallback (install transformers for the full model)"
            )
            # st.caption(model_note)