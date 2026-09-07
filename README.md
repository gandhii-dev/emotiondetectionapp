# 🧠 Emotion Detection App
An **AI-powered Emotion Detection Application** built with Python and Streamlit that detects emotions from **text, speech, and facial expressions**.
The project combines **Natural Language Processing (NLP), Speech Recognition, and Computer Vision** into a single interactive web application.
---
## 🚀 Features

* 💬 **Text Emotion Detection**
* 🎙️ **Speech-to-Text Emotion Detection**
* 😊 **Facial Emotion Detection**
* 📊 Emotion confidence scores
* 🎨 Interactive Streamlit interface
* 🤖 Pre-trained Transformer model
* 🔄 Keyword-based fallback for text analysis
* 📷 Face detection using OpenCV
* ⚡ Cached AI model loading for better performance
---
## 🛠️ Tech Stack

| Technology                    | Purpose                             |
| ----------------------------- | ----------------------------------- |
| **Python**                    | Core programming language           |
| **Streamlit**                 | Web application and UI              |
| **Hugging Face Transformers** | Text emotion classification         |
| **DistilRoBERTa**             | Pre-trained emotion detection model |
| **SpeechRecognition**         | Speech-to-text conversion           |
| **Google Speech Recognition** | Converts speech into text           |
| **FER**                       | Facial expression/emotion detection |
| **OpenCV**                    | Face detection and image processing |
| **NumPy**                     | Image/array processing              |
| **Pillow (PIL)**              | Image handling                      |
---
## 🧩 How It Works
The application supports three emotion-detection approaches:
### 1️⃣ Text Emotion Detection
The user enters a sentence into the application.
```text
User Text
   ↓
Hugging Face Transformer
   ↓
Emotion Scores
   ↓
Highest-Scoring Emotion
   ↓
Result + Confidence
```
Example:
```text
Input:
"I got selected for my dream job. I am extremely happy!"
Output:
😄 JOY
```
The application uses:
```text
j-hartmann/emotion-english-distilroberta-base
```
as the pre-trained emotion classification model.
---
### 2️⃣ 🎙️ Speech Emotion Detection
The speech pipeline first converts speech into text.
```text
Audio File
    ↓
SpeechRecognition
    ↓
Google Speech Recognition
    ↓
Text Transcript
    ↓
Emotion Classification
    ↓
Detected Emotion
```
For example:
```text
Audio:
"I am really excited about this opportunity."

        ↓

Transcript:
"I am really excited about this opportunity."

        ↓

Emotion:
😄 JOY
```
The current implementation detects the emotion from the **transcribed text**, rather than analyzing vocal characteristics such as pitch or tone.
---
### 3️⃣ 😊 Facial Emotion Detection
The application uses the **FER (Facial Expression Recognition)** library to analyze facial expressions.
```text
Image
  ↓
FER
  ↓
Face Detection
  ↓
Facial Expression Analysis
  ↓
Emotion Scores
  ↓
Dominant Emotion
```
Example:
```text
😊 HAPPY — 94%
```
## OpenCV Haar Cascade is also used for face detection, particularly as a fallback when FER does not return emotion results.
## 🧠 Text Detection Fallback
If the Transformer model cannot be loaded, the application uses a **keyword-based emotion detection system**.
For example:
```text
happy, joyful, excited → Joy
sad, crying, lonely → Sadness
angry, furious, irritated → Anger
scared, terrified, anxious → Fear
surprised, shocked → Surprise
```
The system counts matching keywords and calculates emotion scores. If no emotion-related keyword is found, it returns **Neutral**.
---
## 📊 Confidence Scores
After detecting the emotion, the application displays confidence scores for different emotions.
Example:
```text
Joy        92%
Surprise    4%
Neutral     2%
Sadness     1%
Anger       1%
```
The emotion with the highest score is selected as the dominant emotion.
---
## 🎨 User Interface
The interface is created using **Streamlit** and customized with CSS.
It includes:
* 🧠 Application header
* 🎛️ Detection mode selector
* 💬 Text input
* 🎙️ Audio uploader
* 📊 Confidence bars
* 😊 Emotion result cards
* 📱 Responsive wide layout
---
## 📂 Project Structure
```text
emotion-detection-app/
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
    └── screenshots/
```
> Rename `app.py` according to the actual Python filename in your repository.
---
## ⚙️ Installation
### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```
### 2. Open the project
```bash
cd emotion-detection-app
```
### 3. Create a virtual environment
```bash
python -m venv venv
```
### 4. Activate the environment
**Windows:**
```bash
venv\Scripts\activate
```
**macOS/Linux:**
```bash
source venv/bin/activate
```
### 5. Install dependencies
```bash
pip install -r requirements.txt
```
---
## ▶️ Run the Application
Run:
```bash
streamlit run app.py
```
The Streamlit application will open in your browser.
---
## 📦 Requirement's
Create a `requirements.txt` file containing the required libraries, for example:
```text
streamlit
numpy
opencv-python
Pillow
transformers
torch
SpeechRecognition
fer
```
For MP3 processing, additional audio dependencies may be required depending on your environment.
---
## 🔄 Overall Architecture

```text
                    🧠 EMOTION DETECTION APP
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          💬 TEXT          🎙️ SPEECH        😊 FACE
             │                │                │
             │                ▼                │
             │          Speech-to-Text         │
             │                │                │
             ▼                ▼                ▼
       Transformer       Transformer           FER
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     Emotion Prediction
                              │
                              ▼
                    📊 Confidence Scores
                              │
                              ▼
                       Streamlit UI
```
---
## 🌟 Key Highlights
* Uses a **pre-trained Transformer model** instead of training an NLP model from scratch.
* Supports **multiple input modalities**.
* Provides emotion confidence scores.
* Includes a fallback mechanism when the Transformer model is unavailable.
* Uses OpenCV for computer vision operations.
* Provides an easy-to-use web interface through Streamlit.
---
## 🔮 Future Improvements
Possible future improvements include:
* 🎙️ Direct voice-emotion recognition using audio features
* 📹 Real-time webcam emotion detection
* 🌐 Multi-language text emotion detection
* 🧠 Fine-tuning the model on a custom dataset
* 📈 Emotion history and analytics
* 👥 Multi-face emotion tracking
* ☁️ Cloud deployment
* 🔐 Privacy-focused local processing
* 📊 Emotion trends and visualization
---
## ⚠️ Limitation's
* Speech emotion detection currently works by **transcribing speech into text and analyzing the transcript**.
* Keyword detection is only a fallback and is less sophisticated than the Transformer model.
* Facial emotion recognition can be affected by lighting, camera quality, face angle, and occlusion.
* Emotion prediction is an AI estimation and should not be treated as a definitive assessment of a person's actual emotional state.
---
## ⭐ Support
If you find this project useful, consider giving the repository a ⭐ on GitHub.
---
## 📄 License
This project is intended for educational and demonstration purposes.
