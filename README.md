# 🎤 Real-Time GenAI TelePrompter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange.svg)](https://github.com/openai/whisper)

> **AI-Powered Sales Coach with Real-Time Speech Transcription and Intelligent Coaching Suggestions**

A cutting-edge AI application that provides real-time speech transcription and intelligent sales coaching suggestions to help sales professionals improve their performance during live conversations.

---

## 🌟 Features

### 🎯 Core Functionality

- **🎤 Real-Time Speech Transcription:** High-accuracy speech-to-text using OpenAI Whisper.
- **🤖 AI-Powered Coaching:** GPT-4o generates contextual sales suggestions.
- **🌍 Multi-Language Support:** 13+ languages with auto-detection.
- **📱 Professional UI:** Dark-themed, responsive web interface.
- **🔄 Real-Time Processing:** 5-second audio cycles for immediate feedback.

### 🚀 Advanced Features

- **Dual AI System:** GPT-4o intelligence with rule-based fallback.
- **Conversation Context:** AI understands conversation flow.
- **Debug Mode:** Comprehensive troubleshooting tools.
- **Session Export:** JSON and TXT format downloads.
- **Audio Quality Monitoring:** Real-time audio feedback.
- **Visual Status Indicators:** Clear recording states.

---

## 📸 Screenshots

<!-- Add screenshots of your app here for better illustration! -->
<!-- Example:
![Main Interface](screenshots/main_interface.png)
![Debug Mode](screenshots/debug_mode.png)
-->

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Microphone access
- Internet connection
- OpenAI API key (optional, for AI features)

### Quick Start

1. **Clone the repository**
    ```
    git clone https://github.com/Shruti192903/Real_Time_GenAI_TelePrompter.git
    cd Real_Time_GenAI_TelePrompter
    ```
2. **Create virtual environment**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```
4. **Set up OpenAI API Key (Optional)**
    - **Method 1:** Set environment variable
        ```
        export OPENAI_API_KEY="your-api-key-here"
        ```
    - **Method 2:** Streamlit secrets
        - Create a folder named `.streamlit` in your project root, and inside it, create a file named `secrets.toml`.
        ```
        # .streamlit/secrets.toml
        OPENAI_API_KEY = "your-api-key-here"
        ```
5. **Run the application**
    ```
    streamlit run app.py
    ```
6. **Open in browser**
    - Navigate to [http://localhost:8501](http://localhost:8501)
    - Allow microphone permissions when prompted

---

## 🗂️ Project Structure
``` 
Real_Time_GenAI_TelePrompter/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

## ⚙️ Usage

-   Click **Start Recording** to begin capturing audio in real time (5-second cycles).
-   View live **transcripts** and **AI sales suggestions** side-by-side.
-   Switch between **AI-powered** and **rule-based** suggestions in the sidebar.
-   Enable **Debug Mode** for troubleshooting and audio device info.
-   Export your session as **JSON** or **TXT** after stopping the recording.

## 💡 Tips

-   Speak clearly and minimize background noise.
-   Use Debug Mode if you have issues with audio or transcription.
-   For best AI suggestions, provide your OpenAI API key in the sidebar.

## 📝 `requirements.txt`
```
  streamlit
  openai
  whisper
  sounddevice
  numpy
  wave
```

  ## ⭐ Acknowledgements

-   [OpenAI Whisper](https://github.com/openai/whisper)
-   [Streamlit](https://streamlit.io/)
-   [OpenAI GPT-4o](https://openai.com/)
