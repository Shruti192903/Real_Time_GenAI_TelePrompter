# 🎤 Real-Time GenAI TelePrompter

**AI Sales Coach with Live Transcription ✨**

A lightweight Streamlit app that provides real-time speech transcription and AI-powered sales coaching suggestions during live conversations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Features

- **🎤 Real-time Transcription** - Live speech-to-text using local Whisper
- **🤖 AI Sales Coaching** - GPT-4o powered suggestions with rule-based fallback
- **🌍 Multi-language Support** - Auto-detection and translation capabilities
- **⏱️ Session Management** - Start/stop with timing and analytics
- **📦 Export Functionality** - Download session data in JSON/TXT formats
- **🎨 Professional UI** - Clean, modern interface with real-time updates

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Microphone access
- OpenAI API key (optional, for AI suggestions)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Real-Time-GenAI-TelePrompter.git
cd Real-Time-GenAI-TelePrompter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage
1. **Start the app** - Opens in your browser at `http://localhost:8501`
2. **Configure AI** - Enter OpenAI API key in sidebar (optional)
3. **Start Recording** - Click the recording button to begin
4. **View Live Results** - See transcription and coaching suggestions in real-time
5. **Export Session** - Download your session data when finished

## 🤖 AI Configuration

### Option 1: In-App Configuration
- Enter your OpenAI API key in the sidebar
- Toggle AI suggestions on/off as needed

### Option 2: Environment Variable
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 3: Secrets File
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Speech-to-Text**: OpenAI Whisper (local)
- **Translation**: Google Translator
- **AI Coaching**: OpenAI GPT-4o
- **Audio Processing**: sounddevice, wave, numpy

## 📊 Requirements Fulfillment

✅ **Audio Input** - Microphone access with start/stop controls  
✅ **Real-time Transcription** - Live STT with timestamps  
✅ **LLM Integration** - GPT-4o with context-aware suggestions  
✅ **Prompt Display** - Categorized coaching suggestions  
✅ **Session Handling** - Full session management and export  

## 🔒 Privacy & Security

- All processing runs locally (except OpenAI API calls)
- No data stored permanently without user action
- API keys handled securely through Streamlit secrets
- Audio files are temporary and automatically cleaned up

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI team for Whisper and GPT models
- Streamlit team for the excellent web framework
- Python community for robust audio processing libraries

---

**Built with ❤️ for sales professionals who want an AI wingman for their calls!**
