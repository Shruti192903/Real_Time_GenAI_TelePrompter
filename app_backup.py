import streamlit as st
import time
import tempfile
import json
import random
from datetime import datetime
import sounddevice as sd
import numpy as np
import wave
import os

# Initialize session state FIRST
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "last_record_time" not in st.session_state:
    st.session_state.last_record_time = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "enable_translation" not in st.session_state:
    st.session_state.enable_translation = False
if "source_language" not in st.session_state:
    st.session_state.source_language = "auto"
if "target_language" not in st.session_state:
    st.session_state.target_language = "en"
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "use_llm" not in st.session_state:
    st.session_state.use_llm = False
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Try to import libraries
try:
    import whisper
    # Try to load model with error handling
    with st.spinner("Loading Whisper model..."):
        WHISPER_MODEL = whisper.load_model("base")  
    LOCAL_WHISPER_AVAILABLE = True
    st.success("✅ Whisper model loaded successfully")
except Exception as e:
    LOCAL_WHISPER_AVAILABLE = False
    WHISPER_MODEL = None
    st.error(f"❌ Whisper loading error: {e}")

# Around line 590-595, replace with:

                    if transcript:
                        # Show success message
                        mode = "GPT-4o" if OPENAI_AVAILABLE else "Rule-based"
                        st.success(f"✅ Transcribed ({mode}): {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                        
                        # Add to transcript
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        entry = {
                            "timestamp": timestamp,
                            "original": transcript,
                            "language": detected_lang
                        }
                        st.session_state.transcript.append(entry)
                        
                        # Generate suggestions using original transcript
                        suggestion_text = transcript                        # Around line 580-620, replace the entire processing section:
                        
                                        # Process transcript
                                        if audio is not None:
                                            # Save audio to temporary file
                                            temp_filename = f"temp_audio_{int(time.time())}.wav"
                                            with wave.open(temp_filename, 'wb') as wf:
                                                wf.setnchannels(1)
                                                wf.setsampwidth(2)
                                                wf.setframerate(16000)
                                                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
                                            
                                            # Transcribe
                                            transcript, detected_lang = transcribe_audio(temp_filename)
                                            
                                            # Clean up temp file
                                            try:
                                                os.remove(temp_filename)
                                            except:
                                                pass
                                            
                                            if transcript:
                                                # Show success message
                                                mode = "GPT-4o" if OPENAI_AVAILABLE else "Rule-based"
                                                st.success(f"✅ Transcribed: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                                                
                                                # Add to transcript
                                                timestamp = datetime.now().strftime("%H:%M:%S")
                                                entry = {
                                                    "timestamp": timestamp,
                                                    "original": transcript,
                                                    "language": detected_lang
                                                }
                                                st.session_state.transcript.append(entry)
                                                
                                                # Generate AI suggestions
                                                if OPENAI_AVAILABLE:
                                                    ai_suggestion = get_ai_suggestion(transcript)
                                                    if ai_suggestion:
                                                        st.session_state.suggestions.append({
                                                            "timestamp": timestamp,
                                                            "suggestion": ai_suggestion,
                                                            "type": "GPT-4o"
                                                        })
                                                
                                                # Always add rule-based suggestions as backup
                                                rule_suggestions = get_sales_suggestion(transcript)
                                                for suggestion in rule_suggestions:
                                                    st.session_state.suggestions.append({
                                                        "timestamp": timestamp,
                                                        "suggestion": suggestion,
                                                        "type": "Rule-based"
                                                    })
                                            
                                            else:
                                                st.warning("⚠️ No speech detected. Try speaking louder or closer to the microphone.")
                                        
                                        else:
                                            st.error("❌ Failed to record audio. Check your microphone permissions.")try:
    import openai
    LLM_AVAILABLE = True
    
    # Check for API key from multiple sources
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        api_key = st.session_state.openai_api_key
    
    if api_key:
        openai.api_key = api_key
        LLM_ENABLED = True
    else:
        LLM_ENABLED = False
        
except ImportError:
    LLM_AVAILABLE = False
    LLM_ENABLED = False

# Page config
st.set_page_config(
    page_title="🎙️ Real-Time GenAI TelePrompter",
    layout="wide",
    # page_icon="🎙️"
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .recording-status {
        background: linear-gradient(90deg, #ff4444, #ff6666);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
    }
    .stopped-status {
        background: linear-gradient(90deg, #28a745, #34ce57);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
    }
    .transcript-entry {
        background: #2d2d2d;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .suggestion-entry {
        background: #2a4d3a;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid #81C784;
    }
    .debug-info {
        background: #1a1a2e;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #0f3460;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Check if Whisper is available
if not LOCAL_WHISPER_AVAILABLE:
    st.error("❌ Whisper not available. Install with: `pip install openai-whisper`")
    st.stop()

def check_audio_devices():
    """Check available audio devices"""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        return input_devices
    except Exception as e:
        st.error(f"Audio device error: {e}")
        return []

def get_llm_suggestion(text, conversation_history=""):
    """Generate AI suggestions using OpenAI GPT-4o"""
    if not LLM_ENABLED or not text.strip():
        return get_sales_suggestion(text)  # Fallback to rule-based
    
    try:
        # System prompt for sales coaching
        system_prompt = """You are an expert AI sales coach assistant helping a sales representative during a live call. 

Based on the conversation transcript, provide 1-2 SHORT, actionable suggestions to help the sales rep. 
Your suggestions should be:
- Brief (15 words max each)
- Immediately actionable 
- Sales-focused (closing, objection handling, discovery, etc.)
- Formatted with appropriate emoji tags: 💡 Tip, ⚠️ Reminder, ❗ Alert, 🎯 Opportunity, 🔥 Action

Focus on:
- Objection handling techniques
- Discovery questions to ask
- Closing opportunities 
- Urgency creation
- Value proposition reinforcement
- Next steps guidance

Return only the suggestion(s), no explanation."""

        # Prepare the conversation context
        full_context = conversation_history + "\n\nLatest: " + text if conversation_history else text
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation: {full_context}"}
            ],
            max_tokens=100,
            temperature=0.7,
            timeout=5  # 5 second timeout for real-time needs
        )
        
        suggestion = response.choices[0].message.content.strip()
        return [suggestion]
        
    except Exception as e:
        st.warning(f"AI suggestion error: {e}")
        # Fallback to rule-based suggestions
        return get_sales_suggestion(text)

def record_audio(duration=5, sample_rate=16000):
    """Record audio with improved settings"""
    try:
        # Use higher quality settings
        audio_data = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='float32',
            device=None,  # Use default device
            blocking=True  # Wait for recording to complete
        )
        
        # Normalize audio
        audio_data = audio_data.flatten()
        
        # Check if audio was actually recorded
        max_amplitude = np.max(np.abs(audio_data))
        if st.session_state.debug_mode:
            st.write(f"🎤 Audio recorded: duration={duration}s, max_amplitude={max_amplitude:.4f}")
        
        # Require minimum audio level
        if max_amplitude < 0.001:
            st.warning("⚠️ Very quiet audio detected. Speak louder or check microphone.")
            return None
            
        return audio_data
        
    except Exception as e:
        st.error(f"❌ Recording error: {e}")
        return None

def save_audio(audio, filename, samplerate=16000):
    """Save audio to WAV file with validation"""
    try:
        # Ensure audio is not empty
        if len(audio) == 0:
            st.error("Empty audio buffer")
            return False
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            # Convert to 16-bit integers
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        
        # Verify file was created and has content
        if os.path.exists(filename) and os.path.getsize(filename) > 44:  # WAV header is 44 bytes
            if st.session_state.debug_mode:
                st.write(f"💾 Audio saved: {os.path.getsize(filename)} bytes")
            return True
        else:
            st.error("Failed to save audio file properly")
            return False
            
    except Exception as e:
        st.error(f"Save audio error: {e}")
        return False

def transcribe_audio(audio_file_path):
    """Transcribe audio with better settings"""
    if not LOCAL_WHISPER_AVAILABLE:  # ✅ Fixed variable name
        return "Whisper not available", "unknown"
    
    try:
        # Use the pre-loaded model
        model = WHISPER_MODEL
        
        # Transcribe with better parameters
        result = model.transcribe(
            audio_file_path,
            language=st.session_state.source_language if st.session_state.source_language != "auto" else None,
            task="transcribe",
            temperature=0.0,  # More deterministic
            best_of=1,
            beam_size=1,
            word_timestamps=False,
            initial_prompt="This is a sales conversation or business meeting."  # Context hint
        )
        
        text = result["text"].strip()
        detected_language = result.get("language", "unknown")
        
        if st.session_state.debug_mode:
            st.write(f"🎤 Transcribed: '{text}' (language: {detected_language})")
            st.write(f"🔍 Confidence indicators: segments={len(result.get('segments', []))}")
        
        # Filter out very short or nonsensical results
        if len(text) < 3:
            if st.session_state.debug_mode:
                st.write("⚠️ Transcription too short, skipping")
            return "", detected_language
            
        return text, detected_language
        
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        if st.session_state.debug_mode:
            st.write(f"❌ Full error: {str(e)}")
        return f"Transcription error: {e}", "unknown"

def get_sales_suggestion(text):
    """Generate sales coaching suggestions (rule-based fallback)"""
    text_lower = text.lower()
    
    suggestions = []
    
    # Price/Budget related
    if any(word in text_lower for word in ['price', 'cost', 'expensive', 'budget', 'money']):
        suggestions.append("💡 Focus on ROI and value, not just price")
    
    # Objections
    if any(word in text_lower for word in ['no', 'not interested', "can't", "won't", 'problem']):
        suggestions.append("❗ Address objections with empathy - ask 'What would change your mind?'")
    
    # Decision making
    if any(word in text_lower for word in ['think', 'consider', 'maybe', 'decision']):
        suggestions.append("🎯 Ask clarifying questions to understand their decision process")
    
    # Competition
    if any(word in text_lower for word in ['competitor', 'other company', 'alternative']):
        suggestions.append("⭐ Highlight your unique differentiators")
    
    # Timeline/Urgency
    if any(word in text_lower for word in ['when', 'timeline', 'soon', 'quickly']):
        suggestions.append("⏰ Create urgency - what happens if they delay?")
    
    # Stakeholders
    if any(word in text_lower for word in ['team', 'boss', 'manager', 'others']):
        suggestions.append("👥 Identify all decision makers involved")
    
    # Questions
    if any(word in text_lower for word in ['how', 'what', 'why', 'question']):
        suggestions.append("💬 Great engagement! Give thorough, clear answers")
    
    # Positive signals
    if any(word in text_lower for word in ['interested', 'good', 'like', 'sounds']):
        suggestions.append("✅ Positive signal! Keep building momentum")
    
    if not suggestions:
        suggestions = [random.choice([
            "🎯 Ask open-ended questions to discover needs",
            "💪 Share relevant success stories",
            "🎧 Listen actively and summarize what you hear",
            "🔥 Focus on business impact and outcomes"
        ])]
    
    return suggestions

# Sidebar: AI Coach Settings
with st.sidebar:
    st.markdown("### 🤖 AI Coach Settings")
    
    # Debug mode toggle
    debug_mode = st.checkbox(
        "🔧 Debug Mode",
        value=st.session_state.debug_mode,
        help="Show detailed debugging information"
    )
    st.session_state.debug_mode = debug_mode
    
    # Audio device info
    if debug_mode:
        st.markdown("### 🎙️ Audio Devices")
        devices = check_audio_devices()
        if devices:
            for i, device in enumerate(devices):
                st.caption(f"{i}: {device['name']}")
        else:
            st.error("No input devices found")
    
    if LLM_AVAILABLE:
        # API Key input
        api_key_input = st.text_input(
            "🔑 OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key to enable GPT-4o suggestions",
            placeholder="sk-..."
        )
        
        if api_key_input != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key_input
            if api_key_input:
                openai.api_key = api_key_input
                LLM_ENABLED = True
                st.success("🤖 GPT-4o Enabled!")
                st.rerun()
            else:
                LLM_ENABLED = False
        
        # Show current status
        if LLM_ENABLED:
            st.success("✅ GPT-4o Ready")
            
            # Toggle for using LLM
            use_llm = st.checkbox(
                "🤖 Use AI Suggestions",
                value=st.session_state.use_llm,
                help="Enable GPT-4o powered suggestions (costs API credits)"
            )
            st.session_state.use_llm = use_llm
            
            if use_llm:
                st.success("🤖 AI Mode: ON")
                st.caption("💰 Using OpenAI API (costs apply)")
            else:
                st.info("🤖 AI Mode: OFF")
                st.caption("🆓 Using rule-based suggestions")
        else:
            st.warning("⚠️ Enter API key to enable AI")
            st.info("🆓 Using rule-based suggestions")
            
        # Instructions for getting API key
        with st.expander("❓ How to get OpenAI API Key"):
            st.markdown("""
            **Step-by-step:**
            1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
            2. Sign in or create account
            3. Click "Create new secret key"
            4. Copy the key (starts with 'sk-')
            5. Paste it in the field above
            """)
    else:
        st.error("📦 OpenAI not installed")
        st.code("pip install openai")

# Header
st.markdown("""
## 🎤 Real-Time GenAI TelePrompter
**AI Coach with Live Transcription and Translation✨**

### 🤖 AI Coach Configuration
""")

# Status display
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.running:
        st.markdown('<div class="recording-status">🔴 RECORDING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stopped-status">⏹️ STOPPED</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.running and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        st.metric("⏱️ Duration", f"{elapsed // 60}:{elapsed % 60:02d}")
    else:
        st.metric("⏱️ Duration", "00:00")

with col3:
    if LLM_ENABLED and st.session_state.use_llm:
        st.success("🤖 GPT-4o Active")
    elif LLM_AVAILABLE:
        st.info("🤖 Rule-based Mode")
    else:
        st.warning("🤖 AI Unavailable")

# Language settings
st.markdown("---")
lang_col1, spacer_col = st.columns([1, 2])

with lang_col1:
    source_lang = st.selectbox(
        "🎤 Source Language", 
        ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
        index=0,
        format_func=lambda x: {
            "auto": "🔍 Auto-detect",
            "en": "🇺🇸 English", "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
            "de": "🇩🇪 German", "it": "🇮🇹 Italian", "pt": "🇵🇹 Portuguese",
            "ru": "🇷🇺 Russian", "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
            "zh": "🇨🇳 Chinese", "ar": "🇸🇦 Arabic", "hi": "🇮🇳 Hindi"
        }.get(x, x)
    )
    st.session_state.source_language = source_lang

# Control buttons
st.markdown("---")
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("▶️ Start Recording", type="primary", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.last_record_time = 0
        st.session_state.transcript = []
        st.session_state.suggestions = []
        st.success("🎤 Recording started!")
        st.rerun()

with btn_col2:
    if st.button("⏹️ Stop", type="secondary", disabled=not st.session_state.running):
        st.session_state.running = False
        st.warning("⏹️ Recording stopped")
        st.rerun()

# Recording tips
st.markdown("---")
tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **🎧 Recording Notes**
    - Speak clearly and at normal pace
    - Minimize background noise
    - Stay close to microphone
    - Each recording cycle is 5 seconds
    - **Enable Debug Mode** in sidebar for troubleshooting
    """)

with tips_col2:
    st.markdown("""
    **💼 Sales Tips**
    - Understand client's needs first
    - Focus on benefits, not features
    - Build rapport and trust
    - Handle objections gracefully
    """)

# Real-time processing
if st.session_state.running:
    current_time = time.time()
    
    # Record every 5 seconds
    if current_time - st.session_state.last_record_time >= 5:
        st.session_state.last_record_time = current_time
        
        # Show recording indicator
        with st.spinner("🎤 Recording and processing..."):
            # Record audio
            audio = record_audio(duration=5)
            
            if audio is not None:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                
                if save_audio(audio, temp_filename):
                    # Transcribe
                    transcript, detected_lang = transcribe_audio(temp_filename)
                    
                    if transcript:
                        # Add to transcript (NO TRANSLATION)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        entry = {
                            "timestamp": timestamp,
                            "original": transcript,
                            "language": detected_lang
                        }
                        st.session_state.transcript.append(entry)
                        
                        # Generate suggestions using original transcript
                        suggestion_text = transcript
                        
                        # Build conversation history for AI context
                        conversation_history = ""
                        if st.session_state.use_llm and LLM_ENABLED and len(st.session_state.transcript) > 1:
                            recent_entries = st.session_state.transcript[-3:]
                            history_parts = []
                            for h_entry in recent_entries[:-1]:  # Exclude current entry
                                h_text = h_entry.get('translated', h_entry.get('original', ''))
                                if h_text:
                                    history_parts.append(h_text)
                            conversation_history = " ".join(history_parts)
                        
                        # Generate suggestions
                        if st.session_state.use_llm and LLM_ENABLED:
                            suggestions = get_llm_suggestion(suggestion_text, conversation_history)
                        else:
                            suggestions = get_sales_suggestion(suggestion_text)
                        
                        # Add suggestions to session state
                        for suggestion in suggestions:
                            sugg_entry = {
                                "timestamp": timestamp,
                                "text": suggestion,
                                "source": "AI" if (st.session_state.use_llm and LLM_ENABLED) else "Rule-based"
                            }
                            st.session_state.suggestions.append(sugg_entry)
                        
                        # Show success message
                        mode = "AI" if (st.session_state.use_llm and LLM_ENABLED) else "Rule-based"
                        st.success(f"✅ Transcribed ({mode}): {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                    else:
                        if st.session_state.debug_mode:
                            st.info("🔇 No speech detected in this audio chunk")
                
                # Clean up temp file
                try:
                    os.unlink(temp_filename)
                except:
                    pass
            else:
                if st.session_state.debug_mode:
                    st.error("❌ Failed to record audio")
        
        # Auto-refresh for continuous recording
        time.sleep(0.5)
        st.rerun()

# Display transcript and suggestions
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Live Transcript")
    
    if st.session_state.transcript:
        # Show last 5 entries
        recent_entries = st.session_state.transcript[-5:]
        
        for entry in recent_entries:
            st.markdown(f'<div class="transcript-entry">', unsafe_allow_html=True)
            st.markdown(f"**🕒 {entry['timestamp']}** ({entry['language']})")
            st.markdown(f"**Text:** {entry['original']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption(f"📊 Total entries: {len(st.session_state.transcript)}")
    else:
        st.info("💡 Start recording to see transcripts here...")

with col2:
    st.subheader("🤖 AI Sales Coach")
    
    if st.session_state.suggestions:
        # Show last 3 suggestions with source indicator
        recent_suggestions = st.session_state.suggestions[-3:]
        
        for sugg in recent_suggestions:
            st.markdown(f'<div class="suggestion-entry">', unsafe_allow_html=True)
            st.markdown(f"**🕒 {sugg['timestamp']}**")
            
            # Show source of suggestion
            source = sugg.get('source', 'Rule-based')
            if source == "AI":
                st.markdown(f"🤖 **AI Coach:** {sugg['text']}")
            else:
                st.markdown(f"📋 **Rule-based:** {sugg['text']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption(f"💡 Total suggestions: {len(st.session_state.suggestions)}")
    else:
        st.info("🎯 AI coaching suggestions will appear here...")

# Debug information
if st.session_state.debug_mode and st.session_state.running:
    st.markdown("---")
    st.subheader("🔧 Debug Information")
    
    debug_col1, debug_col2 = st.columns(2)
    
    with debug_col1:
        st.markdown('<div class="debug-info">', unsafe_allow_html=True)
        st.markdown("**System Status:**")
        st.markdown(f"- Whisper Model: {'✅ Loaded' if LOCAL_WHISPER_AVAILABLE else '❌ Not Available'}")
        st.markdown(f"- Translation: {'✅ Available' if TRANSLATION_AVAILABLE else '❌ Not Available'}")
        st.markdown(f"- OpenAI: {'✅ Connected' if LLM_ENABLED else '❌ Not Connected'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with debug_col2:
        st.markdown('<div class="debug-info">', unsafe_allow_html=True)
        st.markdown("**Recording Stats:**")
        st.markdown(f"- Audio Chunks: {len(st.session_state.transcript)}")
        st.markdown(f"- Suggestions: {len(st.session_state.suggestions)}")
        st.markdown(f"- Last Recording: {st.session_state.last_record_time}")
        st.markdown('</div>', unsafe_allow_html=True)

# Export functionality
if st.session_state.transcript and not st.session_state.running:
    st.markdown("---")
    st.subheader("📦 Export Session")
    
    export_data = {
        "session_info": {
            "start_time": st.session_state.start_time,
            "duration": int(time.time() - st.session_state.start_time) if st.session_state.start_time else 0,
            "total_transcripts": len(st.session_state.transcript),
            "total_suggestions": len(st.session_state.suggestions),
            "ai_mode_used": st.session_state.use_llm and LLM_ENABLED
        },
        "transcript": st.session_state.transcript,
        "suggestions": st.session_state.suggestions
    }
    
    export_json = json.dumps(export_data, indent=2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download JSON",
            export_json,
            file_name=f"teleprompter_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Create TXT format
        txt_lines = [
            f"TelePrompter Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"AI Mode: {'ON' if (st.session_state.use_llm and LLM_ENABLED) else 'OFF'}",
            "=" * 50,
            "",
            "TRANSCRIPT:",
            "-" * 20
        ]
        
        for entry in st.session_state.transcript:
            txt_lines.append(f"[{entry['timestamp']}] {entry.get('translated', entry['original'])}")
        
        txt_lines.extend([
            "",
            "SUGGESTIONS:",
            "-" * 20
        ])
        
        for sugg in st.session_state.suggestions:
            source = sugg.get('source', 'Rule-based')
            txt_lines.append(f"[{sugg['timestamp']}] ({source}) {sugg['text']}")
        
        txt_content = "\n".join(txt_lines)
        
        st.download_button(
            "📄 Download TXT",
            txt_content,
            file_name=f"teleprompter_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def check_audio_quality(audio_data):
    """Check audio quality and provide feedback"""
    max_amp = np.max(np.abs(audio_data))
    rms = np.sqrt(np.mean(audio_data**2))
    
    if st.session_state.debug_mode:
        st.write(f"🔊 Audio stats: max_amplitude={max_amp:.4f}, rms={rms:.4f}")
    
    if max_amp < 0.001:
        return False, "Audio too quiet - speak louder"
    elif max_amp > 0.95:
        return False, "Audio clipping detected - speak softer or move away from mic"
    elif rms < 0.001:
        return False, "Very low audio signal - check microphone"
    else:
        return True, "Audio quality good"