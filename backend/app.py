"""
AI Voice Cloning Tool - English, Hindi, Bangla only
Free & Unlimited on Google Colab
"""

import sys
import os

CHATTERBOX_SRC = os.path.join(os.getcwd(), "chatterbox", "src")
if CHATTERBOX_SRC not in sys.path:
    sys.path.insert(0, CHATTERBOX_Src)

import re
import uuid
import numpy as np
import torch
import soundfile as sf
import gradio as gr

# Supported languages - ONLY English, Hindi, Bangla
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi", 
    "Bangla": "bn",
}

LANGUAGE_HINTS = {
    "en": "Hello! Welcome to the AI Voice Cloning Tool. This tool lets you clone any voice.",
    "hi": "नमस्ते! एआई वॉयस क्लोनिंग टूल में आपका स्वागत है। यह टूल आपको किसी भी आवाज़ को क्लोन करने देता है।",
    "bn": "হ্যালো! এআই ভয়েস ক্লোনিং টুলে আপনাকে স্বাগতম। এই টুলটি আপনাকে যেকোনো কণ্ঠস্বর ক্লোন করতে দেয়।",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        print("⏳ Loading ChatterboxMultilingualTTS model...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        print("✅ Model loaded successfully!")
    return MODEL

def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters"""
    text = re.sub(r"[*#\-–—]", " ", text)
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|"
        r"[\U0001F680-\U0001F6FF]|[\U00002702-\U000027B0]|"
        r"[\U0001F1E0-\U0001F1FF]",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

def split_text(text: str, max_chars: int = 250) -> list:
    """Split long text into smaller chunks"""
    words = text.split()
    chunks = []
    current = ""
    
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                chunks.append(current)
            current = word
    
    if current:
        chunks.append(current)
    
    return chunks

def clone_voice(
    text: str,
    audio_prompt_path: str,
    language_name: str,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    seed: int = 0,
) -> str:
    """Generate cloned voice from text and reference audio"""
    
    if not text.strip():
        raise gr.Error("❌ Please enter some text to synthesize.")
    
    if not audio_prompt_path:
        raise gr.Error("❌ Please upload a reference voice file (WAV or MP3).")
    
    lang_code = SUPPORTED_LANGUAGES.get(language_name, "en")
    text = clean_text(text)
    chunks = split_text(text)
    
    print(f"🎤 Generating {len(chunks)} chunk(s) in {language_name}")
    
    model = get_model()
    all_audio = []
    
    for i, chunk in enumerate(chunks):
        print(f"   Generating chunk {i+1}/{len(chunks)}...")
        
        wav = model.generate(
            chunk,
            language_id=lang_code,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        
        all_audio.append(wav.squeeze(0).cpu().numpy())
    
    combined = np.concatenate(all_audio)
    out_path = f"cloned_voice_{uuid.uuid4().hex[:8]}.wav"
    sf.write(out_path, combined, model.sr)
    
    print(f"✅ Saved to {out_path}")
    return out_path

# Custom CSS
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #e94560;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.app-header h1 {
    color: #e94560;
    font-size: 2rem;
    margin: 0;
}
.app-header p {
    color: #a0a8c0;
    margin: 8px 0 0;
}
.generate-btn {
    background: linear-gradient(90deg, #e94560, #c73652) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}
footer {
    visibility: hidden;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="AI Voice Cloning Tool") as demo:
    
    gr.HTML("""
    <div class="app-header">
        <h1>🎙️ AI Voice Cloning Tool</h1>
        <p>English · हिन्दी · বাংলা — Free & Unlimited on Google Colab</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            language = gr.Dropdown(
                choices=["English", "Hindi", "Bangla"],
                value="English",
                label="🌐 Select Language",
                info="Choose the language for the output speech"
            )
            
            text_input = gr.Textbox(
                label="✍️ Text to Synthesize",
                placeholder="Type or paste your text here...",
                lines=5,
                value="Hello! Welcome to the AI Voice Cloning Tool."
            )
            
            audio_upload = gr.Audio(
                label="🎤 Reference Voice (5-15 seconds recommended)",
                sources=["upload", "microphone"],
                type="filepath",
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                exaggeration = gr.Slider(
                    0.25, 2.0, value=0.5, step=0.05,
                    label="Expressiveness",
                    info="Higher = more emotional"
                )
                temperature = gr.Slider(
                    0.05, 2.0, value=0.8, step=0.05,
                    label="Temperature", 
                    info="Higher = more variation"
                )
                cfg_weight = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="CFG Weight",
                    info="0 = language transfer mode"
                )
                seed = gr.Number(
                    value=0, label="Seed",
                    info="0 = random, other = reproducible"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Cloned Voice",
                variant="primary",
                elem_classes="generate-btn"
            )
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="🔊 Generated Audio",
                type="filepath",
                show_download_button=True,
            )
            
            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background: #1a1a2e; border-radius: 10px;">
                <h4>📋 Instructions:</h4>
                <ol style="color: #a0a8c0;">
                    <li>Select your language (English/Hindi/Bangla)</li>
                    <li>Type or paste the text you want to be spoken</li>
                    <li>Upload a clear 5-15 second voice sample (WAV/MP3)</li>
                    <li>Click "Generate Cloned Voice"</li>
                    <li>Download the result!</li>
                </ol>
            </div>
            
            <div style="margin-top: 15px; padding: 15px; background: #0f1a2e; border-radius: 10px;">
                <p style="color: #607898; font-size: 0.8rem;">
                ⚠️ <strong>Disclaimer:</strong> For ethical use only. Do not impersonate without consent.
                </p>
            </div>
            """)
    
    def update_hint(lang):
        code = SUPPORTED_LANGUAGES[lang]
        return LANGUAGE_HINTS[code]
    
    language.change(fn=update_hint, inputs=language, outputs=text_input)
    generate_btn.click(
        fn=clone_voice,
        inputs=[text_input, audio_upload, language, exaggeration, temperature, cfg_weight, seed],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
