"""
Patched chatterbox __init__.py — adds Bangla (bn) to SUPPORTED_LANGUAGES
"""

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    
    if "bn" not in SUPPORTED_LANGUAGES:
        SUPPORTED_LANGUAGES["bn"] = "Bangla"
        print("✅ Bangla (bn) injected into SUPPORTED_LANGUAGES")
    
    __all__ = [
        "ChatterboxTTS",
        "ChatterboxVC", 
        "ChatterboxMultilingualTTS",
        "SUPPORTED_LANGUAGES",
    ]
except ImportError:
    __all__ = ["ChatterboxTTS", "ChatterboxVC"]
