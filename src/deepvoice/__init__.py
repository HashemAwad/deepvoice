"""DeepVoice: audited fake-voice detection study (CMPS 261)."""

import os

# Let PyTorch run ops the Apple GPU (mps) lacks on the CPU; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
