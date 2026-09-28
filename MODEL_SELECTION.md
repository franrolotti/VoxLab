# Model selection for VoxLab v0.1

VoxLab v0.1 ships with exactly **one** TTS backend. This document records how it
was chosen, what was measured, and what the licensing implications are.

> Research date: **September 2026**. The TTS landscape moves fast — re-check
> before adding backends.

## TL;DR

**Selected backend: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)**
running on ONNX Runtime through
[`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx), with the weights
pulled from
[`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).

| | |
|---|---|
| Weights licence | Apache-2.0 (commercial use allowed) |
| Download | ~355 MB (fp32 model 326 MB + 54 voices × 0.5 MB) |
| Full install (code + deps + model) | **~0.5 GB** (budget: 5 GB) |
| Needs PyTorch | No |
| GPU | Not required. RTF ≈ 0.35 on an Apple M4 CPU (≈3× faster than real time) |
| Languages | English (US/UK), Spanish, French, Italian, Portuguese (BR), Hindi, Japanese, Chinese |
| Spanish voices | 3 native (`ef_dora`, `em_alex`, `em_santa`) + blending |
| Voice cloning | **No** (see "Trade-off" below) |

The runner-up, **Chatterbox Multilingual** (MIT, zero-shot cloning), is the
planned second backend and the reason the voice layer already models
reference audio.

## Candidates

| Model | Params / download | Licence (weights) | Commercial | Spanish | Multi-speaker out of the box | Voice cloning | CPU viable | Install | Maintained |
|---|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** (hexgrad) | 82M / 0.33 GB | Apache-2.0 | ✅ | ✅ 3 voices | ✅ 54 voices | ❌ | ✅ fast | `pip`, no torch via ONNX | Stable (v1.0, 11M+ downloads) |
| **Chatterbox Multilingual** (Resemble AI) | 500M / 3.2 GB | MIT | ✅ (Perth watermark embedded) | ✅ 23 langs | ❌ 1 built-in voice | ✅ zero-shot | ⚠️ slow | `pip`, pins torch 2.6, transformers 5.2, gradio 6.8 | Active (v3 T3 checkpoint 2026) |
| **Qwen3-TTS 0.6B** (Alibaba) | 0.6B / 2.5 GB per variant | Apache-2.0 | ✅ | ✅ 10 langs | ✅ 9 voices (*CustomVoice*) | ✅ (*Base*, separate 2.5 GB model) | ⚠️ | `pip install qwen-tts`, pins transformers 4.57 | Active (Jan 2026) |
| **Fun-CosyVoice3 0.5B** (Alibaba) | 0.5B / several GB (repo 9.8 GB) | Apache-2.0 | ✅ | ✅ 9 langs | ❌ | ✅ zero-shot | ⚠️ | Repo clone + submodules, no pip package | Active |
| **F5-TTS** (SWivid) | 336M / 1.35 GB | **CC-BY-NC-4.0** | ❌ | ⚠️ community fine-tunes only | ❌ | ✅ | ⚠️ | `pip`, torch | Active |
| **XTTS-v2** (Coqui) | ~470M / 1.9 GB | **Coqui Public Model License** | ❌ (Coqui shut down, no relicensing) | ✅ 17 langs | ✅ | ✅ 6 s clip | ⚠️ | `coqui-tts` fork, torch | Community fork (idiap) |
| Piper / MeloTTS | 5–200 MB | MIT | ✅ | ✅ | per-model | ❌ | ✅ | easy | Active |
| Higgs Audio v3, VibeVoice, IndexTTS2, Orpheus | 1.5–8B | mixed | mixed | mixed | — | ✅ (most) | ❌ | heavy | — |

Large models (≥1.5B) were discarded early: they cannot fit a 5 GB budget once
PyTorch/CUDA wheels are counted, and need a GPU for usable speed. F5-TTS and
XTTS-v2 were discarded on licence grounds (non-commercial weights). Piper and
MeloTTS are light but clearly behind Kokoro in naturalness.

That left three serious candidates: **Kokoro**, **Chatterbox Multilingual** and
**Qwen3-TTS 0.6B**. Qwen3-TTS needs *two* 2.5 GB checkpoints to offer both
preset voices and cloning (the 0.6B *CustomVoice* model cannot clone and the
*Base* model has no preset voices), which alone uses the whole storage budget;
its preset speakers are native Chinese/English/Japanese/Korean speakers. It was
not benchmarked locally.

## Local measurements

Hardware: Apple M4, 16 GB RAM, macOS 15.6, CPU inference, Python 3.11/3.12.
Test set: six short Spanish lines from the VoxLab example dialogue plus one
English sentence. Intelligibility was scored by transcribing every clip with
Whisper *small* (a temporary tool, not a VoxLab dependency) and checking that
the transcript matches the input text.

| | Kokoro-82M (ONNX fp32) | Chatterbox Multilingual v2 | Chatterbox Multilingual v3 | Chatterbox v3 + Spanish reference clip |
|---|---|---|---|---|
| Install size (venv) | **126 MB** | 973 MB (+ `setuptools<81` fix) | same | same |
| Weights | **355 MB** | 3.2 GB | 3.2 GB | 3.2 GB |
| Real-time factor (CPU) | **0.35** | 5.7 | 5.3 | 5.9 |
| Real-time factor (MPS) | n/a | 10–17 | — | — |
| Peak RAM | < 1 GB | ~7 GB | ~7 GB | ~7 GB |
| Lines transcribed correctly (7) | **7/7** | 3/7 | 3/7 | 5/7 |
| Typical failure | — | trailing babble after short lines, truncated English | same, misread words | trailing hallucinated words |

Findings:

- Chatterbox produces a very natural timbre and can clone voices, but on
  **short dialogue lines** — exactly what VoxLab is built for — it often keeps
  generating after the sentence ends (hallucinated words, repeated phrases).
  This happened with both the built-in voice and a Spanish reference clip, and
  with both the v2 (the one the PyPI release loads) and v3 checkpoints.
- On CPU Chatterbox is ~5× *slower* than real time; a short dialogue takes
  minutes. Apple MPS was slower still in this test.
- The PyPI release (0.1.7) needs `setuptools<81` at runtime (its watermarking
  dependency imports `pkg_resources`) and pins `gradio`, `transformers` and
  `torch`, which makes it hard to co-install with other tools.
- Kokoro was flawless on the same lines, ~15× faster, ~8× smaller on disk and
  has no PyTorch dependency when run through ONNX Runtime.

## Decision

For VoxLab's weighting — **quality + size + ease of install + Spanish +
voice cloning + resource usage** — Kokoro wins five of the six criteria.
Voice cloning is the one it loses.

Reliability is part of quality: a dialogue tool that randomly appends
hallucinated words to a line is not usable without manual review of every
line. A clean, predictable voice is also the right input for VoxLab's
post-processing presets (`radio`, `arcade_80s`, …), which are where the
stylistic variety comes from.

### Trade-off: voice cloning

Kokoro cannot clone voices. VoxLab handles this explicitly instead of hiding
it:

- Backends declare their capabilities (`supports_cloning`, native
  parameters, languages, built-in voices). Voice profiles may reference an
  audio clip; if the active backend cannot clone, VoxLab says so clearly.
- Kokoro voices can be **blended** (`em_alex:0.6,am_michael:0.4`), which gives
  new timbres without cloning, and VoxLab adds a **pitch** control
  (speed-compensated resampling, artefact-free but formants move with pitch),
  so characters can be differentiated further.
- **Chatterbox Multilingual** is the planned optional backend for cloning
  (roadmap v0.2). Adding it only requires a new class in `voxlab/tts/` and a
  line in the backend registry; the dialogue, voice, audio and CLI layers do
  not change.

## Storage budget

| Item | Size |
|---|---|
| VoxLab code | < 1 MB |
| Python dependencies (onnxruntime, numpy, scipy, soundfile, espeak-ng loader, phonemizer, huggingface-hub, PyYAML) | ~180–250 MB |
| Kokoro fp32 model | 326 MB |
| Voice styles (54) | 28 MB (+ 28 MB packed copy for fast loading) |
| **Total** | **~0.6 GB** |

Models live in `~/.cache/voxlab` (configurable, never inside the repository).
`voxlab clean` removes them. A smaller quantised model (`q8f16`, 86 MB) can be
selected in `config.yaml` at a small quality cost.

## Licences

| Component | Licence | Notes |
|---|---|---|
| VoxLab (this repo) | MIT | |
| Kokoro-82M weights (hexgrad) | Apache-2.0 | Commercial use allowed. Keep the licence notice when redistributing weights. |
| ONNX export (onnx-community) | Apache-2.0 | Same weights, converted. |
| kokoro-onnx | MIT | Inference wrapper. |
| onnxruntime | MIT | |
| **phonemizer** | **GPL-3.0** | Text → phoneme conversion used by kokoro-onnx. |
| **espeak-ng** (via `espeakng-loader`) | **GPL-3.0** | Phonemisation backend, shipped as a shared library. |
| numpy, scipy, soundfile, huggingface-hub, PyYAML | BSD / Apache-2.0 / MIT | |

### Licence compatibility notes

- **Generated audio is not covered by the GPL.** The GPL applies to the
  phonemiser software, not to the audio it helps produce. Kokoro's Apache-2.0
  licence places no restrictions on outputs.
- VoxLab's own source is MIT and does not include or modify GPL code; it
  depends on `kokoro-onnx`, which imports `phonemizer` and loads `espeak-ng`
  at runtime. Installing these packages side by side with `pip` is fine.
- **If you redistribute VoxLab as a single bundled binary** (PyInstaller,
  Docker image, installer…), that bundle includes GPL-3.0 components and you
  must comply with the GPL-3.0 for the bundle (source availability, licence
  text). This is the only known incompatibility and it does not affect normal
  `pip` usage.
- Voices included with Kokoro are synthetic and Apache-2.0. For voice cloning
  with a future backend, only clone voices you have permission to use.

## Sources

- Kokoro-82M model card and voices: <https://huggingface.co/hexgrad/Kokoro-82M>,
  <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>
- Kokoro ONNX export: <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>
- kokoro-onnx: <https://github.com/thewh1teagle/kokoro-onnx>
- Chatterbox: <https://github.com/resemble-ai/chatterbox>,
  <https://huggingface.co/ResembleAI/chatterbox>,
  <https://www.resemble.ai/learn/models/chatterbox-multilingual>
- Qwen3-TTS: <https://github.com/QwenLM/Qwen3-TTS>,
  <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base>
- CosyVoice 3: <https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512>
- F5-TTS licence: <https://huggingface.co/SWivid/F5-TTS>
- XTTS-v2 licence: <https://huggingface.co/coqui/XTTS-v2>,
  <https://localaimaster.com/blog/xtts-coqui-commercial-license>
- Overviews: <https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models>,
  <https://localclaw.io/blog/local-tts-guide-2026>,
  <https://modal.com/blog/open-source-tts>
