# VoxLab

**Local, free, open-source AI voice lab.** Write a dialogue in a text file,
give each character a voice, pick a sound style, get a WAV — all on your own
machine.

```bash
voxlab generate examples/dialogue.txt --preset arcade_80s
```

```text
[OPERATOR]
¿Me recibes?

[COMPUTER]
AFIRMATIVO.

[OPERATOR]
Inicia protocolo.

[COMPUTER]
PROTOCOLO INICIADO.
```

→ `output/dialogue.wav`: two characters, two voices, 1980s arcade speech chip.
Swap `--preset arcade_80s` for `--preset clean` and the same engine gives you
natural narration instead.

Built for game NPCs, audiobooks, podcasts, radio plays, demos, sci-fi and
cyberpunk ship computers, and anything else that needs voices. The retro
aesthetics are **audio presets**, not a limitation of the engine.

---

## Contents

1. [What VoxLab is](#what-voxlab-is)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Downloading the model](#downloading-the-model)
6. [First example](#first-example)
7. [Dialogue format](#dialogue-format)
8. [Voices](#voices)
9. [Voice cloning](#voice-cloning)
10. [Presets](#presets)
11. [Arcade 80s](#arcade-80s)
12. [Offline use and privacy](#offline-use-and-privacy)
13. [Hardware requirements](#hardware-requirements)
14. [Disk space](#disk-space)
15. [Limitations](#limitations)
16. [Model licences](#model-licences)
17. [Roadmap](#roadmap)

---

## What VoxLab is

A small command-line tool that turns multi-character scripts into finished
audio. It combines:

- **one** high-quality, lightweight neural TTS model
  ([Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)), chosen after
  comparing current open models — see [`MODEL_SELECTION.md`](MODEL_SELECTION.md);
- a **voice system** that maps characters to voices;
- an **audio processing** chain (EQ, compression, bit-crushing, reverb…)
  configured with simple YAML presets.

The TTS model produces a clean, natural voice; style comes afterwards from
audio processing. That keeps the speech intelligible and lets one engine cover
everything from audiobooks to 8-bit robots.

It is deliberately not a giant wrapper around every TTS model.

## Features

- 🗣️ Multi-character dialogues from plain text files
- 🎭 50+ built-in voices, voice **blending** and custom voice profiles
- 🌍 Spanish (Spain, Latin America, Río de la Plata), English (US/UK), French, Italian, Portuguese and Hindi
- 🎛️ Six presets: `clean`, `radio`, `arcade_80s`, `crt_terminal`, `cyberpunk`, `sci_fi` — and your own
- 🎚️ Per-line parameters: speed, pitch, volume, pause, language
- 💾 WAV export, 48 kHz / 24-bit by default (44.1 kHz and 16-bit available)
- 🔒 100 % local inference once the model is downloaded; no cloud APIs
- ⚡ ~4× faster than real time on a laptop CPU, no GPU needed
- 📦 ~0.6 GB total footprint, no PyTorch
- 🧩 Pluggable TTS backends (voice cloning ready at the interface level)

## Architecture

```text
TXT dialogue ─▶ Dialogue parser ─▶ Voice assignment ─▶ TTS backend ─▶ Per-line FX ─▶ Mixer ─▶ Preset ─▶ WAV
               voxlab.dialogue    voxlab.voices       voxlab.tts      (pitch, volume)  voxlab.audio
```

```text
src/voxlab/
├── cli.py            # argument parsing only
├── config.py         # config.yaml loading and validation
├── pipeline.py       # orchestrates the steps above (plan → render → export)
├── benchmark.py      # `voxlab benchmark`
├── storage.py        # model cache size and `voxlab clean`
├── dialogue/         # parser + data models
├── voices/           # voice profiles, casting, built-in voices
├── tts/
│   ├── base.py       # TTSBackend interface + capabilities
│   ├── factory.py    # backend registry (`tts.backend: auto`)
│   └── kokoro.py     # Kokoro-82M on ONNX Runtime
└── audio/
    ├── effects.py    # effect functions (filters, compressor, bitcrush, reverb…)
    ├── presets.py    # YAML presets → effect chains
    ├── mixer.py      # joins lines with gaps
    └── export.py     # resampling, dithering, WAV writing
```

The pipeline talks only to the abstract `TTSBackend`. Each backend declares its
**capabilities** — languages, voice cloning, parameters it handles natively —
and VoxLab adapts: unsupported parameters are emulated in post-processing
(`pitch`, `volume`, `pause`) or ignored with a warning; cloning voices are
rejected clearly on backends that cannot clone.

**Adding a backend** (e.g. Chatterbox, F5-TTS, XTTS, CosyVoice) means writing
one `TTSBackend` subclass and one registry entry in `tts/factory.py`. Dialogue,
voices, audio and CLI code stay untouched.

## Installation

Requires **Python 3.10–3.13** (3.12 recommended) on Linux, macOS or Windows.
No GPU, PyTorch or system packages needed — the phonemiser (espeak-ng) ships as
a Python wheel.

```bash
git clone https://github.com/franrolotti/VoxLab.git
cd VoxLab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.12 && uv pip install -e .
```

For development: `pip install -e ".[dev]"`, then `pytest` and `ruff check .`.

## Downloading the model

```bash
voxlab models --download
```

This fetches Kokoro-82M (ONNX, ~355 MB) from Hugging Face into
`~/.cache/voxlab` — never into the repository. If you skip this step, the
model is downloaded automatically the first time you generate audio
(disable with `storage.auto_download: false`).

```bash
voxlab models   # backend, model, licence, size, status and disk usage
```

A smaller model is available if disk space matters more than speed (on CPU the
quantised variants are smaller but *slower* than fp32):

```yaml
# config.yaml
tts:
  options:
    variant: q8f16   # fp32 (default, 326 MB) | fp16 | q8 | q8f16 (86 MB)
```

## First example

```bash
voxlab generate examples/dialogue.txt --preset arcade_80s
voxlab generate examples/dialogue.txt --preset clean
voxlab generate examples/dialogue.txt --preset cyberpunk --output output/demo.wav
voxlab generate examples/dialogue.txt --voice OPERATOR=deep --sample-rate 44100 --bit-depth 16
voxlab generate examples/dialogue.txt --dry-run      # show who says what, no synthesis
```

Other commands:

```bash
voxlab voices              # voice profiles
voxlab voices --speakers   # the model's built-in speakers
voxlab presets             # audio presets
voxlab models              # backends and model status
voxlab benchmark           # speed and memory on this machine
voxlab clean               # delete downloaded models
voxlab --help
voxlab generate --help
```

## Dialogue format

```text
# Lines starting with '#' are comments.

[OPERATOR]
¿Me recibes?

[COMPUTER|speed=0.9|pitch=-2]
AFIRMATIVO.
Iniciando secuencia.        <- consecutive lines join into one utterance

[NARRATOR|lang=en|pause=1200]
Meanwhile, on the bridge...
```

- `[SPEAKER]` starts a block; every non-empty line until the next header is
  what that speaker says.
- Optional parameters follow `|` as `key=value`:

| Parameter | Range | Implemented by | Meaning |
|---|---|---|---|
| `speed` | 0.5–2.0 | model (Kokoro) | speaking rate |
| `pitch` | -12–12 | VoxLab | semitones up/down |
| `volume` | -30–12 | VoxLab | gain in dB |
| `pause` | 0–10000 | VoxLab | silence after the line, in ms (default `audio.gap_ms`) |
| `lang` | e.g. `es`, `es-ar`, `en`, `en-gb` | model | language/dialect of the line |

### Spanish variants

| Code | Variant | Pronunciation |
|---|---|---|
| `es` / `es-es` | Spain | *distinción* (`cielo` → /θ/), lateral `ll` |
| `es-419` (`es-mx`) | Latin America | *seseo* (`cielo` → /s/), *yeísmo* |
| `es-ar` (`es-uy`) | Río de la Plata | *seseo* + *sheísmo* (`calle`, `yo` → /ʃ/) |

Set it per line (`[CRONISTA|lang=es-ar]`), per voice (`language: es-ar`), per
run (`--language es-ar`) or globally (`tts.language: es-ar`). The variants
change **pronunciation**; intonation still comes from the model's Spanish
voices, so the rioplatense melody is only approximated. Write *voseo*
directly in the text (`vos querés`).

Parameters a backend does not support are ignored with a warning. A text file
**without any header** is read as narration, one utterance per paragraph.

Errors point at the offending line (`line 12 [COMPUTER]: parameter 'speed'
must be between 0.5 and 2`), and everything is validated before the model is
loaded.

## Voices

A **voice** is a profile describing how a character sounds. Speakers map to
voices in this order:

1. `--voice SPEAKER=voice` on the command line
2. the `cast:` section of `config.yaml`
3. a voice with the same name as the speaker (`OPERATOR` → `operator`)
4. `voice.default` (with a warning), or an error with `--strict` / `voice.strict: true`

Built-in voices: `default`, `narrator`, `operator`, `computer`, `female`,
`male`, `deep`. Each one picks a suitable speaker for Spanish and English.

Create your own in `voices/<name>/voice.yaml` (user voices override built-ins
with the same name):

```yaml
# voices/captain/voice.yaml
description: Grizzled starship captain
speaker:
  kokoro: {es: em_santa, en: am_onyx}   # per backend, per language
# speaker: em_alex:0.6,em_santa:0.4    # or a blend of built-in speakers
language: es
params: {speed: 0.95, pitch: -1}
preset: radio                          # optional: always process this voice
```

Or from the command line:

```bash
voxlab voices add captain --speaker "em_alex:0.6,em_santa:0.4" --language es
voxlab voices --speakers               # list speaker ids to use/blend
```

The `pitch` control resamples the audio (and asks the model for the inverse
tempo change so speed is preserved). It is artefact-free but moves formants
with the pitch, so large shifts sound less natural — ideal for robots and
computers, use ±1–3 semitones for human characters.

## Voice cloning

The voice system supports **reference audio**:

```bash
voxlab voices add captain --reference my_recording.wav
```

This copies the clip to `voices/captain/` (git-ignored). Backends advertise
whether they can clone; **Kokoro, the v0.1 backend, cannot**. With Kokoro, a
voice that only has a reference clip fails with a clear message, and a voice
with both a reference and a `speaker` uses the speaker (with a warning).

Why ship without cloning? The cloning models tested for v0.1 hallucinated extra
words on short dialogue lines and were ~20× slower on CPU; details in
[`MODEL_SELECTION.md`](MODEL_SELECTION.md). An optional cloning backend is the
first item on the roadmap. Only clone voices you have permission to use.

## Presets

Presets are YAML effect chains in [`presets/`](presets/). They run on the
final mix at the output sample rate.

| Preset | Sound |
|---|---|
| `clean` | Natural voice. Rumble filter, gentle leveling, normalisation. Nothing else. |
| `radio` | Two-way radio: 300–3400 Hz band, presence boost, compression, light saturation and hiss. |
| `arcade_80s` | 8-bit / 11 kHz speech chip through a small cabinet speaker. |
| `crt_terminal` | Terminal speaker, partial bit reduction, faint 15.7 kHz CRT whine. |
| `cyberpunk` | Grit, chorus, heavy compression, slap delay and a neon-city space. |
| `sci_fi` | Ring-modulated starship computer with doubling and a large hall. |

All six were checked with automatic speech recognition: every line of the
example dialogue stays intelligible (priority: **intelligible > aesthetic >
extreme effects**).

Make your own by copying one into `presets/` (or any folder listed in
`preset.dirs`) and editing it; a user preset with a built-in name overrides
it. You can also pass a path: `--preset my/preset.yaml`.

```yaml
name: walkie_talkie
description: Cheap handheld radio
chain:
  - effect: bandpass
    low_hz: 500
    high_hz: 3000
  - effect: saturation
    drive: 3
    mix: 0.6
  - effect: noise
    level_db: -40
    color: white
  - effect: normalize
    peak_db: -1
```

Available effects: `gain`, `normalize`, `compressor`, `highpass`, `lowpass`,
`bandpass`, `peaking_eq`, `saturation`, `bitcrush`, `ring_mod`, `noise`,
`tone`, `chorus`, `delay`, `reverb`. Parameters are validated when the preset
is loaded; add `enabled: false` to a step to bypass it. See
[`src/voxlab/audio/effects.py`](src/voxlab/audio/effects.py) for parameters
and defaults.

A voice profile can also have its own `preset:` — e.g. only the `COMPUTER`
sounds like a CRT while the operator stays clean.

## Arcade 80s

Sampled speech in 1980s arcade cabinets and home computers came out of 8-bit
DACs at low sample rates through small, band-limited speakers. `arcade_80s`
recreates that chain:

1. **High-pass 180 Hz** — tiny speakers have no low end.
2. **+4 dB presence at 2.5 kHz** before degrading, so consonants survive.
3. **Compression** — keeps quiet syllables above the quantisation noise.
4. **Bitcrush: 8 bits, 11 025 Hz sample-and-hold** — the core of the sound:
   quantisation grain plus the metallic aliasing of low-rate playback.
5. **Low-pass 6.5 kHz** — the cabinet speaker.
6. **Light saturation** — an overdriven amplifier.
7. **Very small reverb (8 % wet)** — the cabinet box.
8. **Normalise to -1 dBFS.**

It sounds great on the `computer` voice (lowered, slightly slower female
voice). For a harsher chip, lower `bits` to 6 or `downsample_hz` to 8000 in a
copy of the preset.

## Offline use and privacy

- After the model download, generation runs **entirely on your machine**.
  Text, audio, voices and reference clips are never sent anywhere.
- VoxLab disables Hugging Face telemetry and, once the model is cached, sets
  the Hub to offline mode for the rest of the process
  (`privacy.offline: true`).
- The model is pinned to a specific revision; VoxLab checks the local cache
  first and does not contact the network when the files are present.
- The only network access is installing Python packages and the one-time
  model download (`voxlab models --download`). To prepare an air-gapped
  machine, copy `~/.cache/voxlab` from a machine that has downloaded it.

## Hardware requirements

| | Minimum | Measured (Apple M4) |
|---|---|---|
| CPU | any 64-bit x86-64 / ARM64 CPU | RTF 0.27 (3.8× real time) |
| RAM | 2 GB free | ~0.9 GB peak |
| GPU | not needed | not used |

An NVIDIA GPU can be used with `pip install onnxruntime-gpu` and
`tts.device: cuda`, but Kokoro is fast enough on CPU that it is rarely worth
it. See [`benchmarks/`](benchmarks/README.md) and run `voxlab benchmark` on
your machine.

## Disk space

| Item | Size |
|---|---|
| VoxLab + Python dependencies (onnxruntime, numpy, scipy, …) | ~230 MB |
| Kokoro-82M fp32 model + 54 voices (`~/.cache/voxlab`) | ~365 MB |
| **Total** | **~0.6 GB** (budget: 5 GB) |

With `variant: q8f16` the model shrinks to ~115 MB (slower on CPU). Output WAVs are ~8.6 MB
per minute at 48 kHz / 24-bit mono.

```bash
voxlab models            # shows the model directory size
voxlab clean --dry-run   # what would be deleted
voxlab clean             # delete ~/.cache/voxlab (asks first; -y to skip)
```

`voxlab clean` only touches `storage.model_dir`; it refuses to delete your home
folder, the current directory or anything that looks like a project.

## Limitations

- **No voice cloning in v0.1** (Kokoro cannot clone). See above.
- **Spanish voices:** three native speakers (`ef_dora`, `em_alex`,
  `em_santa`); blending and pitch give more variety. English has ~20 voices.
- **Languages:** es (es-es, es-419, es-ar), en (US/UK), fr, it, pt-BR, hi. Japanese and Chinese
  voices exist in the model but need a different phonemiser; not supported yet.
- **No emotion/style control**: Kokoro reads text neutrally. Punctuation
  (`!`, `?`, `…`) still shapes intonation.
- **Pitch** shifts formants too (see [Voices](#voices)).
- Lines are **sequential**; no overlapping speech. Output is **mono WAV**.
- Tested on macOS (Apple Silicon). Linux and Windows are supported by all
  dependencies but not yet covered by CI.

## Model licences

| Component | Licence | Commercial use |
|---|---|---|
| VoxLab code | MIT | ✅ |
| Kokoro-82M weights (and ONNX export) | Apache-2.0 | ✅ |
| kokoro-onnx, onnxruntime | MIT | ✅ |
| phonemizer, espeak-ng (phonemisation) | **GPL-3.0** | ✅ for use; see note |

Audio you generate is yours to use, including commercially. The GPL-3.0
components only matter if you **redistribute VoxLab bundled** into a single
binary or image: that bundle must then comply with the GPL-3.0. Normal
`pip install` use is unaffected. Full analysis, and the licences of the models
that were *not* chosen (several are non-commercial), in
[`MODEL_SELECTION.md`](MODEL_SELECTION.md#licences).

## Roadmap

**v0.1 (this release)** — text → dialogue → TTS → voices → audio FX → WAV.

**v0.2**
- Optional voice-cloning backend (Chatterbox Multilingual), installed as an
  extra: `pip install "voxlab[chatterbox]"`
- Batch generation (folders of dialogues, one WAV per line for game engines)
- Loudness normalisation (LUFS) and stereo placement per character

**v0.3+**
- More TTS backends through the same interface
- Advanced voice cloning workflows
- Timeline editor, overlapping lines and sound cues
- GUI
- Video, lip-sync data and diarisation

Out of scope: training or fine-tuning models, music or sound-effect
generation, cloud services.

## License

VoxLab is released under the [MIT License](LICENSE).
