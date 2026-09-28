# Benchmarks

`voxlab benchmark` measures the model that is already installed. It never
downloads additional models.

```bash
voxlab benchmark                     # Spanish + English sentences, 1 run each
voxlab benchmark --runs 3            # repeat each sentence
voxlab benchmark --language es       # only Spanish
voxlab benchmark --json benchmarks/results/my-machine.json
```

It reports:

| Metric | Meaning |
|---|---|
| Load | Time to load the model into memory |
| gen (s) | Wall-clock synthesis time per sentence (after a warm-up call) |
| audio (s) | Duration of the generated audio |
| RTF | Real-time factor = gen / audio. Below 1.0 is faster than real time |
| Peak memory | Peak resident memory of the process (not available on Windows) |
| System / Runtime | CPU, RAM, OS, Python, backend, model variant, ONNX providers |

`benchmarks/results/` is git-ignored; attach JSON reports to issues when
reporting performance problems.

## Reference results

Kokoro-82M, fp32 ONNX, CPU only (`CPUExecutionProvider`), 3 runs per sentence.

| Machine | Load | RTF (es) | RTF (en) | Overall | Peak memory |
|---|---|---|---|---|---|
| Apple M4, 10 threads, 16 GB, macOS 15.6, Python 3.12 | 0.3 s | 0.22–0.28 | 0.25–0.36 | **0.265** (3.8× real time) | 881 MB |

The first English sentence after Spanish ones is slower (≈0.8 RTF) because the
phonemiser switches language; subsequent sentences are not affected.

### Voice cloning (Qwen3-TTS 0.6B, MLX GPU)

```bash
voxlab benchmark --backend qwen --voice fran --language es
```

| Machine | Load | RTF (es) | Peak memory |
|---|---|---|---|
| Apple M4, 16 GB, macOS 15.6 | 2.1 s | 2.3–2.5 short lines, 1.2 long lines — **1.65** overall | 2.7 GB |

Measured with a 25 s reference clip; shorter references are faster.

For context, the same machine ran Chatterbox Multilingual (evaluated during
model selection) at an RTF of 5–6 on CPU, i.e. about 20× slower. See
[`MODEL_SELECTION.md`](../MODEL_SELECTION.md).
