"""Emotional Basque voices: HiTZ/Aholab StyleTTS2 (Antton, Maider; neutral/happy/sad).

Model: https://huggingface.co/HiTZ/StyleTTS2-eu_emo (Apache-2.0). Model code:
https://github.com/AArriandiaga/StyleTTS2_basque (MIT), fetched at a pinned
commit and checksum-verified; only the inference modules are imported.

* The acoustic model runs natively with PyTorch (Apple GPU through MPS, or CPU).
* The Basque phonemiser (Aholab ``modulo1y2``) is a Linux x86-64 binary, run
  in the same throwaway container as the aHoTTS backend.
* The 2.1 GB training checkpoint is downloaded once, reduced to the inference
  weights in fp16 (~0.4 GB) and deleted.

Install with ``pip install "voxlab[emotion]"``; needs Docker running.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import math
import re
import string
import subprocess
import sys
import types
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from voxlab.errors import BackendError
from voxlab.tts.ahotts import IMAGE, PLATFORM, fetch
from voxlab.tts.base import (
    BackendCapabilities,
    ModelInfo,
    SpeakerInfo,
    SynthesisRequest,
    SynthesisResult,
    TTSBackend,
)

log = logging.getLogger(__name__)

MODEL_REPO = "HiTZ/StyleTTS2-eu_emo"
MODEL_REVISION = "11ef181447a413f3624ce84147c6b85f803f6655"
CHECKPOINT = (
    "epoch_2nd_00030.pth",
    "a97881444c2d1503318fb5e2d46ab1102d51b8e85a1dda83d94f8c18ea2ae4a7",
)
CHECKPOINT_MB = 2140
CONFIG = (
    "config_basque_multispeaker_phoneme_wavlm_emo_plbertemo_no_acc.yml",
    "921189f3245affe3a0eaf3277fd707fe42e25c4ebbc73c258a44ad14072f4dad",
)
PHONEMIZER_FILES: dict[str, str] = {
    "phonemizer/modulo1y2/modulo1y2": (
        "c122bd6197e5e360d534957322f8d98a06cb3bcb4d412ee9978e891ae1b43e8a"
    ),
    "phonemizer/dict/eu_dicc.dic": (
        "4a4c6553965ac7c7937b599d3e8a3d8d94df48a0bdef943a84c63f4b261172f8"
    ),
}
REFERENCES: dict[str, str] = {
    "Demo/ref_antton_neu.wav": "d8b9f0bfa5b18f13a8bfe01516b632c385c0e7d9ad6b67a1e5bac64b51672877",
    "Demo/ref_antton_poz.wav": "16ec10f6b96b9fb05ec2b2ef2fce83af2c2f6d1792004500e20992503d7de1e5",
    "Demo/ref_antton_tri.wav": "645eacbd0dfa22c30a13aee1c1fe25ccbbdec2a8147bae5fed213e54c2a8fd9a",
    "Demo/ref_maider_neu.wav": "1a73ce618192e1d115bd3f8d42425fa2c5a923ac8462f2f0baeae8c7208db49d",
    "Demo/ref_maider_poz.wav": "96cd261f6e82e3ca927be491bfeba378f06310adbcedfc54baaa1810fa3beec0",
    "Demo/ref_maider_tri.wav": "c4e941ac642f8d98cacd86cf47f411765c08dc00ba27e1ccbf2e807c868e2513",
}

CODE_REPO = "AArriandiaga/StyleTTS2_basque"
CODE_REVISION = "a1ac47f09ea2953dee2b3fe589427ddaaa604cf5"
CODE_FILES: dict[str, str] = {
    "models.py": "c17b830bcd7539ffc9ae6cec0d8f17f79df573a8677d1d214c65355e5191a301",
    "Modules/__init__.py": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "Modules/hifigan.py": "1ce902afae4882c9dfb9f8a85141eeb4ae3e6de7002ae9212efb484a68d00ad7",
    "Modules/utils.py": "2204d95f1d6f734437d4dfbdfa7e994ed973a6e9dafd79a05a38713b6a8b3322",
    "Modules/discriminators.py": "2eb7f2f35b63820f19e0e4ab9941afe8a3e8f7fff15098f30a07992fac5dc558",
    "Modules/diffusion/__init__.py": (
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    ),
    "Modules/diffusion/diffusion.py": (
        "16a04f4b45dcc4949a1829764051741d62dc97e61b34da2f6b0babe0cf39cc95"
    ),
    "Modules/diffusion/modules.py": (
        "59355c91d0060fb2083d958ba8e6e64debb92205fad132e1b9f946bf4eb9989f"
    ),
    "Modules/diffusion/sampler.py": (
        "933f78cba5963847b44987afaf84b2e6d8fdc1e2b13932eb41bd5426ef5a9032"
    ),
    "Modules/diffusion/utils.py": (
        "326b27b3eda8d3e170b2aee9510331262c767b9198070df916184b7181e2416f"
    ),
    "Utils/PLBERT_phoneme/config.yml": (
        "68da45fada88a69ccecbc86f4acffdd2c1055aa89cd6dd36d2185e646e3f40df"
    ),
    "LICENSE": "7f023c12a1d6b25583506f6fc77f5287c7c71b65c0f886922f14153bcc07dfa9",
}
# Checkpoint parts used for inference (the rest are discriminators and optimiser state).
INFERENCE_KEYS = (
    "bert",
    "bert_encoder",
    "predictor",
    "decoder",
    "text_encoder",
    "predictor_encoder",
    "style_encoder",
    "diffusion",
)
WEIGHTS_MB = 400

SPEAKERS = {"antton": "male", "maider": "female"}
DEFAULT_SPEAKER = "maider"
DEFAULT_INTENSITY = 0.8
EMOTIONS: dict[str, str] = {
    "neu": "neu",
    "neutral": "neu",
    "neutroa": "neu",
    "poz": "poz",
    "poza": "poz",
    "pozik": "poz",
    "happy": "poz",
    "joy": "poz",
    "alegre": "poz",
    "tri": "tri",
    "tristura": "tri",
    "triste": "tri",
    "sad": "tri",
}
SAMPLE_RATE = 24000
REQUIRED_MODULES = ("torch", "transformers", "einops", "einops_exts", "munch", "yaml")

# Aholab SAMPA -> IPA, and multi-character IPA -> the single symbols the model was trained on
# (from phonemizer/eu_phonemizer.py in the model repository).
_SAMPA_TO_IPA = {
    "g": "ɡ",
    "tS": "tʃ",
    "ts`": "tʂ",
    "gj": "ɟ",
    "jj": "ʝ",
    "B": "β",
    "T": "θ",
    "D": "ð",
    "s`": "ʂ",
    "S": "ʃ",
    "G": "ɣ",
    "J": "ɲ",
    "L": "ʎ",
    "r": "ɾ",
    "rr": "r",
    "Z": "ʒ",
    "ph": "pʰ",
    "kh": "kʰ",
    "th": "tʰ",
}
_SINGLE_CHAR = {
    "tʃ": "C",
    "ts": "V",
    "tʂ": "P",
    "'i": "I",
    "'e": "E",
    "'a": "A",
    "'o": "O",
    "'u": "U",
    "pʰ": "H",
    "kʰ": "K",
    "tʰ": "T",
}
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"<>“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"  # noqa: E501
SYMBOLS = {s: i for i, s in enumerate([_PAD, *_PUNCTUATION, *_LETTERS, *_LETTERS_IPA])}
_WORDS = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class StyleTTS2Backend(TTSBackend):
    name = "styletts2"
    capabilities = BackendCapabilities(
        voice_cloning=False,
        languages=("eu",),
        native_params=frozenset({"speed", "emotion", "intensity", "seed"}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed = int(self.options.get("seed", 0))
        self.diffusion_steps = int(self.options.get("diffusion_steps", 5))
        # Timbre: share taken from the diffusion sample instead of the reference clip.
        self.alpha = float(self.options.get("alpha", 0.3))
        self.embedding_scale = float(self.options.get("embedding_scale", 1.0))
        self._model: Any = None
        self._sampler: Any = None
        self._device = "cpu"
        self._styles: dict[str, Any] = {}

    @classmethod
    def is_installed(cls) -> bool:
        import shutil

        return shutil.which("docker") is not None and all(
            importlib.util.find_spec(m) for m in REQUIRED_MODULES
        )

    # --- files --------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.model_dir / "styletts2"

    @property
    def code_dir(self) -> Path:
        return self.root / f"code-{CODE_REVISION[:12]}"

    @property
    def weights_file(self) -> Path:
        return self.root / f"weights-{MODEL_REVISION[:12]}-fp16.pth"

    def _model_file(self, path: str) -> Path:
        return self.root / "model" / path

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            backend=self.name,
            model_id=MODEL_REPO,
            variant="emo-fp16",
            license="Apache-2.0",
            download_mb=CHECKPOINT_MB,
            url=f"https://huggingface.co/{MODEL_REPO}",
        )

    def _small_files(self) -> dict[Path, tuple[str, str]]:
        """Local path -> (URL, sha256) for everything except the checkpoint."""
        model = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"
        code = f"https://raw.githubusercontent.com/{CODE_REPO}/{CODE_REVISION}"
        files = {self._model_file(CONFIG[0]): (f"{model}/{CONFIG[0]}", CONFIG[1])}
        for path, sha in {**PHONEMIZER_FILES, **REFERENCES}.items():
            files[self._model_file(path)] = (f"{model}/{path}", sha)
        for path, sha in CODE_FILES.items():
            files[self.code_dir / path] = (f"{code}/{path}", sha)
        return files

    def is_downloaded(self) -> bool:
        return self.weights_file.is_file() and all(p.is_file() for p in self._small_files())

    def download(self) -> None:
        for path, (url, sha) in self._small_files().items():
            if not path.is_file():
                fetch(url, path, sha)
        self._model_file(next(iter(PHONEMIZER_FILES))).chmod(0o755)
        if not self.weights_file.is_file():
            self._download_weights()

    def _download_weights(self) -> None:
        import torch

        name, sha = CHECKPOINT
        full = self.root / name
        log.info(
            "Downloading %s/%s (~%d MB, deleted after extracting ~%d MB of inference weights)",
            MODEL_REPO,
            name,
            CHECKPOINT_MB,
            WEIGHTS_MB,
        )
        if not full.is_file():
            fetch(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}", full, sha)
        try:
            # Checksum-pinned file; it also holds optimiser state, so full unpickling is needed.
            params = torch.load(full, map_location="cpu", weights_only=False)
            params = params.get("net", params)
            missing = [k for k in INFERENCE_KEYS if k not in params]
            if missing:
                raise BackendError(f"Checkpoint lacks {', '.join(missing)}")
            slim = {
                key: OrderedDict(
                    (k.removeprefix("module."), v.half() if v.is_floating_point() else v)
                    for k, v in params[key].items()
                )
                for key in INFERENCE_KEYS
            }
            tmp = self.weights_file.with_suffix(".part")
            torch.save(slim, tmp)
            tmp.replace(self.weights_file)
        finally:
            full.unlink(missing_ok=True)

    # --- inference ----------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_installed():
            raise BackendError(
                'Emotional Basque voices need Docker and the extra: pip install "voxlab[emotion]"'
            )
        if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
            raise BackendError("Docker is installed but not running: start Docker Desktop")
        if not self.is_downloaded():
            if not self.allow_download:
                raise BackendError(
                    "StyleTTS2 model not found and auto-download is disabled. "
                    "Run `voxlab models --download --backend styletts2` first."
                )
            self.download()
        if subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
        ).returncode:
            subprocess.run(["docker", "pull", "--platform", PLATFORM, IMAGE], check=True)

        import torch

        self._device = self._select_device(torch)
        log.info("Loading StyleTTS2 (Basque, emotional) on %s", self._device)
        with _quiet():
            self._model, self._sampler = self._build(torch)

    def _select_device(self, torch: Any) -> str:
        device = self.device.lower()
        if device == "auto":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps", "cuda"):
            raise BackendError(f"Unsupported device {self.device!r} for StyleTTS2")
        return device

    def _build(self, torch: Any) -> tuple[Any, Any]:
        import yaml

        code = str(self.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        # models.py imports the training-only ASR/F0 networks (and torchaudio) at module
        # level; inference never uses them, so stand-ins keep those imports cheap.
        for module, attr in (("Utils.ASR_basque.models", "ASRCNN"), ("Utils.JDC.model", "JDCNet")):
            stub = types.ModuleType(module)
            setattr(stub, attr, object)
            sys.modules.setdefault(module, stub)
        for package in ("Utils", "Utils.ASR_basque", "Utils.JDC"):
            sys.modules.setdefault(package, types.ModuleType(package))

        import models as st  # the pinned StyleTTS2_basque models.py
        from Modules.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
            KDiffusion,
            LogNormalDistribution,
        )
        from Modules.hifigan import Decoder
        from munch import munchify
        from transformers import AlbertConfig, AlbertModel

        cfg = yaml.safe_load(self._model_file(CONFIG[0]).read_text(encoding="utf-8"))
        args = munchify(cfg["model_params"])
        plbert = yaml.safe_load(
            (self.code_dir / "Utils/PLBERT_phoneme/config.yml").read_text(encoding="utf-8")
        )

        class Albert(AlbertModel):
            def forward(self, *a: Any, **kw: Any) -> Any:
                return super().forward(*a, **kw).last_hidden_state

        bert = Albert(AlbertConfig(**plbert["model_params"]))
        dec = args.decoder
        transformer = st.StyleTransformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            context_features=args.style_dim * 2,
            **args.diffusion.transformer,
        )
        diffusion = st.AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=bert.config.max_position_embeddings,
            embedding_features=bert.config.hidden_size,
            embedding_mask_proba=args.diffusion.embedding_mask_proba,
            channels=args.style_dim * 2,
            context_features=args.style_dim * 2,
        )
        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(
                mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
            ),
            sigma_data=args.diffusion.dist.sigma_data,
            dynamic_threshold=0.0,
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer
        nets = {
            "bert": bert,
            "bert_encoder": torch.nn.Linear(bert.config.hidden_size, args.hidden_dim),
            "predictor": st.ProsodyPredictor(
                style_dim=args.style_dim,
                d_hid=args.hidden_dim,
                nlayers=args.n_layer,
                max_dur=args.max_dur,
                dropout=args.dropout,
            ),
            "decoder": Decoder(
                dim_in=args.hidden_dim,
                style_dim=args.style_dim,
                dim_out=args.n_mels,
                resblock_kernel_sizes=dec.resblock_kernel_sizes,
                upsample_rates=dec.upsample_rates,
                upsample_initial_channel=dec.upsample_initial_channel,
                resblock_dilation_sizes=dec.resblock_dilation_sizes,
                upsample_kernel_sizes=dec.upsample_kernel_sizes,
            ),
            "text_encoder": st.TextEncoder(
                channels=args.hidden_dim,
                kernel_size=5,
                depth=args.n_layer,
                n_symbols=args.n_token,
            ),
            "predictor_encoder": st.StyleEncoder(
                dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
            ),
            "style_encoder": st.StyleEncoder(
                dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
            ),
            "diffusion": diffusion,
        }
        weights = torch.load(self.weights_file, map_location="cpu", weights_only=True)
        for key, net in nets.items():
            state = {k: v.float() if v.is_floating_point() else v for k, v in weights[key].items()}
            state.pop("embeddings.position_ids", None)
            result = net.load_state_dict(state, strict=False)
            missing = [k for k in result.missing_keys if not k.endswith("position_ids")]
            if missing or result.unexpected_keys:
                raise BackendError(
                    f"StyleTTS2 weights do not match the model ({key}): "
                    f"missing {missing[:3]}, unexpected {result.unexpected_keys[:3]}"
                )
            net.eval().to(self._device)
        sampler = DiffusionSampler(
            diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        return types.SimpleNamespace(**nets), sampler

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.reference_audio is not None:
            raise BackendError("This StyleTTS2 backend does not clone voices")
        if not self.supports_language(request.language):
            raise BackendError(f"StyleTTS2 voices only speak Basque (eu), not {request.language!r}")
        speaker = request.speaker or DEFAULT_SPEAKER
        if speaker not in SPEAKERS:
            raise BackendError(
                f"Unknown StyleTTS2 speaker {speaker!r}. Available: {', '.join(SPEAKERS)}"
            )
        params = request.params
        emotion = str(params.get("emotion", "neu")).lower()
        if emotion not in EMOTIONS:
            raise BackendError(
                f"Unknown emotion {emotion!r}; use neu (neutral), poz (happy) or tri (sad)"
            )
        # How much of the emotional reference's prosody to keep (the model repo's
        # ``1 - beta``). Its default of 0.3 mutes the emotion; 0.8 keeps it clear.
        intensity = float(params.get("intensity", DEFAULT_INTENSITY))
        if not 0.0 <= intensity <= 1.0:
            raise BackendError(f"intensity must be between 0 and 1, got {intensity:g}")
        speed = float(params.get("speed", 1.0))
        seed = int(params.get("seed", self.seed))
        tokens = self._tokens(request.text)
        self.load()

        import torch

        style = self._style(torch, speaker, EMOTIONS[emotion])
        torch.manual_seed(seed)
        try:
            with torch.no_grad():
                audio = self._infer(torch, tokens, style, intensity, speed)
        except RuntimeError as exc:
            raise BackendError(f"StyleTTS2 failed on {request.text[:40]!r}: {exc}") from exc
        return SynthesisResult(audio, SAMPLE_RATE)

    def _infer(
        self, torch: Any, tokens: list[int], ref: Any, intensity: float, speed: float
    ) -> np.ndarray:
        beta = 1.0 - intensity
        # Port of Synthesizer.run() from the model repository's inference.py.
        m, dev = self._model, self._device
        tokens_t = torch.LongTensor([0, *tokens]).unsqueeze(0).to(dev)
        lengths = torch.LongTensor([tokens_t.shape[-1]]).to(dev)
        mask = torch.gt(torch.arange(int(lengths.max())).unsqueeze(0).to(dev) + 1, lengths[:, None])

        t_en = m.text_encoder(tokens_t, lengths, mask)
        bert_dur = m.bert(tokens_t, attention_mask=(~mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        noise = torch.randn((1, 256)).unsqueeze(1).to(dev)
        s_pred = self._sampler(
            noise=noise,
            embedding=bert_dur,
            embedding_scale=self.embedding_scale,
            features=ref,
            num_steps=self.diffusion_steps,
        ).squeeze(1)
        s = beta * s_pred[:, 128:] + (1 - beta) * ref[:, 128:]
        ref_vec = self.alpha * s_pred[:, :128] + (1 - self.alpha) * ref[:, :128]

        d = m.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1) / speed
        pred_dur = torch.round(duration.squeeze(0)).clamp(min=1).long().cpu()
        aln = torch.zeros(pred_dur.numel(), int(pred_dur.sum()))
        frame = 0
        for i, n in enumerate(pred_dur.tolist()):
            aln[i, frame : frame + n] = 1
            frame += n
        aln = aln.unsqueeze(0).to(dev)
        en = d.transpose(-1, -2) @ aln
        asr = t_en @ aln
        # HiFi-GAN decoder: shift features one frame to the right.
        en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
        asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
        f0, n_pred = m.predictor.F0Ntrain(en, s)
        out = m.decoder(asr, f0, n_pred, ref_vec.squeeze().unsqueeze(0))
        return out.squeeze().float().cpu().numpy()[..., :-50].astype(np.float32)

    def _style(self, torch: Any, speaker: str, emotion: str) -> Any:
        key = f"{speaker}_{emotion}"
        if key not in self._styles:
            path = self._model_file(f"Demo/ref_{key}.wav")
            mel = torch.from_numpy(_style_mel(_load_reference(path))).to(self._device)
            with torch.no_grad():
                ref_s = self._model.style_encoder(mel[None, None])
                ref_p = self._model.predictor_encoder(mel[None, None])
            self._styles[key] = torch.cat([ref_s, ref_p], dim=1)
        return self._styles[key]

    def _tokens(self, text: str) -> list[int]:
        text = text.strip().replace("…", "...")
        if text and text[-1] not in ".!?;:":
            text += "."
        normalized = re.sub(r"\.{2,}", ".", self._modulo1y2(text, "-TxtMode=Word"))
        phonemes = align_phonemes(normalized, self._modulo1y2(normalized))
        return [SYMBOLS[ch] for ch in phonemes if ch in SYMBOLS]

    def _modulo1y2(self, text: str, *flags: str) -> str:
        # The text is fully validated first: the phonemiser reads ISO-8859-15.
        try:
            data = text.encode("iso-8859-15")
        except UnicodeEncodeError as exc:
            raise BackendError(f"The Basque phonemiser cannot read {text[exc.start]!r}") from exc
        phon = self._model_file("phonemizer")
        result = subprocess.run(
            [
                *("docker", "run", "--rm", "-i", "--platform", PLATFORM, "--network", "none"),
                *("-v", f"{phon}:/p:ro", IMAGE),
                *("/p/modulo1y2/modulo1y2", *flags, "-Lang=eu", "-HDic=/p/dict/eu_dicc"),
            ],
            input=data,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            error = result.stderr.decode(errors="replace").strip()
            raise BackendError(f"Basque phonemiser failed: {error}")
        return result.stdout.decode("iso-8859-15").strip()

    def speakers(self) -> list[SpeakerInfo]:
        return [SpeakerInfo(id=s, language="eu", gender=g) for s, g in SPEAKERS.items()]

    def default_speaker(self, language: str) -> str | None:
        return DEFAULT_SPEAKER

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "model": MODEL_REPO, "device": self._device}
        if importlib.util.find_spec("torch"):
            import torch

            info["torch"] = torch.__version__
        return info


def align_phonemes(normalized: str, raw: str) -> str:
    """Aholab word phonemes -> the model's IPA string, keeping punctuation in place.

    Port of ``Phonemizer.getPhonemes(use_single_char=True)`` plus the clean-up in
    ``Synthesizer._phonemize`` (model repository).
    """
    chunks = [c.strip() for c in raw.replace("\n", " | _ | ").split(" | ")]
    phonemes = [c for c in chunks if c and c != "_"]
    words = _WORDS.findall(normalized)
    plain = [w for w in words if w not in string.punctuation]
    while phonemes and len(phonemes) < len(plain):
        phonemes.append(phonemes[-1])

    def convert(seq: str) -> str:
        ipa = [_SAMPA_TO_IPA.get(p, p) for p in seq.split() if p != "-"]
        return "".join(_SINGLE_CHAR.get(p, p) for p in ipa)

    out: list[str] = []
    index = 0
    for word in words:
        if word in string.punctuation:
            out.append(word)
        elif index < len(phonemes):
            out.append(convert(phonemes[index]))
            index += 1
    out.extend(convert(p) for p in phonemes[index:])
    return " ".join(" ".join(out).replace(" | ", " ").replace("|", " ").split())


def _load_reference(path: Path, top_db: float = 30.0) -> np.ndarray:
    """Mono 24 kHz reference with leading/trailing silence trimmed (as in the model repo)."""
    import soundfile as sf

    from voxlab.audio.dsp import resample

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = resample(audio.mean(axis=1), sr, SAMPLE_RATE)
    frame, hop = 2048, 512
    if wav.size >= frame:
        frames = np.lib.stride_tricks.sliding_window_view(wav, frame)[::hop]
        rms = np.sqrt((frames**2).mean(axis=1))
        loud = np.nonzero(rms > rms.max() * 10 ** (-top_db / 20))[0]
        if loud.size:
            wav = wav[loud[0] * hop : min((loud[-1] + 1) * hop + frame, wav.size)]
    return wav


def _style_mel(wav: np.ndarray) -> np.ndarray:
    """Normalised log-mel as torchaudio.MelSpectrogram() with the model repo's settings.

    Note: the repo builds the transform without ``sample_rate``, so its filterbank
    assumes torchaudio's 16 kHz default even though the audio is 24 kHz.
    """
    import torch

    n_fft, win, hop, n_mels, fb_rate = 2048, 1200, 300, 80, 16000
    spec = (
        torch.stft(
            torch.from_numpy(wav).float(),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=torch.hann_window(win),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        ** 2
    )

    def to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    freqs = torch.linspace(0, fb_rate // 2, n_fft // 2 + 1)
    mels = torch.linspace(to_mel(0.0), to_mel(fb_rate / 2), n_mels + 2)
    points = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    diff = points[1:] - points[:-1]
    slopes = points.unsqueeze(0) - freqs.unsqueeze(1)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fbank = torch.clamp(torch.min(down, up), min=0)
    mel = (spec.transpose(0, 1) @ fbank).transpose(0, 1)
    mel = (torch.log(1e-5 + mel) + 4.0) / 4.0
    if mel.shape[-1] < 16:
        mel = torch.nn.functional.pad(mel, (0, 16 - mel.shape[-1]))
    return mel.numpy()


@contextlib.contextmanager
def _quiet():
    """Silence the model code's prints and library warnings unless debugging."""
    if log.isEnabledFor(logging.DEBUG):
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield
