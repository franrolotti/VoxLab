"""Command-line interface. Thin layer: parses arguments and delegates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from voxlab import __version__
from voxlab.audio.export import SUPPORTED_BIT_DEPTHS, SUPPORTED_SAMPLE_RATES
from voxlab.audio.presets import PresetLibrary
from voxlab.config import Config, load_config
from voxlab.dialogue.parser import load_dialogue
from voxlab.errors import VoxLabError
from voxlab.pipeline import GenerateOptions, Pipeline
from voxlab.storage import clean_model_dir, dir_size, human_size
from voxlab.tts.factory import (
    backend_names,
    create_backend,
    get_backend_class,
    resolve_backend_name,
)
from voxlab.voices.manager import VoiceManager

log = logging.getLogger("voxlab")


def main(argv: Sequence[str] | None = None) -> int:
    # Privacy: never send usage telemetry to the Hugging Face Hub.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0
    try:
        config = load_config(args.config)
        return args.handler(args, config) or 0
    except VoxLabError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxlab",
        description="VoxLab: local AI voice generation for dialogues, with audio style presets.",
    )
    parser.add_argument("--version", action="version", version=f"voxlab {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="only show warnings and errors")
    sub = parser.add_subparsers(title="commands", metavar="<command>")

    gen = sub.add_parser(
        "generate",
        help="synthesize a dialogue file to WAV",
        description="Synthesize a dialogue file into a single WAV file.",
    )
    gen.add_argument("dialogue", type=Path, help="dialogue .txt file")
    gen.add_argument("-p", "--preset", help="audio preset name or .yaml path (default: config)")
    gen.add_argument("-o", "--output", type=Path, help="output .wav (default: output/<name>.wav)")
    gen.add_argument(
        "--voice",
        action="append",
        default=[],
        metavar="SPEAKER=VOICE",
        help="assign a voice to a speaker (repeatable)",
    )
    gen.add_argument("--language", help="default language code, e.g. es, en")
    gen.add_argument("--sample-rate", type=int, choices=SUPPORTED_SAMPLE_RATES)
    gen.add_argument("--bit-depth", type=int, choices=SUPPORTED_BIT_DEPTHS)
    gen.add_argument(
        "--strict", action="store_true", help="fail on speakers without an assigned voice"
    )
    gen.add_argument(
        "--dry-run", action="store_true", help="validate and show the plan without synthesizing"
    )
    gen.set_defaults(handler=cmd_generate)

    voices = sub.add_parser(
        "voices", help="list voice profiles", description="List voice profiles, or manage them."
    )
    voices.add_argument(
        "--speakers", action="store_true", help="list the backend's built-in speakers instead"
    )
    voices.set_defaults(handler=cmd_voices)
    voices_sub = voices.add_subparsers(title="subcommands", metavar="<subcommand>")
    add = voices_sub.add_parser("add", help="create a voice profile in the voices folder")
    add.add_argument("name", help="voice name (lowercase, e.g. captain)")
    add.add_argument("--reference", type=Path, help="reference clip for voice cloning")
    add.add_argument("--speaker", help="backend speaker id or blend, e.g. em_alex:0.7,am_adam:0.3")
    add.add_argument("--language", help="default language of the voice")
    add.add_argument("--description", default="", help="short description")
    add.add_argument("--force", action="store_true", help="overwrite an existing profile")
    add.set_defaults(handler=cmd_voices_add)

    presets = sub.add_parser("presets", help="list audio presets")
    presets.set_defaults(handler=cmd_presets)

    models = sub.add_parser(
        "models",
        help="show TTS backends and model status",
        description="Show TTS backends, model files and disk usage.",
    )
    models.add_argument("--download", action="store_true", help="download the configured model")
    models.set_defaults(handler=cmd_models)

    bench = sub.add_parser(
        "benchmark",
        help="measure speed and memory of the installed model",
        description="Benchmark the installed model (never downloads others).",
    )
    bench.add_argument("--language", nargs="+", help="languages to test (default: es en)")
    bench.add_argument("--runs", type=int, default=1, help="repetitions per sentence")
    bench.add_argument("--json", type=Path, help="also write the report as JSON")
    bench.set_defaults(handler=cmd_benchmark)

    clean = sub.add_parser(
        "clean",
        help="delete downloaded models and caches",
        description="Delete VoxLab's model directory (storage.model_dir).",
    )
    clean.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    clean.add_argument("--dry-run", action="store_true", help="only show what would be removed")
    clean.set_defaults(handler=cmd_clean)
    return parser


# --- commands ---------------------------------------------------------------


def cmd_generate(args: argparse.Namespace, config: Config) -> int:
    dialogue = load_dialogue(args.dialogue)
    options = GenerateOptions(
        preset=args.preset,
        output=args.output,
        cast=_parse_assignments(args.voice),
        language=args.language,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        strict=True if args.strict else None,
    )
    pipeline = Pipeline(config)
    if args.dry_run:
        plan = pipeline.plan(dialogue, options)
        _print_table(
            ["#", "speaker", "voice", "lang", "speaker id", "text"],
            [
                [
                    str(j.index + 1),
                    j.line.speaker,
                    j.voice.name,
                    j.request.language,
                    j.request.speaker or "(default)",
                    _shorten(j.line.text, 40),
                ]
                for j in plan.jobs
            ],
        )
        print(f"\npreset: {plan.preset.name}  ->  {pipeline.output_path(dialogue, options)}")
        return 0

    result = pipeline.generate(dialogue, options)
    cast = ", ".join(f"{s}={v}" for s, v in result.cast.items())
    print(
        f"✔ {result.output}  ({result.duration:.1f} s audio, {result.lines} lines, "
        f"preset {result.preset}, {result.elapsed:.1f} s)"
    )
    print(f"  cast: {cast}")
    return 0


def cmd_voices(args: argparse.Namespace, config: Config) -> int:
    if args.speakers:
        backend = create_backend(config)
        speakers = backend.speakers()
        if not speakers:
            print("Model not downloaded yet. Run: voxlab models --download")
            return 1
        _print_table(
            ["speaker id", "language", "gender"], [[s.id, s.language, s.gender] for s in speakers]
        )
        return 0

    manager = VoiceManager(config.voices_dir)
    rows = []
    for voice in manager.all():
        if voice.uses_cloning:
            kind = "clone"
        elif isinstance(voice.speaker, str):
            kind = voice.speaker
        else:
            kind = "preset"
        rows.append(
            [voice.name, voice.source, kind, voice.language or "-", _shorten(voice.description, 50)]
        )
    _print_table(["voice", "source", "speaker", "lang", "description"], rows)
    print(f"\nUser voices folder: {config.voices_dir}")
    return 0


def cmd_voices_add(args: argparse.Namespace, config: Config) -> int:
    manager = VoiceManager(config.voices_dir)
    voice = manager.add(
        args.name,
        reference=args.reference,
        speaker=args.speaker,
        language=args.language,
        description=args.description,
        overwrite=args.force,
    )
    print(f"✔ Created voice {voice.name!r} in {config.voices_dir / voice.name}")
    if voice.uses_cloning:
        backend_cls = get_backend_class(resolve_backend_name(config.tts.backend))
        if not backend_cls.capabilities.voice_cloning:
            print(
                f"  note: backend {backend_cls.name!r} cannot clone voices; "
                "the reference clip will be used by backends that can."
            )
    return 0


def cmd_presets(args: argparse.Namespace, config: Config) -> int:
    library = PresetLibrary(config.preset_dirs)
    rows = []
    for preset in library.all():
        effects = " > ".join(step.effect for step in preset.chain)
        rows.append([preset.name, _shorten(preset.description, 45), _shorten(effects, 60)])
    _print_table(["preset", "description", "chain"], rows)
    return 0


def cmd_models(args: argparse.Namespace, config: Config) -> int:
    backend = create_backend(config)
    if args.download:
        backend.download()
    rows = []
    for name in backend_names():
        cls = get_backend_class(name)
        active = name == backend.name
        info = backend.model_info() if active else None
        status = ("downloaded" if backend.is_downloaded() else "not downloaded") if active else "-"
        rows.append(
            [
                name + (" *" if active else ""),
                info.model_id if info else "-",
                info.variant if info else "-",
                info.license if info else "-",
                f"~{info.download_mb} MB" if info else "-",
                "yes" if cls.capabilities.voice_cloning else "no",
                status,
            ]
        )
    _print_table(["backend", "model", "variant", "license", "size", "cloning", "status"], rows)
    print(f"\nModel directory: {config.model_dir} ({human_size(dir_size(config.model_dir))})")
    if not backend.is_downloaded():
        print("Download with: voxlab models --download")
    return 0


def cmd_benchmark(args: argparse.Namespace, config: Config) -> int:
    from voxlab.benchmark import run_benchmark

    backend = create_backend(config)
    if not backend.is_downloaded():
        print("The model is not downloaded; benchmark only uses installed models.")
        print("Download with: voxlab models --download")
        return 1
    report = run_benchmark(backend, languages=args.language, runs=max(1, args.runs))
    system = report.system
    print(
        f"System : {system['cpu']} ({system['cpu_count']} threads), "
        f"{system['ram_gb']} GB RAM, {system['os']}, Python {system['python']}"
    )
    print("Runtime: " + ", ".join(f"{k}={v}" for k, v in report.runtime.items()))
    print(f"Load   : {report.load_seconds:.2f} s\n")
    _print_table(
        ["lang", "text", "gen (s)", "audio (s)", "RTF"],
        [
            [
                r.language,
                _shorten(r.text, 40),
                f"{r.seconds:.2f}",
                f"{r.audio_seconds:.2f}",
                f"{r.rtf:.3f}",
            ]
            for r in report.results
        ],
    )
    memory = f"{report.peak_memory_mb:.0f} MB" if report.peak_memory_mb else "n/a"
    print(
        f"\nTotal  : {report.total_seconds:.2f} s for {report.total_audio_seconds:.2f} s of "
        f"audio  ->  RTF {report.rtf:.3f} ({1 / report.rtf:.1f}x real time)"
    )
    print(f"Peak memory: {memory}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        print(f"Report written to {args.json}")
    return 0


def cmd_clean(args: argparse.Namespace, config: Config) -> int:
    target = config.model_dir
    size = clean_model_dir(target, dry_run=True)
    if size == 0 and not target.exists():
        print(f"Nothing to clean: {target} does not exist.")
        return 0
    print(f"This will delete {target} ({human_size(size)}).")
    if args.dry_run:
        return 0
    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "s", "si", "sí"):
            print("Aborted.")
            return 1
    freed = clean_model_dir(target)
    print(f"✔ Freed {human_size(freed)}. Models will be downloaded again on next use.")
    return 0


# --- helpers ----------------------------------------------------------------


def _parse_assignments(items: list[str]) -> dict[str, str]:
    cast: dict[str, str] = {}
    for item in items:
        speaker, sep, voice = item.partition("=")
        if not sep or not speaker.strip() or not voice.strip():
            raise VoxLabError(f"--voice expects SPEAKER=VOICE, got {item!r}")
        cast[speaker.strip()] = voice.strip()
    return cast


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows, strict=False)]
    line = "  ".join(h.upper().ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)s: %(message)s" if verbose else "%(message)s"
    )
    if not verbose:
        # Keep third-party chatter out of normal output.
        for name in ("huggingface_hub", "kokoro_onnx", "phonemizer", "httpx", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    sys.exit(main())
