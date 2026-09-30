#!/usr/bin/env python3
"""
Diarization hyperparameter sweep.

Runs pyannote community-1 diarization on a single audio file across several
(clustering threshold, min_duration_off) settings and reports, for each config:
the number of speakers found, the number of speaker turns, the number of
speaker changes, and a turn timeline. Use it to see which settings recover the
rapid/short turns that the defaults merge into one speaker.

This isolates the diarizer (no transcription needed), so it is fast and the
turn timeline directly exposes merged/missed turns.

Run it on the host (models + HF cache are shared via CACHE_DIR):

    # HF_TOKEN must be set (export it or load .env first)
    uv run python tests/diarize_sweep.py testfiles/Recording\\ 4.flac

    # known speaker count or bounds help a lot -- pass them through:
    uv run python tests/diarize_sweep.py audio.flac --min-speakers 2

    # custom grid as "threshold:min_duration_off" pairs (use 'none' to skip a knob):
    uv run python tests/diarize_sweep.py audio.flac \
        --grid "none:none,0.5:0.1,0.45:0.0,0.4:0.1"
"""

import argparse
import copy
import os
import sys
import time

import whispermlx
from whispermlx.diarize import DiarizationPipeline

DEVICE = os.getenv("DEVICE", "mps")
HF_TOKEN = os.getenv("HF_TOKEN")
SAMPLE_RATE = 16000

# (label, clustering_threshold, min_duration_off). None = leave at model default.
DEFAULT_GRID = [
    ("baseline (model defaults)", None, None),
    ("threshold=0.5", 0.5, None),
    ("threshold=0.5 + min_off=0.1", 0.5, 0.1),
    ("threshold=0.45 + min_off=0.0", 0.45, 0.0),
    ("threshold=0.4 + min_off=0.1", 0.4, 0.1),
]


def _set_scoped(params, section, key, value):
    """Set params[section][key]=value only if that exact path exists. Returns ok."""
    sec = params.get(section)
    if isinstance(sec, dict) and key in sec:
        sec[key] = value
        return True
    return False


def _to_turns(diarize_output):
    """Normalize whatever the diarizer returns into a list of (start, end, speaker)."""
    obj = diarize_output
    if isinstance(obj, tuple):  # (segments, embeddings) when return_embeddings=True
        obj = obj[0]
    if hasattr(obj, "exclusive_speaker_diarization"):
        obj = obj.exclusive_speaker_diarization
    # pandas DataFrame (classic whisperx return)
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return [(float(r.start), float(r.end), str(r.speaker)) for r in obj.itertuples()]
    except Exception:
        pass
    # pyannote Annotation
    if hasattr(obj, "itertracks"):
        return [(float(seg.start), float(seg.end), str(spk)) for seg, _, spk in obj.itertracks(yield_label=True)]
    raise TypeError(f"Unrecognized diarization output type: {type(obj)}")


def _parse_grid(grid_str):
    out = []
    for pair in grid_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        thr_s, _, off_s = pair.partition(":")
        thr = None if thr_s.strip().lower() in ("", "none") else float(thr_s)
        off = None if off_s.strip().lower() in ("", "none") else float(off_s)
        bits = []
        if thr is not None:
            bits.append(f"threshold={thr}")
        if off is not None:
            bits.append(f"min_off={off}")
        out.append((", ".join(bits) or "model defaults", thr, off))
    return out


def main():
    ap = argparse.ArgumentParser(description="pyannote community-1 diarization sweep")
    ap.add_argument("audio", help="path to an audio file")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=50, help="timeline rows to print per config")
    ap.add_argument(
        "--grid",
        default=None,
        help='comma-separated "threshold:min_duration_off" pairs; "none" skips a knob',
    )
    args = ap.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    grid = _parse_grid(args.grid) if args.grid else DEFAULT_GRID

    print(f"Loading audio: {args.audio}")
    audio = whispermlx.load_audio(args.audio)
    print(f"Duration: {len(audio) / SAMPLE_RATE:.1f}s\n")

    print("Loading diarization pipeline (pyannote/speaker-diarization-community-1)...")
    wrapper = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
        device=DEVICE,
    )
    pipe = wrapper.model
    try:
        baseline = copy.deepcopy(dict(pipe.parameters(instantiated=True)))
    except Exception as e:
        print(f"WARNING: could not read default parameters ({e}); only 'baseline' config will run.")
        baseline = None
    if baseline is not None:
        print("Default hyperparameters (valid keys for --grid / DIARIZE_PARAM_OVERRIDES):")
        print(f"  {baseline}\n")

    spk_kwargs = {}
    if args.num_speakers:
        spk_kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers:
        spk_kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers:
        spk_kwargs["max_speakers"] = args.max_speakers
    if spk_kwargs:
        print(f"Speaker constraints: {spk_kwargs}\n")

    summary = []
    for label, thr, min_off in grid:
        notes = []
        if baseline is not None and (thr is not None or min_off is not None):
            params = copy.deepcopy(baseline)
            if thr is not None:
                ok = _set_scoped(params, "clustering", "threshold", thr)
                notes.append(f"clustering.threshold={thr}" + ("" if ok else " (KEY NOT FOUND)"))
            if min_off is not None:
                ok = _set_scoped(params, "segmentation", "min_duration_off", min_off)
                notes.append(f"segmentation.min_duration_off={min_off}" + ("" if ok else " (KEY NOT FOUND)"))
            pipe.instantiate(params)
        elif baseline is None and (thr is not None or min_off is not None):
            print(f"Skipping '{label}': default params unavailable for override.\n")
            continue

        t0 = time.time()
        try:
            turns = _to_turns(wrapper(audio, **spk_kwargs))
        except Exception as e:
            print("=" * 74)
            print(f"CONFIG: {label}\n  FAILED: {e}\n")
            continue
        elapsed = time.time() - t0

        turns.sort(key=lambda t: t[0])
        speakers = sorted({t[2] for t in turns})
        seq = [t[2] for t in turns]
        changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

        print("=" * 74)
        print(f"CONFIG: {label}")
        if notes:
            print(f"  overrides: {', '.join(notes)}")
        print(f"  speakers: {len(speakers)} {speakers}")
        print(f"  turns: {len(turns)} | speaker changes: {changes} | {elapsed:.1f}s")
        print("  timeline:")
        for start, end, spk in turns[: args.max_turns]:
            print(f"    {start:8.2f} - {end:8.2f}  {spk}")
        if len(turns) > args.max_turns:
            print(f"    ... ({len(turns) - args.max_turns} more turns)")
        print()
        summary.append((label, len(speakers), len(turns), changes, elapsed))

    if summary:
        print("=" * 74)
        print("SUMMARY (more turns/changes = finer segmentation, recovers rapid turns)")
        print(f"  {'config':<34} {'spk':>4} {'turns':>6} {'changes':>8} {'sec':>6}")
        for label, nspk, nturns, nch, sec in summary:
            print(f"  {label:<34} {nspk:>4} {nturns:>6} {nch:>8} {sec:>6.1f}")


if __name__ == "__main__":
    main()
