import os
import json
import time
import typer
from datetime import timedelta
from pathlib import Path

from translation_service import translate_srt_file
from utils import get_video_title, normalize_segments, write_srt_pretty

app = typer.Typer()

# Default model with timestamps (Mandarin Chinese)
FUNASR_MODEL_ID = "manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline"

def ensure_wav_16k(src_path: Path) -> Path:
    """Convert to mono 16 kHz WAV (recommended for FunASR ONNX)."""
    wav_path = src_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    cmd = f'ffmpeg -y -i "{src_path}" -ac 1 -ar 16000 "{wav_path}"'
    os.system(cmd)
    return wav_path

def funasr_transcribe(audio_path: Path):
    """
    FunASR transcription (AutoModel/Paraformer-zh) with sentence timestamps.
    Returns {"text": str, "segments": [{"start": s, "end": e, "text": "..."}]}
    """
    from funasr import AutoModel

    # Convert to WAV 16 kHz mono (recommended)
    wav16 = ensure_wav_16k(audio_path)

    # AutoModel: Paraformer-zh + VAD + punctuation (official path)
    # hub="ms" => downloads from ModelScope; sentence_timestamp=True => returns "sentence_info"
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        hub="ms",
    )

    # Generate with sentence timestamps
    res = model.generate(
        input=str(wav16),
        batch_size_s=300,           # segment length for long recordings
        sentence_timestamp=True,    # <-- key: returns "sentence_info"
        return_whole_asr=True       # consistent text in "text" field
    )
    # Typically: res = [ { "text": "...", "sentence_info": [{"start":..,"end":..,"text":"..."}], ... } ]
    # (depending on version there may be minor differences)
    rec = res[0] if isinstance(res, list) and res else (res or {})

    # Text
    full_text = rec.get("text") or rec.get("preds") or ""
    if isinstance(full_text, list):
        full_text = "".join(map(str, full_text))

    # Sentence segments
    sent_info = rec.get("sentence_info") or []
    segments = []
    if isinstance(sent_info, list) and sent_info and isinstance(sent_info[0], dict):
        for i, s in enumerate(sent_info):
            s0 = float(s.get("start", 0) or 0.0)
            e0 = float(s.get("end", s0) or s0)
            txt = (s.get("text") or "").strip()
            segments.append({"id": i, "seek": 0, "start": s0, "end": e0, "text": txt})

    # Fallback: if for some reason sentence_info is missing — try "timestamp" (character-based)
    if not segments:
        ts = rec.get("timestamp") or []
        if isinstance(ts, list) and ts and isinstance(ts[0], (list, tuple)) and len(ts[0]) >= 3:
            # Roughly reconstruct sentences by punctuation
            cur = []
            cur_s = None
            last_e = None
            punct = set("。！？!?，,．.；;、")
            def flush():
                nonlocal cur, cur_s, last_e, segments
                if cur:
                    text = "".join([c[0] for c in cur]).strip()
                    if text:
                        segments.append({
                            "id": len(segments), "seek": 0,
                            "start": float(cur_s) if cur_s is not None else 0.0,
                            "end": float(last_e) if last_e is not None else 0.0,
                            "text": text
                        })
                    cur, cur_s = [], None
            for ch, st, ed in ts:
                st = float(st or 0.0); ed = float(ed or st)
                if cur_s is None: cur_s = st
                cur.append((ch, st, ed)); last_e = ed
                if ch and ch in punct: flush()
            flush()

    # Final fallback
    if not segments:
        segments = [{"id": 0, "seek": 0, "start": 0.0, "end": 0.0, "text": str(full_text)}]

    return {"text": full_text, "segments": segments}


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube URL to transcribe"),
    language: str = typer.Option("en", "--language", "-l", help="Language code ('en', 'pl' for English or Polish)"),
    skip_download: bool = typer.Option(False, "--skip-download", "-s", help="Skip audio download"),
    skip_whisper: bool = typer.Option(False, "--skip-whisper", "-w", help="Skip ASR (FunASR) step"),
    output_dir: str = typer.Option("output", "--output-dir", "-o", help="Base output directory"),
):
    """Transcribe and translate YouTube video from Chinese to English (FunASR)"""

    # Title + output directory
    if not skip_download:
        video_title = get_video_title(url)
        print(f"Video title: {video_title}")
    else:
        video_title = "manual_transcription"

    output_path = Path(output_dir) / video_title
    output_path.mkdir(parents=True, exist_ok=True)

    os.chdir(output_path)
    print(f"Working directory: {output_path.absolute()}")

    # Download audio (MP3), then convert to WAV 16k
    if not skip_download:
        print("Downloading audio...")
        os.system(f'yt-dlp --extract-audio --audio-format mp3 --output "audio.mp3" "{url}"')

    start_time = time.time()
    result = None
    segments = None

    if skip_whisper:
        print("Skipping ASR (FunASR).")
    else:
        print("Starting transcription with FunASR (Paraformer ONNX, ZH)...")
        audio_path = Path("audio.mp3") if Path("audio.mp3").exists() else Path("audio.wav")
        result = funasr_transcribe(audio_path=audio_path)

    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {timedelta(seconds=elapsed_time)}")

    # Processing results
    if skip_whisper:
        print("Warning: ASR was skipped, no transcription available")
        return
    else:
        full_text = result.get("text", "")
        segments_raw = result.get("segments", [])
        segments = normalize_segments(segments_raw) if segments_raw else []

    print(f"Chinese transcription (first 100 chars): {full_text[:100]}")

    # Save
    if not skip_whisper:
        with open("transcription.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if segments:
            write_srt_pretty(segments, "transcription.srt")
            print("SRT file created: transcription.srt")
        else:
            print("No segments to write to SRT.")

    # Translation
    start_time = time.time()
    print("Starting translation from Chinese...")

    if os.path.exists("transcription.srt"):
        translate_srt_file("transcription.srt", "translation.srt", language)
        print("SRT translation completed.")

    elapsed_time = time.time() - start_time
    print(f"Translation completed in {timedelta(seconds=elapsed_time)}")

if __name__ == "__main__":
    app()
