import os
import json
import subprocess
import time
import typer
from datetime import timedelta
from pathlib import Path

from translation_service import translate_srt_file
from utils import get_video_title, normalize_segments, write_srt_pretty, shift_srt_timestamps

app = typer.Typer()

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
    skip_download: bool = typer.Option(False, "--skip-download", "-s", help="Skip audio download"),
    skip_asr: bool = typer.Option(False, "--skip-asr", "-w", help="Skip ASR (FunASR) step"),
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

    if skip_asr:
        print("Skipping ASR (FunASR).")
    else:
        print("Starting transcription with FunASR...")
        audio_path = Path("audio.mp3") if Path("audio.mp3").exists() else Path("audio.wav")
        result = funasr_transcribe(audio_path=audio_path)

    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {timedelta(seconds=elapsed_time)}")

    # Processing results
    if skip_asr:
        print("Warning: ASR was skipped, no transcription available")
        return
    else:
        full_text = result.get("text", "")
        segments_raw = result.get("segments", [])
        segments = normalize_segments(segments_raw) if segments_raw else []

    print(f"Chinese transcription (first 100 chars): {full_text[:100]}")

    # Save
    if not skip_asr:
        with open("transcription.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if segments:
            write_srt_pretty(segments, "transcription.srt")
            print("SRT file created: transcription.srt")
        else:
            print("No segments to write to SRT.")


@app.command()
def translate(
    transcription_dir: str = typer.Argument(..., help="Directory containing transcription.srt file"),
    language: str = typer.Option("en", "--language", "-l", help="Language code ('en', 'pl' for English or Polish)"),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Number of segments to translate per batch (default: 5)"),
):
    """Translate existing transcription SRT file without running transcription"""
    
    transcription_path = Path(transcription_dir)
    
    if not transcription_path.exists():
        print(f"Error: Directory '{transcription_dir}' does not exist")
        return
    
    if not transcription_path.is_dir():
        print(f"Error: '{transcription_dir}' is not a directory")
        return
    
    # Change to the transcription directory
    original_cwd = Path.cwd()
    os.chdir(transcription_path)
    print(f"Working directory: {transcription_path.absolute()}")
    
    try:
        # Check if transcription.srt exists
        transcription_srt = Path("transcription.srt")
        if not transcription_srt.exists():
            print("Error: transcription.srt not found in directory")
            return
    
        print(f"Found transcription file: {transcription_srt.name}")
        
        # Translation
        start_time = time.time()
        print(f"Starting translation to {language}...")
        
        output_file = "translation.srt"
        translate_srt_file("transcription.srt", output_file, language, context_window=3, batch_size=batch_size)
        
        elapsed_time = time.time() - start_time
        print(f"Translation completed in {timedelta(seconds=elapsed_time)}")
        print(f"Translation saved to: {output_file}")
        
    finally:
        os.chdir(original_cwd)


@app.command()
def generate_video(
    url: str = typer.Argument(..., help="YouTube URL to download video"),
    srt_file: str = typer.Option("translation.srt", "--srt", "-s", help="SRT subtitle file to embed"),
    output_dir: str = typer.Option("output", "--output-dir", "-o", help="Base output directory"),
    font_size: int = typer.Option(24, "--font-size", "-f", help="Subtitle font size"),
    font_color: str = typer.Option("white", "--font-color", "-c", help="Subtitle font color"),
    outline_color: str = typer.Option("black", "--outline-color", help="Subtitle outline color"),
    offset_seconds: float = typer.Option(18.0, "--offset", help="Subtitle timing offset in seconds (to skip intro)"),
):
    """Generate MP4 video that burns SRT, caps height at 720px (no upscaling),
    and preserves audio. Requires ffmpeg with libass enabled."""

    # Get video title and prepare output directory
    video_title = get_video_title(url)
    print(f"Video title: {video_title}")

    output_path = Path(output_dir) / video_title
    output_path.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(output_path)
    print(f"Working directory: {output_path.absolute()}")

    try:
        # Ensure SRT exists
        srt_path = Path(srt_file)
        if not srt_path.exists():
            print(f"Error: SRT file '{srt_file}' not found in {output_path}")
            print("Available SRT files in this folder:")
            for f in output_path.glob("*.srt"):
                print(f"  - {f.name}")
            return
        print(f"Using SRT file: {srt_path.name}")

        # Shift SRT timestamps
        shifted_srt = f"{srt_path.stem}_shifted{srt_path.suffix}"
        print(f"Shifting subtitles by {offset_seconds} seconds...")
        if not shift_srt_timestamps(srt_path.name, shifted_srt, offset_seconds):
            print("Error: Failed to shift subtitle timestamps")
            return

        # Download video+audio (prefer >=1080p → 720p → lower). Merge to MKV for codec safety.
        merged_base = f"{video_title}"
        merged_mkv = Path(f"{merged_base}.mkv")
        merged_mp4 = Path(f"{merged_base}.mp4")

        input_path = None
        for candidate in (merged_mp4, merged_mkv):
            if candidate.exists():
                input_path = candidate
                break

        if input_path is None:
            print("Downloading video (bestvideo+bestaudio)...")
            fmt = (
                "bestvideo[height>=1080]+bestaudio/"
                "bestvideo[height=720]+bestaudio/"
                "best"
            )
            try:
                subprocess.run(
                    [
                        "yt-dlp",
                        "-f", fmt,
                        "--merge-output-format", "mkv",
                        "--output", f"{merged_base}.%(ext)s",
                        "--no-playlist",
                        url,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("Error: Failed to download/merge video")
                return

            if merged_mkv.exists():
                input_path = merged_mkv
            elif merged_mp4.exists():
                input_path = merged_mp4
            else:
                # Fallback search
                for ext in (".mp4", ".mkv", ".webm", ".m4v"):
                    c = Path(f"{merged_base}{ext}")
                    if c.exists():
                        input_path = c
                        break
                if input_path is None:
                    print("Error: Download completed but merged file not found")
                    return
        else:
            print(f"Using existing downloaded file: {input_path.name}")

        # Build subtitle style (ASS keys). libass renders SRT → ASS automatically.
        subs_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H{_color_to_hex(font_color)}&,"
            f"OutlineColour=&H{_color_to_hex(outline_color)}&,Outline=2"
        )
        # Prepare SRT path for libavfilter parser
        subs_path_for_filter = Path(shifted_srt).as_posix().replace("'", r"\'")

        # Filtergraph:
        # 1) scale: cap height at 720 (no upscaling). Width auto, even, keep AR
        #    Using -2 chooses closest even width; see ffmpeg scaling docs/examples.
        # 2) setsar=1 to normalize sample aspect ratio (avoids odd stretches)
        # 3) subtitles (libass) to burn SRT
        vf_chain = (
            "scale=-2:'min(720,ih)':flags=lanczos,"
            "setsar=1,"
            f"subtitles=filename='{subs_path_for_filter}':force_style='{subs_style}'"
        )

        # Output file
        output_video = f"{video_title}_with_subtitles.mp4"
        print(f"Generating video with subtitles: {output_video}")

        # Encode H.264 + AAC. Map first video and optional audio (no index → any audio).
        # If input has no audio, ffmpeg will just omit it (no b:a warning thanks to conditional map).
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner", "-loglevel", "warning",
                    "-y",
                    "-i", str(input_path),
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-vf", vf_chain,
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ac", "2", "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-shortest",
                    output_video,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("Error: Failed to generate video with subtitles")
            return

        print(f"✓ Video with subtitles generated successfully: {output_video}")
        print(f"Output path: {output_path / output_video}")

    finally:
        os.chdir(original_cwd)


def _color_to_hex(color_name: str) -> str:
    """Convert color name to BGR hex for ffmpeg subtitles"""
    color_map = {
        "white": "FFFFFF",
        "black": "000000", 
        "red": "0000FF",
        "green": "00FF00",
        "blue": "FF0000",
        "yellow": "00FFFF",
        "cyan": "FFFF00",
        "magenta": "FF00FF",
    }
    return color_map.get(color_name.lower(), "FFFFFF")


if __name__ == "__main__":
    app()
