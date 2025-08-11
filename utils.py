import re
from typing import List, Dict

def to_srt_timestamp(t):
    """Convert time in seconds (float) to SRT timestamp format: HH:MM:SS,mmm"""
    if t is None:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    total_seconds = int(t)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def normalize_segments(segments):
    """
    Accepts:
      - [{"start": s, "end": e, "text": "..."}]  or
      - [[start, end, "text"], ...]
    Returns list of dicts with seconds: [{"start": ..., "end": ..., "text": "..."}]
    Auto-detects ms vs s.
    """
    if not segments:
        return []

    def pick(d, *keys, default=0):
        for k in keys:
            if d.get(k) is not None:
                return d.get(k)
        return default

    if isinstance(segments[0], dict):
        max_time = max(float(pick(seg, "end", "end_time", "ts_end", default=0) or 0) for seg in segments)
        factor = 0.001 if max_time > 10_000 else 1.0
        out = []
        for seg in segments:
            s = float(pick(seg, "start", "start_time", "ts_start", default=0) or 0) * factor
            e = float(pick(seg, "end", "end_time", "ts_end", default=s) or s) * factor
            if e < s:
                e = s
            text = (seg.get("text") or "").strip()
            speaker = seg.get("speaker") or seg.get("spk")
            if speaker:
                text = f"{speaker}: {text}"
            out.append({"start": s, "end": e, "text": text})
        return out

    if isinstance(segments[0], (list, tuple)) and len(segments[0]) >= 3:
        max_time = max(float(s[1]) for s in segments if len(s) >= 2)
        factor = 0.001 if max_time > 10_000 else 1.0
        out = []
        for s in segments:
            start = float(s[0]) * factor
            end = float(s[1]) * factor
            if end < start:
                end = start
            text = str(s[2]).strip() if len(s) > 2 else ""
            out.append({"start": start, "end": end, "text": text})
        return out

    return []

# Heuristics (~37–44 characters/line, 2 lines, min ~0.6–1.0 s, max ~7 s)
DEFAULT_MAX_LINE = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DUR = 0.6
DEFAULT_MAX_DUR = 7.0
DEFAULT_MAX_CPS = 16

_CJK_PUNCT = "。！？；，、：．．｡｡,.!?;:"
_SPACE_SPLIT_RE = re.compile(r"\s+")

def _visible_len(s: str) -> int:
    return len(s)

def _split_long_text_two_lines(text: str, max_len: int) -> List[str]:
    """
    Break text into ≤2 lines, preferring split near the middle:
    - for languages with spaces: split by space
    - for CJK: split by character, avoid breaking right before/after punctuation
    """
    t = text.strip()
    if _visible_len(t) <= max_len:
        return [t]

    # Try to split into 2 lines
    target = _visible_len(t) // 2

    # 1) Try: break by space closest to target
    if " " in t:
        spaces = [m.start() for m in re.finditer(r"\s", t)]
        cut = min(spaces, key=lambda i: abs(i - target)) if spaces else target
    else:
        # 2) CJK fallback: cut closest to target, but not on punctuation
        cut = target
        # move left until not a punctuation character
        while cut > 1 and t[cut-1] in _CJK_PUNCT:
            cut -= 1

    left = t[:cut].strip()
    right = t[cut:].strip()

    # If any line is still too long, hard wrap
    def hard_wrap(s):
        lines = []
        cur = s
        while _visible_len(cur) > max_len:
            lines.append(cur[:max_len])
            cur = cur[max_len:]
        if cur:
            lines.append(cur)
        return lines

    lines = []
    for part in (left, right):
        if _visible_len(part) <= max_len:
            lines.append(part)
        else:
            lines.extend(hard_wrap(part))

    # Return max 2 lines (rest joined into 2nd line — rare in practice)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:]).strip()]
        if _visible_len(lines[1]) > max_len:
            lines[1] = lines[1][:max_len]
    return lines[:2]

def _cps(text: str, start: float, end: float) -> float:
    dur = max(1e-6, end - start)
    return _visible_len(text) / dur

def merge_short_segments(segments: List[Dict], min_dur=DEFAULT_MIN_DUR, max_dur=DEFAULT_MAX_DUR, max_cps=DEFAULT_MAX_CPS):
    """
    Merge too short subtitles with the next one if:
      - gap < 0.25 s
      - after merging dur <= max_dur
      - cps after merging does not exceed max_cps
    """
    if not segments:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        # ensure end >= start
        cur["end"] = max(cur["end"], cur["start"])
        dur = cur["end"] - cur["start"]

        # if short — try merging until it's long enough
        while dur < min_dur and i + 1 < len(segments):
            nxt = segments[i + 1]
            gap = max(0.0, nxt["start"] - cur["end"])
            tentative_text = (cur["text"] + " " + nxt["text"]).strip()
            tentative_end = max(cur["end"], nxt["end"])
            tentative_dur = tentative_end - cur["start"]
            tentative_cps = _cps(tentative_text, cur["start"], tentative_end)
            if gap <= 0.25 and tentative_dur <= max_dur and tentative_cps <= max_cps:
                # merge
                cur["text"] = tentative_text
                cur["end"] = tentative_end
                i += 1
                dur = tentative_dur
            else:
                break

        merged.append(cur)
        i += 1
    return merged

def reflow_lines(segments: List[Dict], max_line_len=DEFAULT_MAX_LINE, max_lines=DEFAULT_MAX_LINES):
    """
    Break lines within subtitle to ≤max_lines, each ≤max_line_len.
    Text is joined with newline characters for SRT.
    """
    out = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            out.append(seg)
            continue
        lines = _split_long_text_two_lines(text, max_line_len)
        # If still > max_lines, join lines 2..N into one line
        if len(lines) > max_lines:
            lines = [lines[0], " ".join(lines[1:]).strip()]
            if _visible_len(lines[1]) > max_line_len:
                lines[1] = lines[1][:max_line_len]
        seg2 = dict(seg)
        seg2["text"] = "\n".join(lines)
        out.append(seg2)
    return out

def clip_long_segments(segments: List[Dict], max_dur=DEFAULT_MAX_DUR):
    """
    If segment has unnaturally long times (> max_dur), shorten to max_dur.
    (Does not change text; safe clip for typical max 7 s guideline.)
    """
    out = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        if e - s > max_dur:
            e = s + max_dur
        out.append({"start": s, "end": e, "text": seg["text"]})
    return out

def improve_readability(segments_norm: List[Dict],
                        max_line_len=DEFAULT_MAX_LINE,
                        max_lines=DEFAULT_MAX_LINES,
                        min_dur=DEFAULT_MIN_DUR,
                        max_dur=DEFAULT_MAX_DUR,
                        max_cps=DEFAULT_MAX_CPS):
    """
    Order:
      1) merge too short segments,
      2) clip too long ones,
      3) break lines.
    """
    segs = merge_short_segments(segments_norm, min_dur=min_dur, max_dur=max_dur, max_cps=max_cps)
    segs = clip_long_segments(segs, max_dur=max_dur)
    segs = reflow_lines(segs, max_line_len=max_line_len, max_lines=max_lines)
    return segs

def write_srt(segments_norm, path="transcription.srt"):
    """Write normalized segments to SRT file"""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments_norm, 1):
            f.write(f"{i}\n{to_srt_timestamp(seg['start'])} --> {to_srt_timestamp(seg['end'])}\n{seg['text']}\n\n")

def write_srt_pretty(segments_norm,
                     path="transcription.srt",
                     max_line_len=DEFAULT_MAX_LINE,
                     max_lines=DEFAULT_MAX_LINES,
                     min_dur=DEFAULT_MIN_DUR,
                     max_dur=DEFAULT_MAX_DUR,
                     max_cps=DEFAULT_MAX_CPS):
    """Like write_srt, but with readability (merge short, break long)."""
    nicer = improve_readability(
        segments_norm,
        max_line_len=max_line_len,
        max_lines=max_lines,
        min_dur=min_dur,
        max_dur=max_dur,
        max_cps=max_cps,
    )
    write_srt(nicer, path)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem compatibility"""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.strip()[:100]
    return filename

def get_video_title(url: str) -> str:
    """Get YouTube video title using yt-dlp"""
    try:
        import subprocess
        result = subprocess.run(
            ["yt-dlp", "--get-title", url],
            capture_output=True,
            text=True,
            check=True
        )
        title = result.stdout.strip()
        return sanitize_filename(title)
    except Exception as e:
        print(f"Warning: Could not get video title: {e}")
        return "unknown_video"


def shift_srt_timestamps(input_srt: str, output_srt: str, offset_seconds: float = 18.0):
    """
    Shift only subtitles that start during intro period (before offset_seconds)
    
    Args:
        input_srt: Path to input SRT file
        output_srt: Path to output SRT file
        offset_seconds: Intro duration in seconds - only subtitles starting before this will be shifted
    """
    import pysrt
    
    try:
        subs = pysrt.open(input_srt, encoding='utf-8')
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return False
    
    shifted_count = 0
    
    # Only shift subtitles that start during intro period
    for sub in subs:
        start_seconds = sub.start.ordinal / 1000.0  # Convert to seconds
        
        # Only shift if subtitle starts during intro (before offset_seconds)
        if start_seconds < offset_seconds:
            # Shift by offset
            start_ms = sub.start.ordinal + int(offset_seconds * 1000)
            end_ms = sub.end.ordinal + int(offset_seconds * 1000)
            
            # Update timestamps
            sub.start = pysrt.SubRipTime.from_ordinal(start_ms)
            sub.end = pysrt.SubRipTime.from_ordinal(end_ms)
            shifted_count += 1
    
    try:
        subs.save(output_srt, encoding='utf-8')
        print(f"Shifted {shifted_count} subtitles that started during intro period")
        print(f"Shifted SRT saved to: {output_srt}")
        return True
    except Exception as e:
        print(f"Error saving shifted SRT file: {e}")
        return False
