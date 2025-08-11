import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def is_repetitive_text(text: str, min_repeat_length: int = 3, max_repetitions: int = 5) -> bool:
    """
    Check if text contains repetitive patterns that indicate transcription errors
    """
    text = text.strip()
    if len(text) < min_repeat_length * 2:
        return False

    for pattern_length in range(min_repeat_length, min(len(text) // 2, 10) + 1):
        pattern = text[:pattern_length]
        count = 0
        pos = 0
        while pos + pattern_length <= len(text):
            if text[pos:pos + pattern_length] == pattern:
                count += 1
                pos += pattern_length
            else:
                break

        if count > max_repetitions:
            return True

    return False

def get_context_segments(subs, current_index, context_window=2):
    """
    Get context segments around current segment
    
    Args:
        subs: List of subtitle segments
        current_index: Index of current segment
        context_window: Number of segments before and after to include
    
    Returns:
        Tuple of (previous_segments, current_segment, next_segments)
    """
    start_idx = max(0, current_index - context_window)
    end_idx = min(len(subs), current_index + context_window + 1)
    
    previous_segments = [subs[i].text for i in range(start_idx, current_index)]
    current_segment = subs[current_index].text
    next_segments = [subs[i].text for i in range(current_index + 1, end_idx)]
    
    return previous_segments, current_segment, next_segments

def translate_zh_with_context(previous_segments, current_segment, next_segments, target_language="en"):
    """
    Translate Chinese text with context from surrounding segments
    
    Args:
        previous_segments: List of previous segments for context
        current_segment: Current segment to translate
        next_segments: List of next segments for context
        target_language: Target language code
    
    Returns:
        Translation of current segment
    """
    language_map = {
        "en": "English",
        "pl": "Polish"
    }
    
    target_lang_name = language_map.get(target_language, "English")
    
    # Build context string
    context_parts = []
    
    if previous_segments:
        context_parts.append("PREVIOUS SEGMENTS:")
        for i, seg in enumerate(previous_segments):
            context_parts.append(f"[{i-len(previous_segments)+1}] {seg}")
    
    context_parts.append("CURRENT SEGMENT TO TRANSLATE:")
    context_parts.append(f"[0] {current_segment}")
    
    if next_segments:
        context_parts.append("NEXT SEGMENTS:")
        for i, seg in enumerate(next_segments):
            context_parts.append(f"[{i+1}] {seg}")
    
    context_text = "\n".join(context_parts)
    
    system_prompt = f"""You are an expert Chinese-to-{target_lang_name} subtitle translator. You will receive context from surrounding segments to help with translation continuity.

TRANSLATION RULES:
- Translate ONLY the CURRENT SEGMENT (marked as [0])
- Use the context segments to understand the flow and maintain continuity
- Make the translation grammatically complete and natural in {target_lang_name}
- Complete sentence fragments naturally while preserving original meaning
- Adapt Chinese formal/bureaucratic language to equivalent {target_lang_name} expressions
- Ensure the translation flows well with the surrounding context

OUTPUT: Provide only the translated text of the current segment [0], no explanations or segment numbers."""

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Here are the subtitle segments with context:\n\n{context_text}\n\nTranslate the current segment [0] to {target_lang_name}:"
                }
            ],
            temperature=0.2,
            max_tokens=150,
        )
        
        result = completion.choices[0].message.content.strip()
        print(result)
        return result
        
    except Exception as e:
        raise Exception(f"Translation API error: {e}")

def translate_batch_with_context(segments_with_context, target_language="en", batch_size=5):
    """
    Translate multiple Chinese segments with context in a single API call
    
    Args:
        segments_with_context: List of tuples (index, previous_segments, current_segment, next_segments)
        target_language: Target language code
        batch_size: Number of segments to process in one batch
    
    Returns:
        Dictionary mapping segment index to translation
    """
    language_map = {
        "en": "English",
        "pl": "Polish"
    }
    
    target_lang_name = language_map.get(target_language, "English")
    
    # Build batch prompt
    batch_parts = []
    batch_parts.append(f"You are an expert Chinese-to-{target_lang_name} subtitle translator. You will receive multiple segments with their context to translate efficiently.")
    batch_parts.append("")
    batch_parts.append("TRANSLATION RULES:")
    batch_parts.append("- Translate ONLY the CURRENT SEGMENTS (marked with →)")
    batch_parts.append("- Use context segments to understand flow and maintain continuity")
    batch_parts.append(f"- Make translations grammatically complete and natural in {target_lang_name}")
    batch_parts.append("- Complete sentence fragments naturally while preserving original meaning")
    batch_parts.append(f"- Adapt Chinese formal/bureaucratic language to equivalent {target_lang_name} expressions")
    batch_parts.append("- Ensure translations flow well with surrounding context")
    batch_parts.append("")
    batch_parts.append("SEGMENTS TO TRANSLATE:")
    batch_parts.append("")
    
    # Add each segment with its context
    for i, (seg_idx, previous_segments, current_segment, next_segments) in enumerate(segments_with_context):
        batch_parts.append(f"SEGMENT {i+1} (Index: {seg_idx}):")
        
        if previous_segments:
            batch_parts.append("Previous context:")
            for j, seg in enumerate(previous_segments):
                batch_parts.append(f"  [{j-len(previous_segments)+1}] {seg}")
        
        batch_parts.append(f"→ TRANSLATE THIS: {current_segment}")
        
        if next_segments:
            batch_parts.append("Next context:")
            for j, seg in enumerate(next_segments):
                batch_parts.append(f"  [{j+1}] {seg}")
        
        batch_parts.append("")
    
    batch_parts.append(f"OUTPUT FORMAT: Provide translations in order, one per line, format: 'SEGMENT X: [translation]' where X is the segment number (1-{len(segments_with_context)})")
    
    batch_text = "\n".join(batch_parts)
    
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {
                    "role": "user",
                    "content": batch_text
                }
            ],
            temperature=0.2,
            max_tokens=800,
        )
        
        result = completion.choices[0].message.content.strip()
        print(f"Batch translation result:\n{result}")
        
        # Parse batch results
        translations = {}
        lines = result.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('SEGMENT ') and ':' in line:
                try:
                    # Extract segment number and translation
                    segment_part, translation = line.split(':', 1)
                    segment_num = int(segment_part.replace('SEGMENT', '').strip())
                    
                    if 1 <= segment_num <= len(segments_with_context):
                        original_idx = segments_with_context[segment_num - 1][0]
                        translations[original_idx] = translation.strip()
                except (ValueError, IndexError):
                    continue
        
        return translations
        
    except Exception as e:
        raise Exception(f"Batch translation API error: {e}")

def translate_srt_file(input_srt: str, output_srt: str, language: str = "en", context_window: int = 3, batch_size: int = 5):
    """
    Translate SRT file from Chinese to target language with batch processing and context awareness
    
    Args:
        input_srt: Path to input SRT file in Chinese
        output_srt: Path to output SRT file in target language
        language: Target language code
        context_window: Number of segments before/after to use as context
        batch_size: Number of segments to process in each batch (default: 5)
    """
    import pysrt
    from rich.progress import track

    try:
        subs = pysrt.open(input_srt, encoding='utf-8')
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return

    skipped_count = 0
    total_segments = len(subs)

    print(f"Starting batch translation of {total_segments} segments (batch size: {batch_size}, context window: {context_window})...")

    # Validate language
    if language not in ["en", "pl"]:
        print(f"Unsupported language: {language}. Defaulting to English translation.")
        language = "en"

    # Process segments in batches
    for batch_start in track(range(0, total_segments, batch_size), description="Processing batches..."):
        batch_end = min(batch_start + batch_size, total_segments)
        
        # Prepare batch segments
        batch_segments = []
        skip_indices = set()
        
        for i in range(batch_start, batch_end):
            sub = subs[i]
            
            # Check if text is repetitive (transcription error)
            if is_repetitive_text(sub.text):
                print(f"\nSkipping repetitive text in segment {i}: {sub.text[:50]}...")
                sub.text = "[Error: Repetitive text detected]"
                skipped_count += 1
                skip_indices.add(i)
            else:
                # Get context segments
                previous_segments, current_segment, next_segments = get_context_segments(subs, i, context_window)
                batch_segments.append((i, previous_segments, current_segment, next_segments))
        
        # Translate batch if there are segments to process
        if batch_segments:
            try:
                translations = translate_batch_with_context(batch_segments, language, len(batch_segments))
                
                # Apply translations
                for seg_idx, translation in translations.items():
                    if seg_idx < len(subs):
                        subs[seg_idx].text = translation
                
                # Handle missing translations (fallback to individual translation)
                for seg_idx, previous_segments, current_segment, next_segments in batch_segments:
                    if seg_idx not in translations:
                        print(f"\nFallback: Individual translation for segment {seg_idx}")
                        try:
                            translated = translate_zh_with_context(previous_segments, current_segment, next_segments, language)
                            subs[seg_idx].text = translated
                        except Exception as e:
                            print(f"Individual translation error for segment {seg_idx}: {e}")
                            subs[seg_idx].text = f"[Translation error: {current_segment}]"
                            
            except Exception as e:
                print(f"\nBatch translation error: {e}")
                print("Falling back to individual translation for this batch...")
                
                # Fallback to individual translation for the entire batch
                for seg_idx, previous_segments, current_segment, next_segments in batch_segments:
                    try:
                        translated = translate_zh_with_context(previous_segments, current_segment, next_segments, language)
                        subs[seg_idx].text = translated
                    except Exception as e:
                        print(f"Individual translation error for segment {seg_idx}: {e}")
                        subs[seg_idx].text = f"[Translation error: {current_segment}]"

    print("\nTranslation completed!")
    print(f"Skipped {skipped_count} segments with repetitive text")

    try:
        subs.save(output_srt, encoding='utf-8')
        print(f"Saved translated SRT to: {output_srt}")
    except Exception as e:
        print(f"Error saving SRT file: {e}")