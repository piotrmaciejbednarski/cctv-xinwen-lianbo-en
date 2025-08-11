import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def is_repetitive_text(text: str, min_repeat_length: int = 3, max_repetitions: int = 5) -> bool:
    """
    Check if text contains repetitive patterns that indicate transcription errors

    Args:
        text: Text to check for repetitive patterns
        min_repeat_length: Minimum length of pattern to check for repetition
        max_repetitions: Maximum allowed repetitions before marking as error

    Returns:
        True if text contains repetitive patterns, False otherwise
    """
    text = text.strip()
    if len(text) < min_repeat_length * 2:
        return False

    # Check for patterns of different lengths
    for pattern_length in range(min_repeat_length, min(len(text) // 2, 10) + 1):
        pattern = text[:pattern_length]

        # Count how many times this pattern repeats at the beginning
        count = 0
        pos = 0
        while pos + pattern_length <= len(text):
            if text[pos:pos + pattern_length] == pattern:
                count += 1
                pos += pattern_length
            else:
                break

        # If pattern repeats more than max_repetitions times, it's likely an error
        if count > max_repetitions:
            return True

    return False

def translate_zh_to_en(text_zh: str) -> str:
    """
    Translate Chinese text to English using Google Gemini API

    Args:
        text_zh: Chinese text to translate

    Returns:
        English translation
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Translate into English (without abbreviations): {text_zh}",
        config=types.GenerateContentConfig(
            system_instruction="You are a precise and faithful translator from Chinese to English. Your task is to provide a single, direct translation without abbreviations, preserving the original meaning, proper nouns, and numerical values. Do not provide any alternative translations or additional text.",
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return str(response.text)

def translate_zh_to_pl(text_zh: str) -> str:
    """
    Translate Chinese text to Polish using Google Gemini API

    Args:
        text_zh: Chinese text to translate

    Returns:
        Polish translation
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Translate into Polish (without abbreviations): {text_zh}",
        config=types.GenerateContentConfig(
            system_instruction="You are a precise and faithful translator from Chinese to Polish. Your task is to provide a single, direct translation without abbreviations, preserving the original meaning, proper nouns, and numerical values. Do not provide any alternative translations or additional text.",
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return str(response.text)

def translate_srt_file(input_srt: str, output_srt: str, language: str = "en"):
    """
    Translate SRT file from Chinese to English, skipping repetitive text

    Args:
        input_srt: Path to input SRT file in Chinese
        output_srt: Path to output SRT file in English
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

    print(f"Starting translation of {total_segments} segments...")

    for sub in track(subs, description="Translating segments..."):
        # Check if text is repetitive (transcription error)
        if is_repetitive_text(sub.text):
            print(f"\nSkipping repetitive text: {sub.text[:50]}...")
            sub.text = "[Error: Repetitive text detected]"
            skipped_count += 1
        else:
            try:
                # Translate text based on specified language
                translated = ""
                if language == "en":
                    translated = translate_zh_to_en(sub.text)
                if language == "pl":
                    translated = translate_zh_to_pl(sub.text)
                if language not in ["en", "pl"]:
                    print(f"Unsupported language: {language}. Defaulting to English translation.")
                    translated = translate_zh_to_en(sub.text)
                sub.text = translated
            except Exception as e:
                print(f"\nTranslation error: {e}")
                sub.text = f"[Translation error: {sub.text}]"

    print("\nTranslation completed!")
    print(f"Skipped {skipped_count} segments with repetitive text")

    try:
        subs.save(output_srt, encoding='utf-8')
        print(f"Saved translated SRT to: {output_srt}")
    except Exception as e:
        print(f"Error saving SRT file: {e}")
