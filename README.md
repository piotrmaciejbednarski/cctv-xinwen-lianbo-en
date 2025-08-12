# cctv-xinwen-lianbo-en

![Generated video](/generated-video.png)

A Python project for Chinese-to-English translations (SRT subtitles) for each episode of CCTV's "Xinwen Lianbo" (「新闻联播」), a valuable resource for language learners and researchers.

## Models Used

- **Transcription**: FunASR model
  - Specifically uses `AutoModel` for Chinese ASR (Paraformer-zh + VAD + punctuation)
  - Provides sentence-level timestamps and punctuation
- **Translation**: OpenRouter API, compatible with OpenAI API (using DeepSeek V3-0324 model)
  - Supports Chinese to English and Chinese to Polish translation
  - Uses system instructions for precise, faithful translation (context window of 3 segments for better context understanding)
  - Batch processing for efficiency (default batch size of 5 segments), change with `--batch-size` option

Transcription model are runned locally using FunASR, while translation is done via OpenRouter API.

[Deepseek V3-0324 price](https://openrouter.ai/deepseek/deepseek-chat-v3-0324) is $0.18 per 1M input tokens, and $0.72 per 1M output tokens (as of August 2025). For full translation of an episode, it costs around $0.007 to $0.008 per episode.

All models are made by Chinese developers, and are optimized for Chinese language processing.

## Dependencies

- Python 3.8+
- `yt-dlp`, `uv` and `ffmpeg` installed in **global environment**
- `funasr` (transcription using Chinese ASR model)
- `dotenv` (for environment variable management)
- `openai` (translation using OpenRouter API or any compatible OpenAI API)
- `typer`
- `torchaudio`, `torch`
- `pysrt`

## Usage

1. Install the required dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

    ```bash
    uv sync
    ```

2. Set `OPENROUTER_API_KEY` variable (in `.env`) to your OpenRouter API key. You can get it from [OpenRouter](https://openrouter.ai/).

3. Go to [Youtube playlist page of CCTV](https://www.youtube.com/playlist?list=PL0eGJygpmOH5xQuy8fpaOvKrenoCsWrKh) and copy the specific URL of the episode you want to download, then run for example:

    ```bash
    uv run main.py transcribe "YOUTUBE_VIDEO_URL"
    ```

    If you want SRT translation in Polish, you can run:

    ```bash
    uv run main.py transcribe "YOUTUBE_VIDEO_URL" --language pl
    ```

    During the transcription step, there is no verbose output, you can't monitor the progress. Please be patient, as it may take a while depending on the length of the source video.

4. Generate translation for generated transcription:

    ```bash
    uv run main.py translate output/{video_title} --language en
    ```

    Or for Polish translation:

    ```bash
    uv run main.py translate output/{video_title} --language pl
    ```

    This will create `translation.srt` file in the same directory.

5. If you want to generate MP4 video with embedded subtitles, you can run (make sure you have `ffmpeg` installed):

    ```bash
    uv run main.py generate-video "YOUTUBE_VIDEO_URL"
    ```

    Or using custom parameters:

    ```bash
    uv run main.py generate-video "YOUTUBE_VIDEO_URL" --srt transcription.srt --font-size 28 --font-color yellow
    ```

    By default it automatically shifts subtitles by 18 seconds to synchronize with the intro, but you can change this with the `--offset` parameter. During the video generation, please be patient, as it may take a while depending on the length of the source video.

## Output

The output will be saved in the `output/{video_title}` directory, with the following files:

- `transcription.json` - Transcription of the episode in JSON format.
- `transcription.srt` - Transcription of the episode in SRT format.
- `translation.srt` - Translation of the episode in SRT format.

Example `translation.srt` file:

```
8
00:00:36,870 --> 00:00:37,930
Since the 18th National Congress of the Communist Party of China,

9
00:00:38,090 --> 00:00:45,090
the Party Central Committee with Comrade Xi Jinping at its core has prioritized ecological conservation as a fundamental strategy for the sustainable development of the Chinese nation,

10
00:00:46,090 --> 00:00:48,170
undertaking a series of pioneering initiatives,

11
00:00:48,610 --> 00:00:51,135
Significant strides have been made in building a Beautiful China.

12
00:00:51,890 --> 00:00:53,510
China Central Television News launches a special series,

13
00:00:53,610 --> 00:00:55,010
documenting ecological transformations.

14
00:00:55,250 --> 00:00:58,190
Today, we begin at the birthplace of the "Two Mountains" philosophy,

15
00:00:58,370 --> 00:00:59,810
in Anji, Zhejiang Province.
```

## Disclaimer

CCTV clearly states that it holds all copyrights to its programs (including news), and distribution without permission is prohibited. Use this project for educational and research purposes only.

For more information, see the [original CCTV statement](https://news.cctv.com/2017/04/26/ARTI9neH8KQH2RzzhkOjEsBZ170426.shtml).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
