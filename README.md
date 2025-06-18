# Video Subtitle Generation and Translation Pipeline

This tool processes video files to generate subtitles and translate them to Japanese using Azure OpenAI.

## Prerequisites

1. **FFmpeg**: Install ffmpeg for audio extraction

   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **Noto Sans JP Font**: Install Japanese font for proper subtitle rendering

   **Ubuntu/Debian:**

   ```bash
   sudo apt install fonts-noto-cjk
   ```

   **macOS:**

   ```bash
   brew install --cask font-noto-sans-cjk-jp
   ```

   **Manual Installation (any OS):**
   1. Download Noto Sans JP from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+JP)
   2. Extract the downloaded ZIP file
   3. Install the font files:
      - **Windows**: Right-click `.ttf` files and select "Install"
      - **macOS**: Double-click `.ttf` files and click "Install Font"
      - **Linux**: Copy `.ttf` files to `~/.local/share/fonts/` or `/usr/share/fonts/`

3. **Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Azure OpenAI Setup** (for translation only):
   - Copy the example environment file:

     ```bash
     cp .env.example .env
     ```

   - Edit `.env` file with your Azure OpenAI credentials:

     ```bash
     AZURE_OPENAI_API_KEY=your_api_key
     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
     ```

## Usage

Process a video file to generate both original and Japanese subtitles:

```bash
python process_video.py /path/to/your/video.mp4
```

The script uses local Whisper for transcription and Azure OpenAI for translation.

## Output Files

For input video `example.mp4`, the script generates:

- `example_subtitles.srt` - Original transcribed subtitles (using local Whisper)
- `example_japanese.srt` - Japanese translated subtitles (using Azure OpenAI)
- `example_japanese_subs.mp4` - Video with embedded Japanese subtitles

## Features

- Extracts audio from any video format supported by FFmpeg
- Transcribes audio using local OpenAI Whisper (medium model)
- Translates subtitles to Japanese using GPT-4
- Embeds Japanese subtitles back into the original video
- Uses Japanese font (Noto Sans JP) for proper character display
- Preserves SRT formatting and timing
- Automatic cleanup of temporary files
- Progress tracking and error handling
- Smart file checking - asks before recreating existing files
