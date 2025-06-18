import os
import sys
import subprocess
from pathlib import Path
from openai import AzureOpenAI
from typing import List, Tuple
import tempfile
import whisper
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video file using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', 
            '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error extracting audio: {result.stderr}")
            return False
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error running ffmpeg: {e}")
        return False

def transcribe_audio_to_srt(audio_path: str, srt_path: str) -> bool:
    """Transcribe audio file to SRT format using local Whisper."""
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("medium")
        
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        
        # Convert Whisper output to SRT format
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"].strip()
            
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
        
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(srt_content.strip())
        
        return True
    except (OSError, IOError) as e:
        print(f"Error with file operations during transcription: {e}")
        return False
    except (RuntimeError, ValueError) as e:
        print(f"Error with Whisper model or transcription: {e}")
        return False

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def parse_srt(file_path: str) -> List[Tuple[str, str, str]]:
    """Parse SRT file and return list of (index, timestamp, text) tuples."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except (OSError, IOError) as e:
        print(f"Error reading SRT file {file_path}: {e}")
        return []
    
    blocks = content.strip().split('\n\n')
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text = '\n'.join(lines[2:])
            subtitles.append((index, timestamp, text))
    
    return subtitles

def translate_text(client: AzureOpenAI, text: str) -> str:
    """Translate text to Japanese using Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text to Japanese. Maintain the original meaning and tone. Format the translation with line breaks (\n) at natural points to ensure subtitles fit well in a video window - aim for maximum 2 lines with roughly equal length. Only return the translated text without any additional commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except (openai.OpenAIError, openai.APIError, openai.RateLimitError) as e:
        print(f"Error with OpenAI API during translation: {e}")
        return text
    except (AttributeError, KeyError) as e:
        print(f"Error parsing OpenAI response: {e}")
        return text

def write_srt(subtitles: List[Tuple[str, str, str]], output_path: str):
    """Write subtitles to SRT file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for i, (index, timestamp, text) in enumerate(subtitles):
                file.write(f"{index}\n")
                file.write(f"{timestamp}\n")
                file.write(f"{text}\n")
                if i < len(subtitles) - 1:
                    file.write("\n")
    except (OSError, IOError) as e:
        print(f"Error writing SRT file {output_path}: {e}")
        raise

def translate_srt_file(client: AzureOpenAI, input_srt: str, output_srt: str) -> bool:
    """Translate SRT file to Japanese."""
    try:
        print(f"Reading {input_srt}...")
        subtitles = parse_srt(input_srt)
        
        if not subtitles:
            print("No subtitles found to translate")
            return False
        
        print(f"Translating {len(subtitles)} subtitle entries...")
        translated_subtitles = []
        
        for i, (index, timestamp, text) in enumerate(subtitles):
            print(f"Translating entry {i+1}/{len(subtitles)}...")
            translated_text = translate_text(client, text)
            translated_subtitles.append((index, timestamp, translated_text))
        
        print(f"Writing translated subtitles to {output_srt}...")
        write_srt(translated_subtitles, output_srt)
        return True
    except (OSError, IOError) as e:
        print(f"Error with file operations during SRT translation: {e}")
        return False

def embed_subtitles_in_video(video_path: str, srt_path: str, output_path: str) -> bool:
    """Embed SRT subtitles into video file using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f"subtitles={srt_path}:force_style='FontName=Noto Sans JP,FontSize=20,WrapStyle=0,PlayResX=1280'",
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error embedding subtitles: {result.stderr}")
            return False
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error running ffmpeg for subtitle embedding: {e}")
        return False

def check_existing_files(video_path: str) -> dict:
    """Check which output files already exist and return their status."""
    video_name = Path(video_path).stem
    output_dir = Path(video_path).parent
    
    files = {
        'audio': output_dir / f"{video_name}_audio.wav",
        'original_srt': output_dir / f"{video_name}_subtitles.srt",
        'japanese_srt': output_dir / f"{video_name}_japanese.srt",
        'output_video': output_dir / f"{video_name}_japanese_subs.mp4"
    }
    
    existing = {}
    for key, path in files.items():
        existing[key] = path.exists()
    
    return existing, files

def ask_user_confirmation(step_name: str, file_path: str) -> bool:
    """Ask user if they want to recreate an existing file."""
    while True:
        response = input(f"\n{step_name} already exists: {file_path}\nDo you want to recreate it? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")

def process_video(video_path: str):
    """Main function to process video file and generate translated subtitles."""
    # Validate input
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    # Check existing files
    existing, file_paths = check_existing_files(video_path)
    
    # Setup Azure OpenAI client (only for translation)
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            print("Error: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env file")
            return
            
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )
    except (ValueError, TypeError) as e:
        print(f"Error with Azure OpenAI configuration: {e}")
        return
    
    # Generate output file names
    video_name = Path(video_path).stem
    output_dir = Path(video_path).parent
    
    audio_path = file_paths['audio']
    original_srt = file_paths['original_srt']
    japanese_srt = file_paths['japanese_srt']
    output_video = file_paths['output_video']
    
    # Check if final output already exists
    if existing['output_video']:
        if not ask_user_confirmation("Final video with Japanese subtitles", str(output_video)):
            print("Skipping processing. Final output already exists.")
            return
    
    try:
        # Step 1: Extract audio from video
        skip_audio = False
        if existing['audio']:
            if not ask_user_confirmation("Audio extraction", str(audio_path)):
                skip_audio = True
                print("Using existing audio file...")
        
        if not skip_audio:
            print("Step 1: Extracting audio from video...")
            if not extract_audio_from_video(str(video_path), str(audio_path)):
                print("Failed to extract audio from video")
                return
            print(f"Audio extracted to: {audio_path}")
        
        # Step 2: Transcribe audio to SRT
        skip_transcription = False
        if existing['original_srt']:
            if not ask_user_confirmation("Original subtitles transcription", str(original_srt)):
                skip_transcription = True
                print("Using existing original subtitles...")
        
        if not skip_transcription:
            print("Step 2: Transcribing audio to subtitles...")
            if not transcribe_audio_to_srt(str(audio_path), str(original_srt)):
                print("Failed to transcribe audio")
                return
            print(f"Original subtitles saved to: {original_srt}")
        
        # Step 3: Translate SRT to Japanese
        skip_translation = False
        if existing['japanese_srt']:
            if not ask_user_confirmation("Japanese translation", str(japanese_srt)):
                skip_translation = True
                print("Using existing Japanese subtitles...")
        
        if not skip_translation:
            print("Step 3: Translating subtitles to Japanese...")
            if not translate_srt_file(client, str(original_srt), str(japanese_srt)):
                print("Failed to translate subtitles")
                return
            print(f"Japanese subtitles saved to: {japanese_srt}")
        
        # Step 4: Embed Japanese subtitles into video
        print("Step 4: Embedding Japanese subtitles into video...")
        if not embed_subtitles_in_video(str(video_path), str(japanese_srt), str(output_video)):
            print("Failed to embed subtitles into video")
            return
        print(f"Video with Japanese subtitles saved to: {output_video}")
        
        print("\n✅ Processing completed successfully!")
        print(f"📁 Original subtitles: {original_srt}")
        print(f"🇯🇵 Japanese subtitles: {japanese_srt}")
        print(f"🎬 Video with Japanese subtitles: {output_video}")
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return
    except (OSError, IOError) as e:
        print(f"❌ File system error during processing: {e}")
        return

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py <video_file_path>")
        print("Example: python process_video.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    process_video(video_path)

if __name__ == "__main__":
    main()
