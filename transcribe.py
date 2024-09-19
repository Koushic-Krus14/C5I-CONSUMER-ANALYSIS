import os
from pydub import AudioSegment
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from moviepy.editor import VideoFileClip  # for extracting audio from video

# Initialize tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def convert_to_wav(file_path):
    """
    Converts an audio file (MP3 or video) to WAV format.
    
    :param file_path: Path to the audio or video file
    :return: Path to the converted WAV file
    """
    # Check if the file is a video (e.g., .mp4)
    if file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        audio_clip = extract_audio_from_video(file_path)
        wav_file_path = file_path.replace(file_path.split('.')[-1], 'wav')
        audio_clip.write_audiofile(wav_file_path, codec='pcm_s16le')
        print(wav_file_path)
        return wav_file_path
    
    # If file is MP3, convert it to WAV
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        wav_file_path = file_path.replace('.mp3', '.wav')
        audio.export(wav_file_path, format="wav")
        return wav_file_path
    
    # Return the file path if it's already in WAV format
    return file_path

def extract_audio_from_video(file_path):
    """
    Extracts audio from a video file.
    
    :param file_path: Path to the video file
    :return: AudioClip object with the audio extracted
    """
    video_clip = VideoFileClip(file_path)
    audio_clip = video_clip.audio
    return audio_clip

def transcribe_audio(file_path):
    """
    Transcribes an audio or video file using a pretrained model.
    
    :param file_path: Path to the audio or video file
    :return: Transcription text or error message
    """
    # Convert to WAV (for both audio and video files)
    wav_file_path = convert_to_wav(file_path)

    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_file_path)
        
        # Resample to 16kHz (required by Wav2Vec2)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Tokenize and transcribe
        input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = logits.argmax(dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        print(transcription)
        return transcription
    
    except Exception as e:
        return f"Error: {str(e)}"

# Now this code can handle MP3, WAV, MP4, AVI, MOV, and MKV files
