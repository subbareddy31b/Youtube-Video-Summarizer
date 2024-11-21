import os
from yt_dlp import YoutubeDL
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize_text_hf(text):
    max_input_length = 1024
    tokens = tokenizer.encode(text, truncation=True, max_length=max_input_length, return_tensors="pt")
    summary_ids = model.generate(tokens, max_length=130, min_length=30, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def transcribe_audio(audio_file, model_type="tiny"):
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_file)
    return result["text"]

def download_audio(youtube_url, output_path="downloads"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        return f"{output_path}/{info_dict['title']}.mp3"
def clean_text(text):
    return ''.join([char if ord(char) < 128 else ' ' for char in text])

youtube_url = "https://www.youtube.com/watch?v=qNxrPri1V0I"
audio_file = download_audio(youtube_url)
print(f"Audio file saved at: {audio_file}")
transcription = transcribe_audio(audio_file, model_type="base")
print("Transcription:\n", transcription)
text = clean_text(transcription)
summary = summarize_text_hf(text)
print("Summary:\n", summary)
