# Youtube-Video-Summarizer

This project is designed to transcribe and summarize YouTube videos automatically. The system downloads the video audio, uses the Whisper model for transcription, and summarizes the transcribed text using the BART model from Hugging Face. Optionally, the retrieval-augmented generation (RAG) method can be added for more sophisticated summarization, though this isn't included in the basic implementation.

## Features

- **Download YouTube Audio**: Extracts audio from YouTube videos.
- **Transcription with Whisper**: Uses the Whisper model for transcribing the audio to text.
- **Summarization with BART**: Summarizes the transcribed text using the BART large model from Hugging Face.

## Dependencies

- `yt-dlp`: For downloading YouTube audio.
- `whisper`: For transcribing audio to text.
- `transformers`: For Hugging Face models such as BART.
- `torch`: PyTorch for running the models.
