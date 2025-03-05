---
title: AI Audio-to-Video Converter
emoji: 🎥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: app.py
---
# AI Audio-to-Video Converter
Convert audio to video with synchronized images using Hugging Face models.

## Features
- **Transcription**: `openai/whisper-large-v3`
- **Image Acquisition**: Stable Diffusion or Google scraping with caching
- **Video Assembly**: MoviePy with FFmpeg
- **Enhancements**: Robust error handling, image caching, temporary file cleanup

## Usage
1. Upload audio (max 50 min).
2. Select video format (9:16, 16:9, or 1:1).
3. Choose image method (Scraper or Generator).
4. Generate and download the video.

## Notes
- Pinned `huggingface_hub==0.20.3` for `diffusers` compatibility.
- Updated `gr.Audio` for Gradio 4.19.2 compatibility.
- Added logging and cleanup for improved debugging and resource management.