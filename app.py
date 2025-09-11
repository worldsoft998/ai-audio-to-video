import gradio as gr
import sys
import os
import logging
import shutil
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import torch
from diffusers import StableDiffusionPipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

# Configuration Constants
SEGMENT_DURATION = 5  # seconds
MAX_AUDIO_DURATION = 3000  # seconds (50 minutes)
IMAGE_CACHE = {}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Debug imports and versions
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path}")

try:
    logger.info("Importing moviepy.editor...")
    from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
    logger.info("Successfully imported moviepy.editor")
except ModuleNotFoundError as e:
    logger.error(f"Failed to import moviepy.editor: {e}")
    raise

try:
    import huggingface_hub
    logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
    from diffusers import StableDiffusionPipeline
    logger.info("Successfully imported diffusers.StableDiffusionPipeline")
except ImportError as e:
    logger.error(f"Failed to import diffusers: {e}")
    raise

# Initialize pipelines
logger.info("Initializing Whisper model...")
whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
logger.info("Whisper model initialized")

logger.info("Initializing Stable Diffusion model...")
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
stable_diffusion = stable_diffusion.to("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Stable Diffusion model initialized")

logger.info("Initializing GPT-2 model...")
text_gen_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
logger.info("GPT-2 model initialized")

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Text: {text}\n\nImage prompt:"
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Helper Functions
def get_resolution(video_format):
    return {"9:16": (720, 1280), "16:9": (1920, 1080), "1:1": (1080, 1080)}.get(video_format, (1920, 1080))

def resize_image_to_format(image_path, resolution):
    try:
        img = Image.open(image_path)
        img_resized = img.resize(resolution, Image.Resampling.LANCZOS)
        base, ext = os.path.splitext(image_path)
        resized_path = f"{base}_resized{ext}"
        img_resized.save(resized_path)
        logger.info(f"Resized image saved to {resized_path}")
        return resized_path
    except Exception as e:
        logger.error(f"Resize failed for {image_path}: {e}")
        return image_path

def transcribe_audio(audio_file_path):
    try:
        result = whisper_asr(audio_file_path, return_timestamps=True)
        logger.info(f"Transcription completed for {audio_file_path}")
        return result['chunks']
    except Exception as e:
        logger.error(f"Transcription failed for {audio_file_path}: {e}")
        raise RuntimeError(f"Transcription failed: {e}")

def segment_audio_and_transcription(chunks, audio_file_path):
    try:
        audio = AudioFileClip(audio_file_path)
        total_duration = audio.duration
        num_segments = int(total_duration // SEGMENT_DURATION)
        segments = []
        for i in range(num_segments):
            start = i * SEGMENT_DURATION
            end = min((i + 1) * SEGMENT_DURATION, total_duration)
            segment_chunks = [chunk for chunk in chunks if start <= chunk['timestamp'][0] < end]
            segment_text = " ".join([chunk['text'] for chunk in segment_chunks])
            segments.append({"start": start, "end": end, "text": segment_text})
        if total_duration % SEGMENT_DURATION:
            start = num_segments * SEGMENT_DURATION
            end = total_duration
            segment_chunks = [chunk for chunk in chunks if start <= chunk['timestamp'][0] < end]
            segment_text = " ".join([chunk['text'] for chunk in segment_chunks])
            segments.append({"start": start, "end": end, "text": segment_text})
        logger.info(f"Segmented {audio_file_path} into {len(segments)} segments")
        return segments, total_duration
    except Exception as e:
        logger.error(f"Segmentation failed for {audio_file_path}: {e}")
        raise RuntimeError(f"Segmentation failed: {e}")

def generate_enhanced_image_prompt(text):
    try:
        full_response = llm_chain.run(text=text)
        prompt = full_response.split("\n\nImage prompt:")[1].strip() if "\n\nImage prompt:" in full_response else full_response.strip()
        logger.info(f"Generated prompt: {prompt}")
        return prompt
    except Exception as e:
        logger.error(f"Prompt generation failed for text '{text}': {e}")
        return "A detailed, colorful scene based on audio content"

def scrape_images(query, max_images=1):
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(f"https://www.google.com/search?tbm=isch&q={query}")
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        img_urls = [img.get("src") for img in soup.find_all("img") if img.get("src", "").startswith("http")][:max_images]
        image_paths = []
        os.makedirs("scraped_images", exist_ok=True)
        for i, url in enumerate(img_urls):
            response = requests.get(url, timeout=10)
            path = f"scraped_images/img_{i}_{int(time.time())}.jpg"
            with open(path, "wb") as f:
                f.write(response.content)
            image_paths.append(path)
        logger.info(f"Scraped {len(image_paths)} images for query '{query}'")
        return image_paths
    except Exception as e:
        logger.error(f"Scraping failed for query '{query}': {e}")
        return []
    finally:
        if driver:
            driver.quit()

def generate_image(prompt):
    try:
        image = stable_diffusion(prompt).images[0]
        os.makedirs("generated_images", exist_ok=True)
        path = f"generated_images/img_{int(time.time())}.png"
        image.save(path)
        logger.info(f"Generated image saved to {path}")
        return path
    except Exception as e:
        logger.error(f"Image generation failed for prompt '{prompt}': {e}")
        return None

def create_text_image(text, resolution):
    try:
        img = Image.new('RGB', resolution, color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        d.text((10, 10), text, font=font, fill=(255, 255, 0))
        path = f"text_images/text_{int(time.time())}.png"
        os.makedirs("text_images", exist_ok=True)
        img.save(path)
        logger.info(f"Text image saved to {path}")
        return path
    except Exception as e:
        logger.error(f"Text image creation failed: {e}")
        return None

def get_cached_image(text, image_method, resolution):
    cache_key = (text, image_method, tuple(resolution))
    if cache_key in IMAGE_CACHE:
        logger.info(f"Retrieved cached image for key {cache_key}")
        return IMAGE_CACHE[cache_key]
    prompt = generate_enhanced_image_prompt(text)
    if image_method == "Image Scraper":
        paths = scrape_images(prompt)
        img_path = paths[0] if paths else None
        if img_path is None:
            logger.info(f"No images scraped for prompt: {prompt}, falling back to generation")
            img_path = generate_image(prompt)
    else:
        img_path = generate_image(prompt)
        if img_path is None:
            logger.info(f"Generation failed for prompt: {prompt}, falling back to scraping")
            paths = scrape_images(prompt)
            img_path = paths[0] if paths else None

    if img_path is None:
        logger.info("No image found, using text image")
        img_path = create_text_image(text, resolution)
    
    resized_path = resize_image_to_format(img_path, resolution)
    IMAGE_CACHE[cache_key] = resized_path
    logger.info(f"Cached image for key {cache_key}")
    return resized_path

def process_segment(segment, video_format, image_method):
    text = segment["text"] if segment["text"] else "generic audio content"
    resolution = get_resolution(video_format)
    resized_path = get_cached_image(text, image_method, resolution)
    return (resized_path, segment["start"], segment["end"])

def process_segments_concurrently(segments, video_format, image_method):
    image_segments = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_segment, seg, video_format, image_method): seg for seg in segments}
        for future in as_completed(futures):
            result = future.result()
            if result:
                image_segments.append(result)
    logger.info(f"Processed {len(image_segments)} segments")
    return image_segments

def create_video_with_segments(image_segments, audio_file_path, video_format):
    try:
        resolution = get_resolution(video_format)
        clips = [
            ImageClip(img_path).set_duration(end - start).set_start(start).resize(resolution)
            for img_path, start, end in image_segments
        ]
        video = CompositeVideoClip(clips, size=resolution)
        audio = AudioFileClip(audio_file_path)
        video = video.set_audio(audio)
        output_path = "final_video.mp4"
        video.write_videofile(output_path, fps=24)
        logger.info(f"Video created at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Video assembly failed: {e}")
        raise RuntimeError(f"Video assembly failed: {e}")

def cleanup_temp_files():
    for dir_name in ["scraped_images", "generated_images", "text_images"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            logger.info(f"Deleted temporary directory: {dir_name}")

def process_audio_to_video(audio_file, video_format, image_method):
    try:
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return "Error: Audio file not found."
        logger.info(f"Starting transcription for {audio_file}")
        chunks = transcribe_audio(audio_file)
        logger.info(f"Starting segmentation for {audio_file}")
        segments, total_duration = segment_audio_and_transcription(chunks, audio_file)
        if total_duration > MAX_AUDIO_DURATION:
            logger.error(f"Audio duration {total_duration} exceeds maximum {MAX_AUDIO_DURATION}")
            return "Error: Audio exceeds 50 minutes."
        logger.info("Processing segments...")
        image_segments = process_segments_concurrently(segments, video_format, image_method)
        logger.info("Creating video...")
        final_video = create_video_with_segments(image_segments, audio_file, video_format)
        logger.info("Video generation completed successfully")
        return final_video
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return f"Error: {str(e)}"
    finally:
        cleanup_temp_files()

# Gradio Interface
audio_input = gr.Audio(type="filepath", label="Upload Audio (Max 50 min)")
format_input = gr.Radio(choices=["9:16", "16:9", "1:1"], label="Video Format", value="16:9")
method_input = gr.Radio(choices=["Image Scraper", "Image Generator"], label="Image Method", value="Image Generator")
video_output = gr.Video(label="Generated Video")

interface = gr.Interface(
    fn=process_audio_to_video,
    inputs=[audio_input, format_input, method_input],
    outputs=video_output,
    title="AI Audio-to-Video Converter",
    description="Convert audio to video with synchronized images using Hugging Face models."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    interface.launch(server_name="0.0.0.0", server_port=8860, share=True, debug=True)
    logger.info("Gradio interface launched successfully")
