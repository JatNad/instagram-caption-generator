"""
    @author     harsh-dhamecha
    @create date 2024-05-29 22:44:17
    @modify date 2024-06-01 14:56:23
    @desc       An app file for FastAPI
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
import openai
import streamlit as st
from contextlib import asynccontextmanager

# Set up the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

app = FastAPI()

class CaptionRequest(BaseModel):
    description: str
    n_captions: int
    caption_style: str
    caption_length: str
    include_emojis: bool
    include_hashtags: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    global blip_processor, blip_model
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")
    yield
    # No cleanup necessary

app.router.lifespan_context = lifespan

@app.post("/describe_images/")
async def describe_images(files: list[UploadFile] = File(...)):
    try:
        images = []
        for file in files:
            image = Image.open(BytesIO(await file.read()))
            images.append(image)
        descriptions = []
        for img in images:
            inputs = blip_processor(images=img, return_tensors="pt")
            outputs = blip_model.generate(**inputs)
            description = blip_processor.decode(outputs[0], skip_special_tokens=True)
            descriptions.append(description)
        combined_description = " ".join(descriptions)
        return JSONResponse(content={"description": combined_description})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.post("/generate_captions/")
async def generate_captions(request: CaptionRequest):
    try:
        emoji_text = 'Include relevant emojis.' if request.include_emojis else 'Do not include emojis.'
        hashtag_text = 'Include relevant hashtags.' if request.include_hashtags else 'Do not include hashtags.'
        prompt = (
            f"Generate {request.n_captions} {request.caption_style} Instagram captions for these images: {request.description}. "
            f"The caption should be {request.caption_length}. "
            f"{emoji_text} {hashtag_text}"
        )

        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7
        )

        captions = [choice.text.strip() for choice in response.choices]
        return JSONResponse(content={"captions": captions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating captions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
