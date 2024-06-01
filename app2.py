'''
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-05-29 22:44:17
 	@modify date 2024-05-29 22:44:26
 	@desc        [description]
 '''

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch
from io import BytesIO

app = FastAPI()

class CaptionRequest(BaseModel):
    description: str
    n_captions: int
    caption_style: str
    caption_length: str
    include_emojis: bool
    include_hashtags: bool

@app.on_event("startup")
async def load_models():
    global blip_processor, blip_model, gpt2_tokenizer, gpt2_model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    blip_model.eval()
    gpt2_model.eval()

@app.post("/describe_images/")
async def describe_images(files: list[UploadFile] = File(...)):
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

@app.post("/generate_captions/")
async def generate_captions(request: CaptionRequest):
    prompt = (
        f"Generate {request.n_captions} {request.caption_style} Instagram captions for these images: {request.description}. "
        f"The caption should be {request.caption_length}. "
        f"{'Include relevant emojis.' if request.include_emojis else ''} "
        f"{'Include relevant hashtags.' if request.include_hashtags else ''}"
    )
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=request.n_captions,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    captions = [gpt2_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return JSONResponse(content={"captions": captions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
