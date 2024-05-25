"""
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-05-25 11:06:48
 	@modify date 2024-05-25 19:09:02
 	@desc        An app file for IG Caption Generator
"""

import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Set the page configuration
st.set_page_config(
    page_title="Instagram Caption Generator",
    page_icon=":camera:",  # Optional, you can use an emoji as an icon
    layout="centered",  # Optional, layout can be "centered" or "wide"
    initial_sidebar_state="auto"  # Optional, can be "auto", "expanded", or "collapsed"
)

# Streamlit app title with color
st.markdown("<h1 style='color: #ff0000;'>Instagram Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #4682b4;'>Upload your images and get creative captions instantly!</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #32cd32;'>Your images are not stored and are completely safe!</p>", unsafe_allow_html=True)

# Set a cache directory and ensure that it exists
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
    return processor, model

@st.cache_resource
def load_openai_model():
    llm = OpenAI(model='gpt-3.5-turbo-instruct')
    return llm

def describe_images(images, processor, model):
    descriptions = []
    for img in images:
        # Generate description using BLIP model
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        descriptions.append(description)
    return " ".join(descriptions)

def get_prompt(n_captions, style, length, emojis, hashtags):
    prompt_template = PromptTemplate(
        input_variables=["image_description", "n_captions", "style", "length", "emojis", "hashtags"],
        template=(
            "Generate {n_captions} {style} Instagram caption for these images: {image_description}. "
            "The caption should be {length}"
            "{emojis} {hashtags}"
        )
    )
    return prompt_template

# Function to generate a caption based on all images
def generate_caption(llm, image_descriptions, n_captions, caption_style, caption_length, emojis, hashtags):
    
    prompt_template = get_prompt(n_captions, caption_style, caption_length, emojis, hashtags)
    
    # Create a LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate caption using the chain
    emoji_text = "Include relevant emojis." if emojis else "Do not include Emojis."
    hashtag_text = f"Include relevant hashtags." if hashtags else "Do not include Hashtags."
    generated_caption = chain.run({
        "image_description": image_descriptions,
        "n_captions": n_captions,
        "style": caption_style.lower(),
        "length": caption_length.lower(),
        "emojis": emoji_text,
        "hashtags": hashtag_text,
    })
    return generated_caption

# Image uploader
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# List to store images
images = []
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        images.append(img)

# Load the models
processor, model = load_blip_model()

# Caption customization options
n_captions = st.selectbox("Select Number of Captions", ["1", "2", "5", "10"], key="n_captions")
caption_style = st.selectbox("Select Caption Style", ["Formal", "Informal", "Humorous", "Inspirational", "Poetic"], key="caption_style")
caption_length = st.selectbox("Select Caption Length", ["Short", "Medium", "Long"], key="caption_length")
emojis = st.checkbox("Include Relevant Emojis", key="emojis")
hashtags = st.checkbox("Include Relevant Hashtags", key="hashtags")

if images:
    generate_button = st.button("Generate Caption")
    if generate_button:
        with st.spinner('Generating captions... Please wait...'):
            image_descriptions = describe_images(images, processor, model)
            llm = load_openai_model()
            caption = generate_caption(llm, image_descriptions, n_captions, caption_style, caption_length, emojis, hashtags)
            st.write("Generated Caption:")
            st.write(caption)
