'''
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-05-25 11:06:48
 	@modify date 2024-05-25 15:01:59
 	@desc        An app file for IG Caption Generator
 '''

import os
import streamlit as st
from PIL import Image
import openai
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

# Set a cache directory
cache_dir = "./model_cache"

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)

# Streamlit app title
st.title("Instagram Caption Generator")

# Image uploader
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# List to store images
images = []

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        images.append(img)

# Option to enable customization
enable_customization = st.checkbox("Enable Customization", help="Customize your caption for a more personalized touch!")

if enable_customization:
    st.markdown("**Try our customization options to create a unique and tailored caption!**")

    # Caption customization options
    n_captions = st.selectbox("How many Captions you want to generate?", ["1", "2", "5", "10"])
    caption_style = st.selectbox("Select Caption Style", ["Formal", "Informal", "Humorous", "Inspirational"])
    caption_length = st.selectbox("Select Caption Length", ["Short", "Medium", "Long"])
    include_emojis = st.checkbox("Include Emojis")
    custom_hashtags = st.text_input("Add Hashtags (comma-separated)")
    language = st.selectbox("Select Language", ["English", "Spanish", "French", "German"])
else:
    n_captions = 1
    caption_style = "Formal"
    caption_length = "Medium"
    include_emojis = False
    custom_hashtags = ""
    language = "English"

# Function to generate a caption based on all images
def generate_caption(images, n_captions, style, length, emojis, hashtags, lang):
    descriptions = []
    for img in images:
        # Generate description using BLIP model
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        descriptions.append(description)
    
    # Combine all descriptions into one
    combined_description = " ".join(descriptions)

    # Define a prompt template with customization options
    prompt_template = PromptTemplate(
        input_variables=["image_description", "n_captions", "style", "length", "emojis", "hashtags", "lang"],
        template=(
            "Generate {n_captions} {style} Instagram caption for these images: {image_description}. "
            "The caption should be {length} and in {lang}. "
            "{emojis} {hashtags}"
        )
    )

    # Initialize the OpenAI model within LangChain
    llm = OpenAI(api_key=openai.api_key)
    
    # Create a LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate caption using the chain
    emoji_text = "Include emojis." if emojis else ""
    hashtag_text = f"Include these hashtags: {hashtags}" if hashtags else ""
    generated_caption = chain.run({
        "image_description": combined_description,
        "n_captions": n_captions,
        "style": style.lower(),
        "length": length.lower(),
        "emojis": emoji_text,
        "hashtags": hashtag_text,
        "lang": lang.lower()
    })
    
    return generated_caption

# Display generated caption
if st.button("Generate Caption") and images:
    caption = generate_caption(images, n_captions, caption_style, caption_length, include_emojis, custom_hashtags, language)
    st.write("Generated Caption:")
    st.write(caption)