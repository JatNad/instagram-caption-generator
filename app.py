'''
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-05-25 11:06:48
 	@modify date 2024-05-25 12:53:00
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

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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
        st.image(img, caption=file.name)

# Function to generate a caption based on all images
def generate_caption(images):
    descriptions = []
    for img in images:
        # Generate description using BLIP model
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        descriptions.append(description)
    
    # Combine all descriptions into one
    combined_description = " ".join(descriptions)

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["image_description"],
        template="Generate a creative Instagram caption for these images: {image_description}"
    )

    # Initialize the OpenAI model within LangChain
    llm = OpenAI(api_key=openai.api_key)
    
    # Create a LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate caption using the chain
    generated_caption = chain.run({"image_description": combined_description})
    return generated_caption

# Display generated caption
if st.button("Generate Caption") and images:
    caption = generate_caption(images)
    st.write("Generated Caption:")
    st.write(caption)
