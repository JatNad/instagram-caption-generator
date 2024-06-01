'''
 	@author 	 harsh-dhamecha
 	@email       harshdhamecha10@gmail.com
 	@create date 2024-05-29 22:35:27
 	@modify date 2024-05-29 22:44:37
 	@desc        Streamlit app file
 '''

import streamlit as st
import requests

st.title("Instagram Caption Generator")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=150)

# Collect inputs from the user
n_captions = st.selectbox("Select Number of Captions", ["1", "2", "5", "10"], key="n_captions")
caption_style = st.selectbox("Select Caption Style", ["Formal", "Informal", "Humorous", "Inspirational", "Poetic"], key="caption_style")
caption_length = st.selectbox("Select Caption Length", ["Short", "Medium", "Long"], key="caption_length")
include_emojis = st.checkbox("Include Relevant Emojis", key="emojis")
include_hashtags = st.checkbox("Include Relevant Hashtags", key="hashtags")

if st.button("Generate Caption"):
    if uploaded_files:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        description_response = requests.post("http://localhost:8000/describe_images/", files=files)
        description = description_response.json()["description"]
        
        caption_request = {
            "description": description,
            "n_captions": int(n_captions),
            "caption_style": caption_style,
            "caption_length": caption_length,
            "include_emojis": include_emojis,
            "include_hashtags": include_hashtags
        }
        
        caption_response = requests.post("http://localhost:8000/generate_captions/", json=caption_request)
        captions = caption_response.json()["captions"]
        st.write("Generated Caption(s):")
        for caption in captions:
            st.write(caption)

