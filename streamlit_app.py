"""
    @author     harsh-dhamecha
    @create date 2024-05-29 22:35:27
    @modify date 2024-06-01 14:57:47
    @desc       Streamlit app file
"""

import streamlit as st
import requests

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
        try:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            description_response = requests.post("http://localhost:8000/describe_images/", files=files)
            description_response.raise_for_status()
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
            caption_response.raise_for_status()
            captions = caption_response.json()["captions"]
            st.write("Generated Caption(s):")
            for caption in captions:
                st.write(caption)
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with backend: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
