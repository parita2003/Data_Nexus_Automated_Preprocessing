import streamlit as st
import subprocess

st.title("Data Nexus - Automated Preprocessing App")

# Main content
st.header("Choose an Option")

# Create layout with 3 columns for the cards
col1, col2, col3 = st.columns(3)

# Text preprocessing card
with col1:
    if st.button("Text Preprocessing"):
        subprocess.Popen(["streamlit", "run", "text.py"])

# Image preprocessing card
with col2:
    if st.button("Image Preprocessing"):
        subprocess.Popen(["streamlit", "run", "img.py"])

# Audio preprocessing card
with col3:
    if st.button("Audio Preprocessing"):
        subprocess.Popen(["streamlit", "run", "audio.py"])
