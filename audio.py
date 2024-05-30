
from pydub import AudioSegment
import tempfile
import os
import streamlit as st
from scipy.signal import butter, filtfilt
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

#Classification App
# Define custom data loader
def custom_data_loader():
    st.write("Define a custom Pytorch Dataset object that uses audio transforms to pre-process an audio file and prepare one data item at a time.")

# Prepare batches of data with the data loader
def prepare_data_loader():
    st.write("Split data for training and validation, load features and labels from Pandas dataframe, and create training and validation Data Loaders.")

# Create model
def create_model():
    st.write("Create a CNN classification architecture to process Mel Spectrogram images, consisting of four convolutional blocks and a linear classifier layer.")

# Training
def training():
    st.write("Define functions for optimizer, loss, and scheduler. Train the model for several epochs, processing a batch of data in each iteration.")

# Inference
def inference():
    st.write("Run an inference loop on validation data, disable gradient updates, and get predictions for unseen data.")

# Visualize audio
def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    plt.colorbar(img, format='%+2.0f dB')
    st.pyplot(fig)

# Generate updated audio file
def generate_updated_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y_updated = y + 0.1 * np.random.randn(len(y))
    updated_audio_path = "updated_audio.wav"
    librosa.output.write_wav(updated_audio_path, y_updated, sr)
    return updated_audio_path

# Visualize audio comparison
def visualize_audio_comparison(original_audio_path, updated_audio_path):
    y_original, sr_original = librosa.load(original_audio_path, sr=None)
    y_updated, sr_updated = librosa.load(updated_audio_path, sr=None)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].set_title("Original Audio")
    librosa.display.waveshow(y_original, sr=sr_original, ax=axs[0])
    axs[1].set_title("Updated Audio")
    librosa.display.waveshow(y_updated, sr=sr_updated, ax=axs[1])
    st.pyplot(fig)

# Main streamlit app
def  Classification_n():
    st.sidebar.title("Audio Classification Process")
    selected_step = st.sidebar.selectbox("Select Step", ["Define Custom Data Loader", "Prepare Data Loader", "Create Model", "Training", "Inference"])

    if selected_step == "Define Custom Data Loader":
        custom_data_loader()
    elif selected_step == "Prepare Data Loader":
        prepare_data_loader()
    elif selected_step == "Create Model":
        create_model()
    elif selected_step == "Training":
        training()
    elif selected_step == "Inference":
        inference()

    st.sidebar.title("Input Data")
    audio_file = st.sidebar.file_uploader("Upload Audio File", type=["wav"])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

    st.sidebar.title("Apply Pre-processing")
    if st.sidebar.button("Apply Pre-processing"):
        # Apply pre-processing to audio file
        # Placeholder code
        st.write("Pre-processing applied successfully.")

        st.sidebar.title("Get Updated Audio")
        if st.sidebar.button("Get Updated Audio"):
            # Generate updated audio file
            updated_audio_path = generate_updated_audio(audio_file)
            st.write("Updated audio generated successfully.")

            st.sidebar.title("Visualize Audio Comparison")
            if st.sidebar.button("Visualize Comparison"):
                visualize_audio_comparison(audio_file, updated_audio_path)

    st.sidebar.title("Visualize Audio")
    if st.sidebar.button("Visualize Audio"):
        if audio_file is not None:
            visualize_audio(audio_file)
        else:
            st.warning("Please upload an audio file first.")


#Time Shift preprocessing
# Function to perform audio time shift
def audio_time_shift():
    st.title('Audio Time Shift App')
    if uploaded_file is not None:
        # Load audio file
        y, sr = librosa.load(uploaded_file)

        # Display original audio
        st.subheader('Original Audio')
        st.audio(uploaded_file)

        # Apply time shift
        max_shift = len(y) // 2  # Maximum shift is half of the length of the audio
        shift_amount = st.slider("Shift Amount", -max_shift, max_shift, 0)

        if shift_amount > 0:
            y_shifted = np.concatenate((y[shift_amount:], np.zeros(shift_amount)))
        elif shift_amount < 0:
            y_shifted = np.concatenate((np.zeros(-shift_amount), y[:shift_amount]))
        else:
            y_shifted = y

        # Display shifted audio
        st.subheader('Shifted Audio')
        st.audio(y_shifted, format='audio/wav', sample_rate=sr)

    
#Spectrography
# Streamlit app
def spectrography_n():
    st.title('Mel Spectrogram App')
    if uploaded_file is not None:
        # Load audio file
        y, sr = librosa.load(uploaded_file)

        # Display original audio
        st.subheader('Original Audio')
        st.audio(uploaded_file)

        # Apply time shift
        max_shift = len(y) // 2  # Maximum shift is half of the length of the audio
        shift_amount = st.slider("Shift Amount", -max_shift, max_shift, 0)

        # Shift audio
        if shift_amount > 0:
            y_shifted = np.concatenate((y[shift_amount:], np.zeros(shift_amount)))
        elif shift_amount < 0:
            y_shifted = np.concatenate((np.zeros(-shift_amount), y[:shift_amount]))
        else:
            y_shifted = y

        # Display shifted audio
        st.subheader('Shifted Audio')
        st.audio(y_shifted, format='audio/wav', sample_rate=sr)

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=y_shifted, sr=sr)

        # Display Mel spectrogram
        st.subheader('Mel Spectrogram')
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time', sr=sr, ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        st.pyplot(fig)


#Resizing the audio
# Streamlit app
def Resize_n():
    st.title('Audio Resize App')
    if uploaded_file is not None:
        # Load audio file
        y, sr = librosa.load(uploaded_file)

        # Display original audio
        st.subheader('Original Audio')
        st.audio(uploaded_file)

        # Select resize method
        resize_method = st.radio("Select resize method:", ("Pad with Silence", "Truncate"))

        # Set target length
        target_length = st.slider("Target Length", min_value=1, max_value=len(y)*2, value=len(y))

        # Resize audio
        if resize_method == "Pad with Silence":
            y_resized = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y_resized = y[:target_length]

        # Display resized audio
        st.subheader('Resized Audio')
        st.audio(y_resized, format='audio/wav', sample_rate=sr)



 #Function to apply time masking
def apply_time_mask(spec, max_mask_time=100):
    masked_spec = spec.copy()
    mask_time = random.randint(0, max_mask_time)
    start = random.randint(0, spec.shape[1] - mask_time)
    masked_spec[:, start:start + mask_time] = 0
    return masked_spec

# Function to apply frequency masking
def apply_freq_mask(spec, max_mask_freq=30):
    masked_spec = spec.copy()
    mask_freq = random.randint(0, max_mask_freq)
    start = random.randint(0, spec.shape[0] - mask_freq)
    masked_spec[start:start + mask_freq, :] = 0
    return masked_spec

# Function to apply both time and frequency masking
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = int(max_mask_pct * n_mels)
    for _ in range(n_freq_masks):
        start = random.randint(0, n_mels - freq_mask_param)
        aug_spec[start:start + freq_mask_param, :] = mask_value

    time_mask_param = int(max_mask_pct * n_steps)
    for _ in range(n_time_masks):
        start = random.randint(0, n_steps - time_mask_param)
        aug_spec[:, start:start + time_mask_param] = mask_value

    return aug_spec


#Masking the audio
def masking_n():
    # Streamlit app
    if uploaded_file is not None:
        # Load audio file
        y, sr = librosa.load(uploaded_file)

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)

        # Display original Mel spectrogram
        st.subheader('Original Mel Spectrogram')
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time', sr=sr, ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        st.pyplot(fig)

        # Display the original audio
        st.subheader('Original Audio')
        st.audio(y, format='audio/wav', sample_rate=sr)

        # Select masking type
        masking_type = st.selectbox("Select Masking Type:", ("None", "Time Masking", "Frequency Masking", "Both Time and Frequency Masking"))

        if masking_type == "Time Masking":
            masked_spec = apply_time_mask(S)
        elif masking_type == "Frequency Masking":
            masked_spec = apply_freq_mask(S)
        elif masking_type == "Both Time and Frequency Masking":
            masked_spec = spectro_augment(S)
        else:
            masked_spec = S  # No masking

        # Display masked Mel spectrogram
        st.subheader('Masked Mel Spectrogram')
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.power_to_db(masked_spec, ref=np.max), y_axis='mel', x_axis='time', sr=sr, ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Masked Mel spectrogram')
        plt.tight_layout()
        st.pyplot(fig)

        # Convert masked Mel spectrogram back to audio
        y_masked = librosa.feature.inverse.mel_to_audio(masked_spec, sr=sr)

        # Display the masked audio
        st.subheader('Masked Audio')
        st.audio(y_masked, format='audio/wav', sample_rate=sr)


#Filters in audio
def apply_filter(y, sr, filter_type, cutoff_freqs):
    nyquist = sr / 2
    if filter_type == 'high-pass':
        cutoff = cutoff_freqs[0] / nyquist
        b, a = butter(5, cutoff, btype='high', analog=False)
    elif filter_type == 'low-pass':
        cutoff = cutoff_freqs[0] / nyquist
        b, a = butter(5, cutoff, btype='low', analog=False)
    elif filter_type == 'band-pass':
        low_cutoff = cutoff_freqs[0] / nyquist
        high_cutoff = cutoff_freqs[1] / nyquist
        b, a = butter(5, [low_cutoff, high_cutoff], btype='band', analog=False)

    y_filtered = filtfilt(b, a, y)
    return y_filtered, sr

def filter_n():

    if uploaded_file is not None:
        # Load audio file
        y, sr = librosa.load(uploaded_file)

        # Display original audio
        st.subheader('Original Audio')
        st.audio(uploaded_file)

        # Select filter type
        filter_type = st.selectbox("Select filter type:", ("High-pass", "Low-pass", "Band-pass"))

        # Set default cutoff frequencies
        default_low_cutoff = 20
        default_high_cutoff = sr // 2

        if filter_type == "High-pass":
            cutoff_freq = st.slider("Cutoff Frequency", 20, sr // 2, 1000)
            y_filtered, sr = apply_filter(y, sr, 'high-pass', [cutoff_freq])
        elif filter_type == "Low-pass":
            cutoff_freq = st.slider("Cutoff Frequency", 20, sr // 2, 1000)
            y_filtered, sr = apply_filter(y, sr, 'low-pass', [cutoff_freq])
        else:
            low_cutoff_freq = st.slider("Low Cutoff Frequency", 20, sr // 2, 500)
            high_cutoff_freq = st.slider("High Cutoff Frequency", low_cutoff_freq, sr // 2, 1000)
            y_filtered, sr = apply_filter(y, sr, 'band-pass', [low_cutoff_freq, high_cutoff_freq])

        # Display filtered audio
        st.subheader('Filtered Audio')
        st.audio(y_filtered, format='audio/wav', sample_rate=sr)  # Specify the sample_rate parameter
        

#Sampling_rate_conversion
def Sampling_rate_conversion():
    if uploaded_file is not None:
        target_sampling_rate = st.selectbox("Select Sampling Rate:", [("44.1 kHz CD Quality and Standard quality audio", 44100), ("48 kHz DVD and digital television", 48000)])

        # if st.button(f"Convert to {target_sampling_rate[0]}"):
            # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Load audio file from the temporary location
        audio = AudioSegment.from_file(temp_file_path)

        # Convert to the target sampling rate
        audio = audio.set_frame_rate(target_sampling_rate[1])

        # Remove the temporary file
        os.remove(temp_file_path)

        # Get audio data as bytes
        audio_bytes = audio.export(format="wav").read()

        # Display the audio
        st.audio(audio_bytes, format="audio/wav")

# Streamlit app
st.title("Data Nexus - Audio Preprocessing")
# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])


task = st.selectbox("Select Task:", ("Sampling Rate Conversion", "Audio Filtering","Masking", "Resizing","Spectography","TimeShift","Auto Classification App"))

if task == "Sampling Rate Conversion":
    Sampling_rate_conversion()
elif task == "Audio Filtering":
    filter_n()
elif task == "Masking":
    masking_n()
elif task == "Resizing":
    Resize_n()
elif task == "Spectography":
    spectrography_n()
elif task == "TimeShift":
    audio_time_shift()
elif task == "Auto Classification App":
    Classification_n()


