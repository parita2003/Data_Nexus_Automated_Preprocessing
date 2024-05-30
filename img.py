import streamlit as st
import numpy as np
import cv2
from skimage import exposure, color, filters, morphology
from PIL import Image
import matplotlib.pyplot as plt
import io

# Helper functions
def apply_grayscaling(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_normalization(image):
    return exposure.equalize_adapthist(image)

def apply_noise_reduction(image):
    # Example: Gaussian smoothing
    return filters.gaussian(image, sigma=1)

def apply_image_resizing(image, target_size):
    return cv2.resize(image, target_size)

def apply_color_correction(image):
    if image.ndim == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] != 3:  # If not RGB, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = color.rgb2lab(image)
    avg_a = np.mean(lab_image[:, :, 1])
    avg_b = np.mean(lab_image[:, :, 2])
    corrected_image = lab_image.copy()
    corrected_image[:, :, 1] = corrected_image[:, :, 1] - ((avg_a - 128) * (corrected_image[:, :, 0] / 255.0) * 1.1)
    corrected_image[:, :, 2] = corrected_image[:, :, 2] - ((avg_b - 128) * (corrected_image[:, :, 0] / 255.0) * 1.1)
    return np.clip(color.lab2rgb(corrected_image), 0, 1)

def apply_segmentation(image):
    # Example: Thresholding
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    return morphology.binary_erosion(binary)

def apply_canny_edge_detection(image):
    return filters.sobel(image)

def apply_harris_corner_detection(image):
    # Example: Use corner_harris() from OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [255, 0, 0]
    return image

def apply_texture_analysis(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        gray_image = image
    else:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply a Gabor filter
    filt_real, filt_imag = filters.gabor(gray_image, frequency=0.6)
    return filt_real

def apply_erosion(image):
    # Create a structuring element
    kernel = np.ones((3, 3), np.uint8)
    # Perform erosion
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded

def apply_dilation(image):
    # Create a structuring element
    kernel = np.ones((3, 3), np.uint8)
    # Perform dilation
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated

st.title("Data Nexus - Image Preprocessing")
# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the image
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # List of preprocessing options
    preprocessing_options = {
        "Grayscaling": apply_grayscaling,
        "Normalization": apply_normalization,
        "Noise Reduction": apply_noise_reduction,
        "Image Resizing": apply_image_resizing,
        "Color Correction": apply_color_correction,
        "Segmentation": apply_segmentation,
        "Canny Edge Detection": apply_canny_edge_detection,
        "Harris Corner Detection": apply_harris_corner_detection,
        "Texture Analysis": apply_texture_analysis,
        "Dilation": apply_dilation,
        "Erosion": apply_erosion
    }

    # Select preprocessing options
    selected_options = st.multiselect("Select preprocessing options", list(preprocessing_options.keys()))

    # Custom image resizing
    custom_resize = st.checkbox("Custom Resize")
    if custom_resize:
        custom_height = st.slider("Height", 1, 1000, image.shape[0])
        custom_width = st.slider("Width", 1, 1000, image.shape[1])
        custom_size = (custom_width, custom_height)
    else:
        custom_size = None

    # Apply selected preprocessing options
    processed_image = image.copy()
    for option in selected_options:
        if option == "Image Resizing" and custom_size is not None:
            processed_image = preprocessing_options[option](processed_image, custom_size)
        else:
            processed_image = preprocessing_options[option](processed_image)
        processed_image = (processed_image * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        st.image(processed_image, caption=option, use_column_width=True)
    # Download option for each step
    if st.button("Download Final Image"):
        output = io.BytesIO()
        processed_image_download = (processed_image * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        Image.fromarray(processed_image_download).save(output, format="PNG")
        output.seek(0)
        st.download_button("Download Final Image", output, "final_image.png", "image/png")