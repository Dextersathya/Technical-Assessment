import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import io
import os

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="KToDYINoPms2su0Tzxtj"  # Replace with your actual API key
)

# Function to draw bounding boxes on the image
def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    
    for pred in predictions:
        # Extract bounding box and label
        x_min = pred['x'] - pred['width'] / 2
        y_min = pred['y'] - pred['height'] / 2
        x_max = pred['x'] + pred['width'] / 2
        y_max = pred['y'] + pred['height'] / 2
        label = pred['class']
        
        # Draw rectangle and label
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min), label, fill="red")
    
    return image

# Streamlit UI
st.title("Image Upload and Inference App")

# Image upload
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Run inference on the image using the Roboflow API
    response = CLIENT.infer(image_path, model_id="rajinikanth/2")
    
    # Parse predictions from the response
    predictions = response['predictions']
    
    # Open the image again to draw bounding boxes
    image = Image.open(image_path)
    
    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image.copy(), predictions)
    
    # Display the image with bounding boxes

    st.image(image_with_boxes, caption="Processed Image with Bounding Boxes", use_container_width=True)


    # Convert the image to bytes for download
    img_byte_arr = io.BytesIO()
    image_with_boxes.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    # Option to download the processed image
    st.download_button("Download Processed Image", img_byte_arr, "processed_image.jpg", "image/jpeg")

    # Optionally, clean up the temporary image file
    os.remove(image_path)
