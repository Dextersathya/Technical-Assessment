# Rajinikanth Image Recognition using YOLOv8

This project focuses on detecting and recognizing images of the Indian actor **Rajinikanth** using YOLOv8, with dataset preparation, augmentation, and a Streamlit-based web UI for testing.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Training](#model-training)
4. [Results](#results)
5. [Web UI for Model Testing](#web-ui-for-model-testing)
6. [Roboflow Link](#roboflow-link)
7. [Setup and Installation](#setup-and-installation)
8. [Usage](#usage)

---

## Project Overview
The goal of this project is to create an object detection model capable of recognizing Rajinikanth's images. The project uses:
- **YOLOv8** for training and detection
- **Roboflow** for image annotation and augmentation
- **Streamlit** to build a web-based UI for testing the model on new images

---

## Dataset Preparation
1. **Images**: 50 images of Rajinikanth were collected and annotated using **Roboflow**.
2. **Augmentation**: To improve model performance, the dataset was augmented using:
   - Random Flip
   - Rotation
   - Blackout
3. **Final Dataset**:
   - **134 images** in total:
     - 117 for Training
     - 11 for Validation
     - 6 for Testing
4. **Data Configuration**: The `data.yaml` file was adjusted to match the dataset location and structure.

---

## Model Training
The YOLOv8 model was trained using the **Ultralytics** library. Below is the training script:

```python
from ultralytics import YOLO

# Initialize the YOLOv8 model
model = YOLO('yolov8n.yaml')  # YOLOv8 nano model configuration

# Train the model
results = model.train(
    data="/content/dataset/data.yaml",  # Path to dataset configuration
    epochs=10,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16    # Batch size
)
```

### Training Details:
- **Epochs**: 10 (to avoid overfitting)
- **Image Size**: 640x640
- **Batch Size**: 16

---

## Results
After training the model for 10 epochs, the following metrics were achieved:
- **mAP (Mean Average Precision)**: 76%
- **Precision**: 78%
- **Recall**: 75%

These results indicate a good balance of detection performance.

---

## Web UI for Model Testing
A simple **Streamlit** web UI was created to allow users to upload and test images with the trained model. The model can detect Rajinikanth's presence in unknown images.

---

## Roboflow Link
You can access the dataset and annotations on Roboflow here:
[**Rajinikanth Dataset on Roboflow**](https://universe.roboflow.com/sathya-a9kst/rajinikanth)

---

## Setup and Installation
Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics streamlit
   ```

3. Download the dataset and update the `data.yaml` file with the correct paths.

---

## Usage
### Train the Model
Run the following script to train the model:
```bash
python train_yolo.py
```

### Launch the Web UI
To test the model using the Streamlit web UI:
```bash
streamlit run app.py
```

Upload an image, and the model will predict if Rajinikanth is detected.

---

## Future Improvements
- Use a larger dataset for better generalization.
- Fine-tune the model with more augmentation techniques.
- Deploy the Streamlit app online for public access.

---

## Acknowledgments
- **Ultralytics** for the YOLOv8 model
- **Roboflow** for dataset annotation and augmentation tools
- **Streamlit** for building the web interface
