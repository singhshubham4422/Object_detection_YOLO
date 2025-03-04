# 🐾 YOLOv5 Animal Detection 🐾

Welcome to the YOLOv5 Animal Detection project! 🧙🏻‍♂️🐅🐺🐻🐾 This project uses YOLOv5, a state-of-the-art object detection model, to detect animals in images and videos. It's designed to be easily run in a Google Colab environment, where you can quickly upload your own images or videos to see the power of animal detection in action!

🔗 **[Open the Colab Notebook here!](https://colab.research.google.com/drive/1GgItU3FM0WCzChGlw4CPTLQ7QfJjHn7a?usp=sharing)**

---

## 🔥 Features

- **Pre-trained YOLOv5 Model**: Detect animals in images and videos without training a model from scratch.
- **Google Colab Ready**: Run everything directly in Google Colab.
- **Fast & Accurate Detection**: YOLOv5 is optimized for both speed and accuracy.
- **Supports Multiple Formats**: Detect animals in both images and videos.
- **Customizable**: Fine-tune the model on your own dataset if needed.

---

## 🛠️ Requirements

- **Google Colab Account** (to run in the cloud).
- **Python 3.x** (already available in Colab).
- **Basic understanding of Object Detection** and YOLOv5.

---

## 🚀 Getting Started

### 1. Clone the YOLOv5 Repository

Clone the official YOLOv5 repository to get the latest version of the code.

```bash
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
```

### 2. Install Dependencies

YOLOv5 requires several libraries. Install them using the following command:

```bash
!pip install -U -r requirements.txt
```

### 3. Upload Images or Videos

In Google Colab, you can upload your images or videos using the following code:

```python
from google.colab import files
uploaded = files.upload()
```

This will allow you to select and upload your local image or video file.

### 4. Run YOLOv5 on the Image or Video

#### For images:

```python
import torch
from PIL import Image

# Load YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to your uploaded image
img_path = '/content/your_uploaded_image.jpg'  # change to your image path

# Perform inference
results = model(img_path)

# Show the results
results.show()

# Save the results
results.save()
```

#### For videos:

```python
# Path to your uploaded video
video_path = '/content/your_uploaded_video.mp4'  # change to your video path

# Perform inference on the video
results = model(video_path)

# Show the results
results.show()

# Save the processed video
results.save()
```

---

## 🐾 How It Works

YOLOv5 is a deep learning model trained to detect objects in images and videos. In this case, we use a pre-trained YOLOv5 model (`yolov5s`) and apply it to detect animals. The model detects animals like cats, dogs, lions, etc., and places bounding boxes around them with labels and confidence scores.

---

## 🎯 Advanced Usage

### 1. Training on Custom Data

If you want to train YOLOv5 on your own dataset, you can prepare your custom dataset and use the following command:

```bash
!python train.py --img 640 --batch 16 --epochs 50 --data /path/to/custom.yaml --weights yolov5s.pt
```

Where `custom.yaml` is your custom dataset configuration file. The dataset should be in YOLO format and include class labels (e.g., dog, cat).

### 2. Fine-Tuning the Pre-Trained Model

You can fine-tune the pre-trained YOLOv5 model on your custom dataset. Use the following command:

```bash
!python train.py --data /path/to/custom.yaml --weights yolov5s.pt --batch-size 16 --epochs 50
```

---

## 📚 Conclusion

This project is an easy-to-use tool for detecting animals in images and videos using the YOLOv5 object detection model. You can run it in Google Colab, upload your own images or videos, and instantly see the results. Whether you are a wildlife enthusiast, zoologist, or just curious, this is a great way to experiment with YOLOv5 for animal detection. 🐾🐕🐈

---

## 💬 Contributions

If you have any suggestions, improvements, or fixes, feel free to open an issue or submit a pull request! Contributions are always welcome. 😊
