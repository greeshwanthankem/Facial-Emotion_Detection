# Emotion Detection Using Deep Learning

## Overview
This project implements an emotion detection system using a deep learning model trained on the FER2013 dataset. The model is capable of recognizing emotions from facial expressions in images or real-time video feeds.

## Project Structure
```
ML_PROJECT/
│── data/
│   ├── fer2013.csv               # Dataset file
│── emotion_detection_model.h5    # Trained model
│── main.ipynb                    # Jupyter notebook with training & evaluation
│── model.png                     # Model architecture visualization
│── README.md                     # Project documentation
```

## Dataset
The project uses the **Facial Expression Recognition (FER2013)** dataset from Kaggle.
- Dataset link: [FER2013 Dataset](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- Contains 35,887 grayscale images of 48x48 pixels categorized into 7 emotions:
  - **Angry**
  - **Disgust**
  - **Fear**
  - **Happy**
  - **Neutral**
  - **Sad**
  - **Surprise**

## Installation
Before running the project, install the required dependencies:
```sh
pip install opencv-python tensorflow pandas numpy matplotlib scikit-learn seaborn
```

## Usage
1. **Train the Model:**
   - Open `main.ipynb` in Jupyter Notebook.
   - Run the cells to load the dataset, preprocess data, and train the model.
   - The trained model is saved as `emotion_detection_model.h5`.

2. **Evaluate the Model:**
   - The notebook includes code for generating a confusion matrix and classification report.
   - Run the evaluation section to see the model's performance.

3. **Real-Time Emotion Detection:**
   - Run the OpenCV-based detection script in `main.ipynb`.
   - The webcam captures faces, detects emotions, and displays predictions in real time.
   - Press `q` to exit.

## Model Architecture
- Convolutional Neural Network (CNN) with layers:
  - **Conv2D, MaxPooling2D, Flatten, Dense, Dropout**
- Categorical cross-entropy loss with Adam optimizer.
- Training for **100 epochs** with a batch size of **64**.

## Results
- Confusion matrix and classification report included in `main.ipynb`.
- Accuracy and loss graphs plotted for training and validation sets.
- Real-time emotion detection using OpenCV.

## License
This project is for educational purposes. Feel free to use and modify it.

