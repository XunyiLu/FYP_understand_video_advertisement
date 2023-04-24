pip install gradio

from google.colab import drive

drive.mount('/content/gdrive/', force_remount=True)

from tensorflow import keras
from imutils import paths
from keras import models
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

import gradio as gr


IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Frames extraction
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Load label
df_path = '/content/gdrive/MyDrive/video_tags_test.csv'
df = pd.read_csv(df_path)
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df["tag"])
)
LABELS = label_processor.get_vocabulary()

# Load the model
model_CNN = models.load_model("/content/gdrive/MyDrive/dataset/CNN")
print('1')
model_RNN = models.load_model("/content/gdrive/MyDrive/dataset/gru_model_bestgrumodel_01")

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = model_CNN.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    # frames = load_video(os.path.join("test", path))
    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = model_RNN.predict([frame_features, frame_mask])[0]


    # for i in np.argsort(probabilities)[::-1]:
    #     print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = []

    for i in sorted_indices:
        class_name = class_vocab[i]
        probability = probabilities[i]
        sorted_probabilities.append((class_name, probability))
    return sorted_probabilities


def gradio_wrapper(video_path):
    predictions = sequence_prediction(video_path)
    print("Raw predictions:", predictions)  # Debug print
    scores = {class_name: score for class_name, score in predictions}
    print("Scores:", scores)  # Debug print
    # formatted_scores = {class_name: f"{score * 100:.1f}%" for class_name, score in scores.items()}
    formatted_scores = {class_name: f"{score :.4f}" for class_name, score in scores.items()}
    print("Formatted scores:", formatted_scores)  # Debug print
    return formatted_scores

# Define the input and output components for the Gradio interface
video_input = gr.inputs.Video(label="Input Video")
label_output = gr.outputs.Label(label="Scores")

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=video_input,
    outputs=label_output,
    title="Video Classification",
    description="Upload a video and get the predicted exciting scores.",
    examples=[
        ["/content/_T5FcBB9GIw.mp4"],
        ["/content/_hnOCUkbix0.mp4"]
    ]
    )


# Launch the interface
iface.launch(share=True)

# Debug: 
test_video = np.random.choice(df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
gradio_wrapper(test_video)
