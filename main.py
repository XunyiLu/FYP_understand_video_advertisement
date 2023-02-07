import pandas as pd
from tensorflow import keras
from pytube import YouTube
# import tensorflow as tf
import numpy as np
# import csv
import cv2
import json
import os

IMG_SIZE = 299
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

dataset_dir = os.getcwd() + '/dataset/'
video_dir = dataset_dir + '/video/'
frame_dir = dataset_dir + '/frame/'
feature_dir = dataset_dir + '/feature/'
train_vid_dir = dataset_dir + '/train_vid/'
test_vid_dir = dataset_dir + '/test_vid/'
print(os.getcwd())

# Create the directories
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(video_dir):
    os.mkdir(video_dir)
if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)
if not os.path.exists(feature_dir):
    os.mkdir(feature_dir)
if not os.path.exists(train_vid_dir):
    os.mkdir(train_vid_dir)
if not os.path.exists(test_vid_dir):
    os.mkdir(test_vid_dir)

# label_path is the path of the cleaned annotation file
label_path = '/Users/lxy/Desktop/FYP/video_Exciting_clean.json'

# This function returns a dataframe contains all names of the effective videos. 924346
# Compared to the previous code, in this case, we don't need to download all the videos.
# Instead, only 976 videos are effective. This saves time.
# effective video means the video with score larger than 0.7 or smaller than 0.3


def df_effective_videos(label_path):
    df = open(label_path)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    df_effective_video = []
    for key in keys:
        if pure_dict[key] >= 0.7:
            df_effective_video.append(key)
        if pure_dict[key] <= 0.3:
            df_effective_video.append(key)
    return df_effective_video


df = df_effective_videos(label_path)
# To save the dataframe in dataset directory
with open(dataset_dir + "df_effective_videos.txt", "w") as f:
    for s in df:
        f.write(str(s) +"\n")


# download videos
def download_effective_videos(df_effective_video):
    for i in df_effective_video:
        data = 'https://www.youtube.com/watch?v=' + i
        print(data)
        yt = YouTube(data)
        try:
            yt.streams.get_highest_resolution().download(video_dir, filename= i + ".mp4")
            print("Successfully download")

        except:
            print("An exception occurred")


download_effective_videos(df)


# crop frames
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frame=0, resize=(IMG_SIZE, IMG_SIZE)):
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

            if len(frames) == max_frame:
                break
    finally:
        cap.release()
    # np.save(frame_dir + "1.np", np.array(frames))
    return np.array(frames)


# load_video("/Users/lxy/PycharmProjects/abcd/dataset/video/1.mp4")


# extract and store features
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


# This is a test. Ignore it. 
# def test():
#     # frame_masks = np.zeros(shape= MAX_SEQ_LENGTH, dtype="bool")
#     # frame_features = np.zeros(
#     #     shape=( MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
#     # )
#     frames = load_video("/Users/lxy/PycharmProjects/abcd/dataset/video/1.mp4")
#     frames = frames[None, ...]
#     # Initialize placeholders to store the masks and features of the current video
#     temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
#     temp_frame_features = np.zeros(
#         shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
#     )
#     # Extract features from the frames of the current video
#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(MAX_SEQ_LENGTH, video_length)
#         for j in range(length):
#             temp_frame_features[i, j, :] = feature_extractor.predict(
#                 batch[None, j, :]
#             )
#         temp_frame_mask[i, :length] = 1   # 1 = not masked, 0 = masked
#
#     np.save(frame_dir + "features", temp_frame_features)
#     np.save(frame_dir + "masks", temp_frame_mask)
#     return temp_frame_features, temp_frame_mask


# test()

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video
    for idx, path in enumerate(video_paths):
        # Gathering all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1   # 1 = not masked, 0 = masked
        np.save(frame_dir + path + "features", temp_frame_features)
        np.save(frame_dir + path + "masks", temp_frame_mask)

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    return frame_features, frame_masks


features = prepare_all_videos(df, video_dir)[0]
masks = prepare_all_videos(df, video_dir)[1]
