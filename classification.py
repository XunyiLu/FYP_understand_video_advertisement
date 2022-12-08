import pandas as pd
from tensorflow import keras
# from imutils import paths
from pytube import YouTube
import tensorflow as tf
# import pandas as pd
import numpy as np
# import imageio
import csv
import cv2
import json
import os
import numpy as np

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



# def download_youtube_video(csv_pathway, video_dir):
#     with open(csv_pathway, 'r') as f:
#         for line in f.readlines():
#             line = line.replace("'", "")
#             data = 'https://www.youtube.com/watch?v=' + line
#             print(data)
#             yt = YouTube(data)
#
#             try:
#                 yt.streams.get_highest_resolution().download(video_dir,
#                                                              filename=line + '.mp4')
#                 print("Successfully download")
#                 # video.download('/User/lxy/Documents/')
#                 # print(data)
#
#             except:
#                 print("An exception occurred")
#     delete_ds_file(video_dir)


csv_pathway = '/Users/lxy/Desktop/FYP/final_video_id_list.csv'
cleaned_label_pathway = '/Users/lxy/Desktop/FYP/video_Exciting_clean.json'


def delete_ds_file(path):
    target_file = path
    result = os.listdir(target_file)
    put = '.DS_Store'  # put:1.txt
    if put in result:  # 精确到后缀名
        # print('Yes')
        os.remove(path + put)
    # else:
    #     # print('No')


# download_youtube_video(csv_pathway, video_dir)


def create_train_df(train_dir, label_pathway):
    delete_ds_file(train_dir)
    video_file_names = []
    for i in os.listdir(train_dir):
        video_file_names.append(i)
    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    train_list = []
    for name in video_file_names:
        for key in keys:
            if name.split('\n', 1)[0] == key:
                if pure_dict[key] >= 0.7:
                    train_list.append([name, "exciting"])
                else:
                    train_list.append([name, "unexciting"])
                break
    name = ["video_name", "label"]
    train_df = pd.DataFrame(columns=name, data=train_list)
    csv_pathway = train_dir + "train_df.csv"
    train_df.to_csv(csv_pathway, encoding='gbk')
    return train_df


def create_test_df(test_dir, label_pathway):
    delete_ds_file(test_dir)
    video_file_names = []
    for i in os.listdir(test_dir):
        video_file_names.append(i)
    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    test_list = []
    for name in video_file_names:
        for key in keys:
            if name.split('\n', 1)[0] == key:
                if pure_dict[key] >= 0.7:
                    test_list.append([name, "exciting"])
                else:
                    test_list.append([name, "unexciting"])
                break
    name = ["video_name", "label"]
    test_df = pd.DataFrame(columns=name, data=test_list)
    csv_pathway = test_dir + "test_df.csv"
    test_df.to_csv(csv_pathway, encoding='gbk')
    return test_df


train_df = create_train_df(train_vid_dir, cleaned_label_pathway)
print(f"Number of videos in train set is {len(train_df)}")
test_df = create_test_df(test_vid_dir, cleaned_label_pathway)
print(f"Number of videos in test set is {len(test_df)}")


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
    return np.array(frames)


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


# Label process
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()

    labels = df["label"].values
    labels = label_processor(labels[..., None]).numpy()

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

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    return (frame_features, frame_masks), labels


train_data = prepare_all_videos(train_df, train_vid_dir)[0]
train_labels = prepare_all_videos(train_df, train_vid_dir)[1]

test_data = prepare_all_videos(test_df, test_vid_dir)[0]
test_labels = prepare_all_videos(test_df, test_vid_dir)[1]

print(f"Frame feature in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

print(f"Frame feature in test set: {test_data[0].shape}")
print(f"Frame masks in test set: {test_data[1].shape}")


# Utility for our sequence model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return rnn_model


# Utility for running experiments
def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(test_vid_dir + path)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
