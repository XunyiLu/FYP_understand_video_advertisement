import os
import random
import os
import pandas as pd
import numpy as np
import json
from tensorflow import keras
import time
from keras import layers


IMG_SIZE = 299
BATCH_SIZE = 64
EPOCHS = 15
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
random.seed(42)
start_time = time.time()

class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])
  
  def call(self, x):
    return self.seq(x)
def delete_ds_file(path):
    target_file = path
    result = os.listdir(target_file)
    put = '.DS_Store'  # put:1.txt
    if put in result:
        # print('Yes')
        os.remove(path + put)



def exciting_unexciting_split(data, label_pathway):
    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    print("key size ",len(keys))
    exciting_data = []
    unexciting_data = []
    for name in data:
     try:
      name1=name
      name=name.split("###")[1].strip()
      for key in keys:
       if name == key:
        if pure_dict[key] >= 0.5:
         exciting_data.append(name1)
        if pure_dict[key] <0.5:
         unexciting_data.append(name1)
        break
     except:
      print("here")
      c=0

    return exciting_data, unexciting_data


def train_test_split(data, test_size):
    """
    Split the data into training and testing sets.
    data: list, array, or DataFrame
    test_size: float between 0 and 1
    """
    data_size = len(data)
    test_size = int(test_size * data_size)

    # Shuffle the data
    random.shuffle(data)

    # Split the data
    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data


def create_train_test_df(video_dir, label_pathway):
    delete_ds_file(video_dir)
    video_files = os.listdir(video_dir)
    data = []
    for file in video_files:
        data.append(file.split(".mp4")[0])
    print(len(data))
    exciting_data, unexciting_data = exciting_unexciting_split(data, label_pathway)
    #print("Exciting data is: ", exciting_data)
    # print("NB of exciting data is ", len(exciting_data))
    # print("Unexciting data is: ", unexciting_data)
    # print("NB of unexciting data is ", len(unexciting_data))

    train_data_ex, test_data_ex = train_test_split(exciting_data, test_size=0.3)
    train_data_unex, test_data_unex = train_test_split(unexciting_data, test_size=0.3)

    # print("train_data_ex is: ", train_data_ex)
    # print("NB of train_data_ex is ", len(train_data_ex))
    #
    # print("test_data_ex is: ", test_data_ex)
    # print("NB of test_data_ex is ", len(test_data_ex))
    #
    # print("train_data_unex is: ", train_data_unex)
    # print("NB of train_data_unex is ", len(train_data_unex))
    #
    # print("test_data_unex is: ", test_data_unex)
    # print("NB of test_data_unex is ", len(test_data_unex))

    train_set = train_data_ex + train_data_unex
    test_set = test_data_ex + test_data_unex
    #print(len(train_set))
    # print("NB of train_set is ", len(train_set))
    # print("NB of test_set is ", len(test_set))

    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    print(len(keys))
    #print(keys)
    train_list = []
    for name in train_set:
        name1=name.strip()
        name=name.split("###")[1].strip()
        #print(name)
        for key in keys:
            if name == key:
                if pure_dict[key] >= 0.7:
                    train_list.append([name1, "exciting"])
                if pure_dict[key] <= 0.3:
                    train_list.append([name1, "unexciting"])
    test_list = []
    for name in test_set:
        name1=name.strip() 
        name = name.split("###")[1].strip()
        for key in keys:
            if name == key:
                if pure_dict[key] >= 0.7:
                    test_list.append([name1, "exciting"])
                if pure_dict[key] <= 0.3:
                    test_list.append([name1, "unexciting"])
    name = ["video_name", "label"]
    train_df = pd.DataFrame(columns=name, data=train_list)
    test_df = pd.DataFrame(columns=name, data=test_list) 
    return train_df, test_df


video_dir = '/Users/prochetasen/Downloads/videos/'
label_pathway = "/Users/prochetasen/Downloads/annotations_videos/video/cleaned_result/video_Exciting_clean.json"

train_df, test_df = create_train_test_df(video_dir, label_pathway)

l=train_df

# Label process
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)
#print(label_processor.get_vocabulary())


def prepare_data(df, basePath):
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
    print(num_samples)
    for idx, path in enumerate(video_paths):
        path=path.split("###")[1]
        try:
         temp_features = np.load(os.path.join(basePath, path) + "_feature.npy")
         temp_masks = np.load(os.path.join(basePath, path) + "_mask.npy")
         frame_features[idx,] = temp_features.squeeze()
         frame_masks[idx,] = temp_masks.squeeze()
        except:
         c=0

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_data(train_df, "/Users/prochetasen/Downloads/features/")
#print(train_labels)

test_data, test_labels = prepare_data(test_df, "/Users/prochetasen/Downloads/features/")

#for i in range(2):
#print(len(train_data[0][i]))
#for j in range(len(train_data[0][i])):
print(train_data[0][4])
print(train_data[0][5])
#print("\n")
#print(train_data[0][1])
#print(train_data[1])
#sys.exit()


print(f"Frame feature in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

print(f"Frame feature in test set: {test_data[0].shape}")
print(f"Frame masks in test set: {test_data[1].shape}")


# Utility for our sequence model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(2000, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = keras.layers.GRU(550)(x)
    x = keras.layers.Dropout(.9)(x)
    x = keras.layers.Dense(500, activation="softmax")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer = keras.optimizers.Adam(learning_rate = 0.0001), metrics=["accuracy"]
    )

    return rnn_model


# Utility for running experiments
def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=False, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    #seq_model.load_weights(filepath)
    l=seq_model.predict([test_data[0],test_data[1]])
    print(l)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()


end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
