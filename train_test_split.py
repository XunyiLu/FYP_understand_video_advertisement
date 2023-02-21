import random
import os


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


#
# video_dir = '/Users/lxy/PycharmProjects/abcd/dataset/video'
# video_files = os.listdir(video_dir)
# print("NB of videos in video_dir is " + str(len(video_files)))
# data = []
# for file in video_files:
#     data.append(file.split(".mp4")[0])
# # print(data)
# train_data, test_data = train_test_split(data, test_size=0.3)

# print("Training data is: ", train_data)
# print("Length of training data is ", len(train_data))
# print("Testing data is: ", test_data)
# print("Length of testing data is ", len(test_data))
