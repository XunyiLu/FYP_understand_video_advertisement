import random
import os
import json
import pandas as pd


def delete_ds_file(path):
    target_file = path
    result = os.listdir(target_file)
    put = '.DS_Store'  # put:1.txt
    if put in result:  # 精确到后缀名
        # print('Yes')
        os.remove(path + put)


def exciting_unexciting_split(data, label_pathway):
    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    exciting_data = []
    unexciting_data = []
    for name in data:
        for key in keys:
            if name == key:
                if pure_dict[key] >= 0.7:
                    exciting_data.append(name)
                if pure_dict[key] <= 0.3:
                    unexciting_data.append(name)
                break

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
    exciting_data, unexciting_data = exciting_unexciting_split(data, label_pathway)
    print("Exciting data is: ", exciting_data)
    print("NB of exciting data is ", len(exciting_data))
    print("Unexciting data is: ", unexciting_data)
    print("NB of unexciting data is ", len(unexciting_data))

    train_data_ex, test_data_ex = train_test_split(exciting_data, test_size=0.3)
    train_data_unex, test_data_unex = train_test_split(unexciting_data, test_size=0.3)

    print("train_data_ex is: ", train_data_ex)
    print("NB of train_data_ex is ", len(train_data_ex))

    print("test_data_ex is: ", test_data_ex)
    print("NB of test_data_ex is ", len(test_data_ex))

    print("train_data_unex is: ", train_data_unex)
    print("NB of train_data_unex is ", len(train_data_unex))

    print("test_data_unex is: ", test_data_unex)
    print("NB of test_data_unex is ", len(test_data_unex))

    train_set = train_data_ex + train_data_unex
    test_set = test_data_ex + test_data_unex

    print("NB of train_set is ", len(train_set))
    print("NB of test_set is ", len(test_set))

    df = open(label_pathway)
    pure_dict = json.load(df)
    keys = pure_dict.keys()
    train_list = []
    for name in train_set:
        for key in keys:
            if name == key:
                if pure_dict[key] >= 0.7:
                    train_list.append([name, "exciting"])
                if pure_dict[key] <= 0.3:
                    train_list.append([name, "unexciting"])
    test_list = []
    for name in test_set:
        for key in keys:
            if name == key:
                if pure_dict[key] >= 0.7:
                    test_list.append([name, "exciting"])
                if pure_dict[key] <= 0.3:
                    test_list.append([name, "unexciting"])
    name = ["video_name", "label"]
    train_df = pd.DataFrame(columns=name, data=train_list)
    train_df = train_df.sample(frac=1, random_state=42)
    csv_pathway = "/Users/lxy/Desktop/"
    train_df.to_csv(csv_pathway + "train.csv", encoding='gbk')
    test_df = pd.DataFrame(columns=name, data=test_list)
    test_df = test_df.sample(frac=1, random_state=42)
    test_df.to_csv(csv_pathway + "test.csv", encoding='gbk')
    return train_df, test_df


video_dir = '/Users/lxy/PycharmProjects/abcd/dataset/video/'
label_pathway = "/Users/lxy/Desktop/FYP/video_Exciting_clean.json"
create_train_test_df(video_dir, label_pathway)

