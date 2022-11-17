import sys
import os
from pytube import YouTube
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, LSTM, Dropout, \
    TimeDistributed, GRU, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import PIL


'''
delete_ds_file function: input the pathway of the file, and it will delete the _DS.store file
'''
def delete_ds_file(path):
    target_file = path
    result = os.listdir(target_file)
    put = '.DS_Store'  # put:1.txt
    if put in result:  # 精确到后缀名
        print('Yes')
        os.remove(path + put)
    else:
        print('No')


def set_up_directory_layout():
    dataset_dir = os.getcwd() + '/dataset/'
    video_dir = dataset_dir + '/video/'
    frame_dir = dataset_dir + '/frame/'
    feature_dir = dataset_dir + '/feature/'
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

    return dataset_dir, video_dir, frame_dir, feature_dir


'''Step 1: download videos from YouTube to seq_dir'''


def download_youtube_videos(video_dir):
    with open('/Users/lxy/Desktop/FYP/final_video_id_list.csv', 'r') as f:
        for line in f.readlines():
            line = line.replace("'", "")
            data = 'https://www.youtube.com/watch?v=' + line
            print(data)
            yt = YouTube(data)

            try:
                # print(yt.)
                yt.streams.get_highest_resolution().download(video_dir, filename=line + ".mp4")
                # video.download('/User/lxy/Documents/')
                # print(data)
            except:
                print("An exception occurred")


'''Step 2: Extract frames of the downloaded videos to youtube_frame_dir'''


def frame_extraction(video_dir, frame_dir):

    ''' Step 2.1 Delete .DS_Store file '''

    target_file = video_dir
    result = os.listdir(target_file)
    put = '.DS_Store'  # put:1.txt
    if put in result:  # 精确到后缀名
        print('Yes')
        os.remove(video_dir + put)
    else:
        print('No')

    '''Step 2.2: Capture frames 并保存到youtube_frame_dir
              
              2.2.1 Add all the file name to a list: file_name_list
              2.2.2 Make a file to store one video's all frames'''
    file_name_list = []
    for file in os.listdir(video_dir):
        file_name = os.fsdecode(file)
        # file_name = file_name.replace("\n", "")
        file_name_list.append(file_name)

    for fileName in file_name_list:
        each_vid_frame_dir = frame_dir + fileName.split('.mp4')[0] + '/'
        if not os.path.exists(each_vid_frame_dir):
            os.mkdir(each_vid_frame_dir)
        capture = cv2.VideoCapture(video_dir + fileName)
        c = 1
        frame_rate = 100
        while True:
            ret, frame = capture.read()
            if ret:
                if c % frame_rate == 0:
                    print("Start video capturing " + str(c) + " frame")
                    cv2.imwrite(each_vid_frame_dir + fileName.split(".mp4")[0] + '_' + str(c) + ".jpg", frame)
                c += 1
            else:
                print('All frames captured.')
                break
        capture.release()


'''
Step 3: Extract features of each frame as a vector, 
store each vector in a sequence, 
store the sequence as a npy file in the feature_dir
'''


def get_model():
    K.clear_session
    inception_base = InceptionV3(weights='imagenet', include_top=True)
    return Model(inputs=inception_base.input,
                 outputs=inception_base.get_layer('avg_pool').output)


def feature_extraction(frame_dir, feature_dir):
    model = get_model()
    delete_ds_file(frame_dir)
    vid_name_list = []
    for vid_name in os.listdir(frame_dir):
        filename = os.fsdecode(vid_name)
        vid_name_list.append(filename)
        # print(filename) dsojsodjf
    frame_dir_name_list = []
    for vid_name in vid_name_list:
        filename = frame_dir + vid_name + "/"
        frame_dir_name_list.append(filename)
        # print(filename)  /Users/lxy/PycharmProjects/new/dataset/frame/w22oOw3RIAk/

    imgs = []
    for x in frame_dir_name_list:
        a = os.fsdecode(x)
        b = os.listdir(x)
        # print(os.fsdecode(x)) # /Users/lxy/PycharmProjects/new/dataset/frame/w22oOw3RIAk/
        # print(os.listdir(x))  #'w22oOw3RIAk\n_200.jpg'
        # print(a + b) #error
        for y in os.listdir(x):
            img_name = os.fsdecode(y)
            imgs.append(img_name)

    for x in frame_dir_name_list:
        sequence = []
        for y in imgs:
            image_path = x+y
            img = cv2.imread(image_path)  # Read BGR of the picture
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img4 = cv2.resize(img, (299, 299))
            cv2.imwrite(image_path, img4)


            image = tf.keras.utils.load_img(image_path, target_size=None)
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            predictions = model.predict(input_arr)

            feature_save_path = feature_dir + y.split(".jpg")[0] + "/"
            if not os.path.exists(feature_save_path):
                os.mkdir(feature_save_path)
            np.save(feature_save_path + "feature", predictions)
            # sequence.append(predictions)





if __name__ == '__main__':

    set_up_directory_layout()
    video_dir = "/Users/lxy/PycharmProjects/new/dataset/video/"
    frame_dir = "/Users/lxy/PycharmProjects/new/dataset/frame/"
    feature_dir = "/Users/lxy/PycharmProjects/new/dataset/feature/"
    # download_youtube_videos(video_dir)
    # frame_extraction(video_dir,frame_dir)
    # delete_ds_file(frame_dir)
    feature_extraction(frame_dir,feature_dir)







