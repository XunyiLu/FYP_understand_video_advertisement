import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import cv2


def get_model():
    K.clear_session
    inception_base = InceptionV3(weights='imagenet', include_top=True)
    return Model(inputs = inception_base.input,
                 outputs = inception_base.get_layer('avg_pool').output)

if __name__ == '__main__':
    # Step 1: Convert picture dimension to (299,299,3)
    image_path = '/Users/lxy/Desktop/1.jpg'
    new_image_path = '/Users/lxy/Desktop/3.jpg'
    img = cv2.imread(image_path) # Read BGR of the picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img4 = cv2.resize(img,(299,299))
    cv2.imwrite(new_image_path, img4)

    # Step 2: Build the model
    model = get_model()

    # Step 3: Extract features
    image = tf.keras.utils.load_img(new_image_path, target_size=None)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    # Step 4: Store the features
    feature_save_path = "/Users/lxy/Desktop/11111111"
    np.save(feature_save_path, predictions)