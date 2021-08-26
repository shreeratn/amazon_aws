import pickle

import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

default_image_size = tuple((256, 256))

loaded_model = pickle.load(open('cnn_model.pkl', 'rb'))
label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


im = convert_image_to_array('test1.png')
np_image_li = np.array(im, dtype=np.float16) / 225.0
npp_image = np.expand_dims(np_image_li, axis=0)

result = loaded_model.predict(npp_image)
print(result)

itemindex = np.where(result == np.max(result))
print("probability:" + str(np.max(result)) + "\n" + label_binarizer.classes_[itemindex[1][0]])
