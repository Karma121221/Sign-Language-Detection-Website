from keras.models import load_model
from keras import backend as keras_backend
import numpy as np
from keras.preprocessing import image
import os

def predictor(image_path):
    keras_backend.clear_session()
    model_path = os.path.abspath('E:\GitHubRepo\Sign-Language-Detection-Website\model.h5')
    classifier = load_model(model_path)
    
    # Prediction of an image
    loaded_image = image.load_img(image_path, target_size=(63, 30))
    img_array = image.img_to_array(loaded_image)
    img_dims = np.expand_dims(img_array, axis=0)
    classifier_result = classifier.predict(img_dims)
    
    predicted_char = ''

    # Map to the character in the alphabet from one-hot encoding.
    for i in range(26):
        if classifier_result[0][i] == 1:
            predicted_char = chr(i + 65)

    keras_backend.clear_session()
    return predicted_char
