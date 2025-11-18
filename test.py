import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("dog_cat_classifier.h5")

img_path = r"D:\DogsCatsClassifier\test pic.jpg"   # change to your image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("Prediction: DOG")
else:
    print("Prediction: CAT")
