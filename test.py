from keras.preprocessing import image
from keras.utils import load_img, img_to_array


from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np

#################################################################################################################################################

model = load_model("brain_tumor.h5") #CHANGE
print("Loaded model from disk")

#################################################################################################################################################

class_label = ["Benign", "Malign", "Normal"] #CHANGE alphabetically

#################################################################################################################################################

def load_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor = img_tensor/255
    return img_tensor

#################################################################################################################################################

img_path = "D:/projects/MachineLearning/HackJMI/model/DISEASES_TRAINED_MODELS/DISEASES/1BRAIN/brain_tumor/test/B_1.jpg" #CHANGE

loaded_image = load_image(img_path)
prediction = model.predict(loaded_image)
class_id = np.argmax(prediction, axis=1) #max value
print(class_label[int(class_id)]) #id wrt dataset

plt.imshow(loaded_image[0])
plt.axis('off')
plt.show()