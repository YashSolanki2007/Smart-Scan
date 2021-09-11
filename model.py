# Imports
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Reshaping the training and testing data
train_img = ImageDataGenerator(rescale=1/255.0)
test_img = ImageDataGenerator(rescale=1/255.0)

# Making the list of class names
class_names = ["Covid", "Normal"]

# Making the training dataset
train_dataset = train_img.flow_from_directory("Covid19-dataset/train",
                                              target_size=(300, 300),
                                              batch_size=10,
                                              class_mode='binary')

# Making the testing dataset
testing_dataset = test_img.flow_from_directory("Covid19-dataset/test",
                                              target_size=(300, 300),
                                              batch_size=10,
                                              class_mode='binary')

# Getting the class indices
print(train_dataset.class_indices)

# Making the model
model = keras.Sequential([
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation='softmax')
])

# Loading the saved model
model = keras.models.load_model("covid-model-2")

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the model
# model.fit(train_dataset, epochs=10)

# Saving the model
model.save("covid-model-2/")

# Getting a model summary
model.summary()

# Evaluating the model on testing data
model.evaluate(testing_dataset)

# Getting the predictions
# test_dir = "Covid19-dataset/train/Normal"
# for i in os.listdir(test_dir):
#     img = image.load_img(test_dir + "//" + i , target_size=(300, 300))
#
#     X = image.img_to_array(img)
#     X = np.expand_dims(X, axis=[0])
#     images = np.vstack([X])
#
#     prediction = model.predict(images)
#     print(class_names[np.argmax(prediction)])


test_img = "test.png"
img = image.load_img(test_img, target_size=(300, 300))

X = image.img_to_array(img)
X = np.expand_dims(X, axis=[0])
images = np.vstack([X])

prediction = model.predict(images)
print(class_names[np.argmax(prediction)])
