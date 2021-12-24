import numpy as np
import pandas as pd
import PIL
from PIL import Image
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
import random
import multiprocessing as mp
from keras_vggface.vggface import VGGFace

import tensorflow as tf

import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json

#memorability-scores.xlsx path
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"
#images
face_images = "faces/10k US Adult Faces Database/Face Images/"


def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

model = VGGFace(model='senet50')
outputs=model.layers[-2].output
output = Dense(1)(outputs)
#fc = Dense(4096)(outputs)
# define new model
model = Model(inputs=model.inputs, outputs=output)
print(model.summary())

def setup_to_finetune(model, N_freeze_layer):
   """Freeze the bottom N_freeze_layers and retrain the remaining top 
      layers.
   """
   for layer in model.layers[:N_freeze_layer]:
      layer.trainable = False
   for layer in model.layers[N_freeze_layer:]:
      layer.trainable = True
      


def load_split(split_file):
    faces_ds = pd.read_excel(split_file)
    X_train, X_test= train_test_split(faces_ds, test_size=0.2, random_state=42)
    scores_train = list(X_train['Hit Rate (HR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'])
    names_test = list(X_test['Filename'])
    FA_test = list(X_test['False Alarm Rate (FAR)'])
    FA_train = list(X_train['False Alarm Rate (FAR)'])
    X_train = [[names_train[i],scores_train[i]-FA_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i],scores_test[i]-FA_test[i]] for i in range(len(scores_test))]
    X_valid = X_test[:int(len(X_test)/2)]
    X_test = X_test[int(len(X_test)/2):]
    return X_train, X_test, X_valid

# Function to load a single image
def load_image(image_file):
    image = Image.open(image_file).resize((224,224)).convert("RGB")
    if random.uniform(0, 1) > 0.5: 
      # random mirroring
      return np.array(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
    else:
      return np.array(image, dtype=np.uint8)


# Function that yields random samples of 10k us face database
def lamem_generator(split_file, batch_size):
    while True:
        random_files = random.sample(split_file, batch_size)
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])
		final_labels = [[i[1]] for i in random_files]
        yield(np.array(inputs_1),np.array(final_labels))

def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))
    
train_split, test_split, valid_split = load_split(mem_score_xlsx)
    
batch_size = 64 

# Create the training & testing data generators
train_gen = lamem_generator(train_split, batch_size=batch_size)
valid_gen = lamem_generator(valid_split, batch_size=batch_size)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="senet_hf/final/epoch_{epoch}.h5"),
    tf.keras.callbacks.CSVLogger("/senet_hf/final/faces_ft.log", separator=",", append=False),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)]
    
my_opt = tf.keras.optimizers.Adam(0.0001)
model.compile(my_opt, euclidean_distance_loss)
model.fit_generator(train_gen, steps_per_epoch=int(len(train_split) / batch_size), epochs=50, verbose=1, 
                    validation_data=valid_gen, validation_steps=int(len(valid_split) / batch_size), callbacks = callbacks)




