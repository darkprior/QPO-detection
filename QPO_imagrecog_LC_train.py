# -*- coding: utf-8 -*-
#script for training the CNN
"""
Created on Mon Mar 13 10:44:08 2023

@author: Denis
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import pickle


#preprocess function: create train, val datasets, load directory, set im dims and batch size
def preprocess(batchval, imheight,imwidth, valsplit,directory):

    data_dir = directory  # path to the directory where the images are stored
    batch_size = batchval
    img_height = imheight
    img_width = imwidth
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=valsplit,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=valsplit,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


    class_names = train_ds.class_names
    print(class_names)
    
    return(train_ds,val_ds,class_names)


#%%

train_ds,val_ds,class_names=preprocess(50,180,180,0.2,'trendata/lc5000')

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

#define an Autotune preprocessor and apply it to the splitted data

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

#%%


#see the augmentations
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")


#train the model, print the model to history
def train_CNN(epochsval, patienceval, deltaval, batchval):
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(180,
                                      180,
                                      3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
      ]
    )
    
    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      # layers.Dropout(0.2),
      layers.MaxPooling2D(),
      # layers.Dropout(0.1),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      # layers.Dropout(0.2),
      layers.MaxPooling2D(),
       layers.Conv2D(64, 3, padding='same', activation='relu'),
      # layers.Dropout(0.4),
       layers.MaxPooling2D(),
       layers.Dropout(0.4),
      layers.Flatten(),
        
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes,activation="softmax", name="outputs")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    custom_early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=patienceval, 
        min_delta=deltaval, 
        mode='max'
    )

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochsval,
      callbacks=[custom_early_stopping]
    )
    return (history, model)

history, model=train_CNN(300, 50, 0.001, 50)


#%%
# # Pickle the history to file and serialize and save the model
def dumpHistModel(historyName, modelName):
    
    (lambda history: pickle.dump(history.history, open(historyName, 'wb')))(history)
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(modelName+'.tflite', 'wb') as f:
      f.write(tflite_model)


dumpHistModel('historyModel5000_lc', 'model5000_lc')

#%%

def displayTrainmetrics():
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    print(np.mean(acc),np.mean(val_acc),np.mean(loss),np.mean(val_loss))
    epochs_range = range(len(history.epoch))
    
    plt.figure(figsize=(15, 7.5))
    plt.rcParams.update({'font.size': 12})
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    # plt.savefig('C:/Users/denis/Desktop/dizplcobr5000_25e_40b_r180x180.png',dpi=800)
    plt.show()

displayTrainmetrics()