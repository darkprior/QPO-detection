# -*- coding: utf-8 -*-
#script for testing the CNN using light curve images
"""
Created on Mon Mar 13 10:44:08 2023

@author: Denis
"""
#%% unused libraries
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential,model_from_json
# from tensorflow.keras.callbacks import EarlyStopping
# import pickle
# from keras.preprocessing.image import ImageDataGenerator
# import pathlib
#%%

import numpy as np
# import PIL
import tensorflow as tf

  
def testCNN(modelName, samplesNo):
    
    TF_MODEL_FILE_PATH = modelName+'.tflite' # The default path to the saved TensorFlow Lite model
    
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    print(interpreter.get_signature_list())
    classify_lite = interpreter.get_signature_runner('serving_default')
    
    pod_68=0
    medzi_6895=0
    medzi_95997=0
    nad_997=0
    pp=[]
    pp2=[]
    for i in range(0,samplesNo):
        # print(i)
        img = tf.keras.utils.load_img(
              'testdata/4lor/lc/4lor_'+str(i+1)+'.png', target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions_lite = classify_lite(sequential_input=img_array)['outputs']
        score_lite = tf.nn.softmax(predictions_lite)
        vyslednaacc=100 * np.max(score_lite)
        print(i, vyslednaacc)
        if round(vyslednaacc,2) < 68:
            pod_68+=1
        if round(vyslednaacc,2) >= 68 and round(vyslednaacc,2) <= 95:
            medzi_6895+=1
        if round(vyslednaacc,2) > 95 and round(vyslednaacc,2) <= 99.7:
            medzi_95997+=1
            pp.append(i)
        if round(vyslednaacc,2) > 99.7:
            nad_997+=1
            pp2.append(i)
    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    # )
    print("pod 68: {}\nmedzi 68 95:{}\nmedzi 95 99.7: {}\nnad 99.7: {}"
          .format(pod_68,medzi_6895,medzi_95997,nad_997))
    
    return(pod_68,medzi_6895,medzi_95997,nad_997)

pocetPod68, pocetMedzi6895,pocet_medzi9597, pocet_nad997=testCNN('dizplcobr5000', 10000)