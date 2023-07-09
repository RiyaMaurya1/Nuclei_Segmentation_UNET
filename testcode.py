# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import tenorflow.keras.savings as sav

model = load_model("Unet_model_nuclie_segmentation.h5")

Img_height = 128
Img_width = 128
color_channel = 3

# img = "Nuclie_dataset/stage1_test/0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925/images/0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925.png"


def do_pred(img):
    X_test = np.zeros((1, Img_height, Img_width, color_channel),dtype=np.uint8)

    img = imread(img)[:,:,:color_channel]
    img = resize(img, (Img_height, Img_width), 
                mode = "constant", preserve_range=True)

    X_test[0] = img
    
    imshow(X_test[0])
    pred_mask = model.predict(X_test)
    # imshow(pred_mask.reshape(128,128))
    return pred_mask

