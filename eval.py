# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:19:23 2022

@author: Pushp jain
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir, tf_dataset

H = 512
W = 512

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    #create_dir("results")

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("D:/DESKTOP/Data Science Projects/Minor Project/Lung Segmentation/files/model.h5")

    dataset_path = "D:/DESKTOP/Data Science Projects/Minor Project/NLM-MontgomeryCXRSet/MontgomerySet"
    (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2) = load_data(dataset_path)

    for x, y1, y2 in tqdm(zip(test_x, test_y1, test_y2), total=len(test_x)):
        image_name = x.split("/")[-1]

        ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_x = cv2.resize(ori_x, (W, H))
        x = ori_x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        ori_y1 = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
        ori_y2 = cv2.imread(y2, cv2.IMREAD_GRAYSCALE)
        ori_y = ori_y1 + ori_y2
        ori_y = cv2.resize(ori_y, (W, H))
        ori_y = np.expand_dims(ori_y, axis=-1)
        ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)

        save_image_path = f"D:/DESKTOP/Data Science Projects/Minor Project/Lung Segmentation/results/{image_name}"
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

        sep_line = np.ones((H, 10, 3)) * 255

        cat_image = np.concatenate([ori_x, sep_line, ori_y, sep_line, y_pred*255], axis=1)
        cv2.imwrite(save_image_path, cat_image)