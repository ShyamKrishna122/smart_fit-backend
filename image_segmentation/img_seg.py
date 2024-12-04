import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

class ImageSegmentation:
    def __init__(self,dice_loss,dice_coef,iou):
        self.H = 512
        self.W = 512
        self.dice_loss = dice_loss
        self.dice_coef = dice_coef
        self.iou = iou

    def create_dir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def saveMask(self,path,img_name):
        with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef, 'dice_loss': self.dice_loss}):
            model = tf.keras.models.load_model("files/model.h5")
        name = img_name

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (self.W, self.H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        #cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imwrite(f"data/results/{name}.png", masked_image)
        mask_path = f"data/results/{name}.png"
        return mask_path


    def segmentation(self,path):
        """ Seeding """
        np.random.seed(42)
        tf.random.set_seed(42)

        """ Directory for storing files """
        self.create_dir("data/results")

        img_name = path.split("\\")[-1].split(".")[0] 
        img_path = path
        mask_path = self.saveMask(img_path,img_name)
        return mask_path