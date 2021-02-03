'''
Autor: Jazielinho
'''

import keyboard
from PIL import ImageGrab
import pyautogui
import numpy as np
import cv2
import tensorflow as tf
from training import config as config_tr
import config

model = None


def load_model() -> tf.keras.Model:
    ''' Cargando el modelo '''
    global model
    if model is None:
        model_file = open(config_tr.MODEL_PATH_JSON, 'r')
        model = model_file.read()
        model_file.close()
        model = tf.keras.models.model_from_json(model)
        model.load_weights(config_tr.MODEL_PATH_H5)
    return model


def prepara_imagen_array(img: np.ndarray) -> np.ndarray:
    ''' Convierte un array al formato permitido por el modelo '''
    img = np.array(img)
    img = cv2.resize(img, config_tr.SHAPE)
    img = np.reshape(img, (1, *config_tr.SHAPE, 3))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    return img


def saltar() -> None:
    ''' Saltar '''
    pyautogui.keyDown('space')
    pyautogui.keyUp('space')


def run() -> None:
    ''' Ejecutar '''
    global model
    model = load_model()

    while True:
        if keyboard.is_pressed('space'):
            break

    while True:
        img = ImageGrab.grab()

        img = prepara_imagen_array(img)

        prob = model.predict(img).ravel()[0]

        action = prob > config.THRESHOLD

        if action:
            saltar()

        if keyboard.is_pressed('escape'):
            break


if __name__ == '__main__':
    run()