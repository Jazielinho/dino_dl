'''
Autor: Jazielinho
'''

import keyboard
from PIL import ImageGrab
import os
import tqdm
import random

from training import config_tr


class DataSet(object):
    ''' clase que crea dataset de entrenamiento '''

    saltar = 'saltar'
    nada = 'nada'
    reglas = [saltar, nada]
    formato = 'PNG'
    train = 'train'
    val = 'val'

    def __init__(self, val_split: int = 0.2) -> None:
        self.imagenes = []
        self.targets = []
        self.nombre_maximo = 0

        nombres_maximos = []
        for regla in DataSet.reglas:
            if not os.path.exists(config_tr.PATH_IMAGES + '/' + DataSet.train + '/' + regla):
                os.makedirs(config_tr.PATH_IMAGES + '/' + DataSet.train + '/' + regla)

            if not os.path.exists(config_tr.PATH_IMAGES + '/' + DataSet.val + '/' + regla):
                os.makedirs(config_tr.PATH_IMAGES + '/' + DataSet.val + '/' + regla)

            lista_imagenes = os.listdir(config_tr.PATH_IMAGES + '/' + DataSet.train + '/' + regla) + \
                             os.listdir(config_tr.PATH_IMAGES + '/' + DataSet.val + '/' + regla)
            if len(lista_imagenes) == 0:
                nombre_maximo = [0]
            else:
                maximo_nombre = [int(x.split('.' + DataSet.formato)[0]) for x in lista_imagenes]
                nombre_maximo = maximo_nombre
            nombres_maximos = nombres_maximos + nombre_maximo

        self.nombre_maximo = max(nombres_maximos)
        self.val_split = val_split

    def genera_datos(self) -> None:
        imagenes = []
        targets = []

        # Empieza a funcionar desde presionar espacio
        while True:
            if keyboard.is_pressed('space'):
                break

        while True:
            # Las imagenes estan en blanco y negro
            imagen = ImageGrab.grab()
            imagenes.append(imagen)

            if keyboard.is_pressed('escape'):
                break

            if keyboard.is_pressed('space') or keyboard.is_pressed('up'):
                targets.append(DataSet.saltar)
            else:
                targets.append(DataSet.nada)

        self.imagenes = imagenes
        self.targets = targets

        self.guardar_info()

    def guardar_info(self) -> None:
        ''' guardamos las imagenes '''
        for imagen, target in tqdm.tqdm(zip(self.imagenes, self.targets), total=len(self.imagenes)):
            self.nombre_maximo += 1
            random_ = random.random()
            if random_ <= 1 - self.val_split:
                image_PATH = config_tr.PATH_IMAGES + '/' + DataSet.train + '/' + target + '/' + str(self.nombre_maximo) + '.' + DataSet.formato
            else:
                image_PATH = config_tr.PATH_IMAGES + '/' + DataSet.val + '/' + target + '/' + str(self.nombre_maximo) + '.' + DataSet.formato
            imagen.save(image_PATH, DataSet.formato)


if __name__ == '__main__':
    self = DataSet()

    self.genera_datos()
