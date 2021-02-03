
import cv2
import numpy as np
import os
import random
from typing import Dict, Tuple, List, Generator
from joblib import delayed, Parallel
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def prepara_array(array: np.array, shape: Tuple[int, int], datagen: ImageDataGenerator) -> np.array:
    ''' Prepara un array usando un generador '''
    nuevo_array = cv2.resize(array, (shape[:2]))
    if datagen is not None:
        return datagen.random_transform(nuevo_array)
    return nuevo_array


def filename_to_array(directory: str, target: str, filename: str, shape: Tuple[int, int],
                      datagen: ImageDataGenerator) -> np.array:
    ''' retorna un array a partir de un archivo '''
    img_ref = load_img(directory + '/' + target + '/' + filename)
    array = img_to_array(img_ref)
    return prepara_array(array=array, shape=shape, datagen=datagen)


def flow_from_directory(directory: str, datagen_args: Dict, shape: Tuple[int, int], batch_size: int, shuffle: bool,
                        parallel: bool, njobs: int) -> Generator[np.array, np.array, None]:
    ''' devuelve un generador que devuelve arrays de imagenes con su target '''

    def fun_parallel(x: str) -> List[np.array]:
        return filename_to_array(directory=directory, target=dict_image_info[x]['target'], filename=x, shape=shape,
                                 datagen=datagen)

    datagen = ImageDataGenerator(**datagen_args)

    dict_image_info = {}
    for enum, target in enumerate(os.listdir(directory)):
        for filename in os.listdir(directory + '/' + target):
            dict_image_info[filename] = {'target': target,
                                         'target_int': enum}

    list_filenames = list(dict_image_info.keys())

    len_files = len(list_filenames)

    num_batches = len_files // batch_size

    while True:

        if shuffle:
            random.shuffle(list_filenames)

        for bid in range(num_batches):
            batch = list_filenames[bid * batch_size: (bid + 1) * batch_size]

            if parallel:
                array_list = Parallel(n_jobs=njobs)(delayed(fun_parallel)(x) for x in batch)
            else:
                array_list = [filename_to_array(directory=directory, target=dict_image_info[x]['target'], filename=x,
                                                shape=shape, datagen=datagen) for x in batch]

            yield np.array(array_list), np.array([dict_image_info[x]['target_int'] for x in batch])


if __name__ == '__main__':
    tr_generator = flow_from_directory(directory='D:/10_PUBLICACIONES/publicaciones/dino_dl/data/train',
                                       datagen_args={},
                                       shape=(224, 224),
                                       batch_size=32,
                                       shuffle=True,
                                       parallel=True,
                                       njobs=8)























