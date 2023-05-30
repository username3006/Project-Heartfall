import pyfirmata
import time
import ecg_plot
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

board = pyfirmata.Arduino('COM6')

it = pyfirmata.util.Iterator(board)
it.start()

board.analog[0].mode = pyfirmata.INPUT


def moving_average(a, n=100):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

beats = []
pbar = tqdm(total=1000)
plt.figure(figsize=(16, 5))
while len(beats) < 1000:
    val = board.analog[0].read()
    if val is None:
        continue
    beats.append(val)
    time.sleep(1 / 125)
    pbar.update(1)

beats = np.array(beats)
beats = moving_average(beats, n=100)

classes = {'normal': 0, 'AFLT': 1, 'BIGU': 2, 'PACE': 3, 'PSVT': 4, 'SARRH': 5, 'SBRAD': 6, 'AFIB': 7, 'STACH': 8, 'SVARR': 9, 'SVTAC': 10, 'TRIGU': 11}
classes = {i : k for k, i in classes.items()}

plt.plot(beats)
plt.savefig('ecg.png')

im = Image.open('ecg.png')
im = im.resize((150, 150))
img = np.asarray(im)
# img = np.expand_dims(img, 0)
img = np.expand_dims(img[...,:3], 0)

loaded_model = tf.keras.models.load_model('model.h5', compile=False)
predictions = loaded_model.predict(img)
print(classes[round(predictions[0][0])])