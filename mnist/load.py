import sys
sys.path.append('..')

import numpy as np
import os
import tensorflow as tf

data_dir = 'data/'
def mnist():
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    #trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY

call_path = os.path.join('data', 'na12878')

def call_labels():
    print('searching %s for labels' % call_path)
    labels = [
        label for label in os.listdir(call_path)
        if os.path.isdir(os.path.join(call_path, label))
    ]
    return labels


def call_set(labels, image_shape, batch_size):
    files = []
    onehots = {}
    for index, label in enumerate(labels):
        p = os.path.join(call_path, label)
        f = [os.path.join(p, f) for f in os.listdir(p)]
        h = [0.] * 1
        h[index] = 1.
        onehots[label] = h
        files += f
        print('found %d files for label %s' % (len(f), label))
    np.random.shuffle(files)
    queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()
    label, value = reader.read(queue)
    image = tf.image.decode_png(value)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, image_shape[:2])
    # image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    # image.set_shape(image_shape)
    image = tf.clip_by_value(image * 5, 0, 255)
    # images, labels = tf.train.batch(
    #     [image, label],
    #     batch_size=batch_size)
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=256 + 3 * batch_size,
        min_after_dequeue=256)
        # shapes=[image_shape, []])
    return labels, images, onehots
