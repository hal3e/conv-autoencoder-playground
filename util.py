import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    ''' Get MNIST training data'''
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels

    assert(len(X_train) == len(y_train))

    print("Image Shape: {}".format(X_train[0].shape))
    print("Training Set:   {} samples".format(len(X_train)))

    # Get 10 unique numbers from the validation set
    unique_labels, indices = np.unique(y_train, return_index=True)
    unique_x = X_train[indices]

    return X_train, y_train, unique_x


def show_numbers(images):
    f, ax = plt.subplots(1, len(images), figsize=(9,1))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(images[i].squeeze(), cmap='gist_gray')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def show_latent_space(lspace):
    f, ax = plt.subplots(figsize=(9,1))
    ax.imshow([lspace], cmap='Spectral')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def linear_interp(a, b, step = 10):
    assert a.shape == b.shape
    cc = np.zeros(shape=[step, a.shape[0]])
    for c, i in zip(np.linspace(0, 1, step), range(len(cc))):
        cc[i] = a + (b - a) * c

    return cc


def sample_latent_space(n_latent, step=10):
    zs = np.zeros(shape=[step,n_latent])
    for i in range(len(zs)):
        zs[i] = np.random.normal(0, 1, n_latent)

    return zs