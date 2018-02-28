import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    '''Get MNIST training data'''
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels

    assert(len(X_train) == len(y_train))

    print("Image Shape: {}".format(X_train[0].shape))
    print("Training Set:   {} samples".format(len(X_train)))

    # Get 10 unique numbers from the validation set
    unique_labels, indices = np.unique(y_train, return_index=True)
    unique_x = X_train[indices]

    return X_train, y_train, unique_x


def get_data_label():
    '''Get MNIST training data with labels'''
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)
    X_train, y_train = mnist.train.images, mnist.train.labels

    assert(len(X_train) == len(y_train))
    y_train = y_train.astype(np.float32)

    print("Image Shape: {}".format(X_train[0].shape))
    print("Training Set:   {} samples".format(len(X_train)))

    # Get 10 unique numbers from the validation set
    unique_x = X_train[[ 7,  4, 13,  1,  2, 27,  3,  0,  5,  8]]
    unique_c = y_train[[ 7,  4, 13,  1,  2, 27,  3,  0,  5,  8]]

    return X_train, y_train, unique_x, unique_c


def show_numbers(images):
    '''Show MNIST images'''
    f, ax = plt.subplots(1, len(images), figsize=(9,1))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # ax[i].imshow(images[i].squeeze(), cmap='gist_gray')
        ax[i].imshow(images[i].squeeze(), cmap='viridis')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def show_latent_space(lspace, color_map='Spectral'):
    '''Show latent space as images'''
    f, ax = plt.subplots(figsize=(9,1))
    ax.imshow([lspace], cmap=color_map)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def linear_interp(a, b, step=10):
    '''Create a linear interpolation between two latent vectors'''
    assert a.shape == b.shape
    cc = np.zeros(shape=[step, a.shape[0]])
    for c, i in zip(np.linspace(0, 1, step), range(len(cc))):
        cc[i] = a + (b - a) * c
    return cc


def sample_latent_space(n_latent, step=10):
    '''Sample latent space from a normal distribution'''
    zs = np.zeros(shape=[step,n_latent])
    for i in range(len(zs)):
        zs[i] = np.random.normal(0, 1, n_latent)
    return zs


def get_mesh_data(min_v=-3, max_v=3, steps=15, zero_axis=2):
    '''Create a mesh latent space representation'''
    data = np.zeros((steps * steps, 3))
    step = (max_v - min_v) / steps
    for i in range(steps):
        for j in range(steps):
            if zero_axis == 0:
                data[i * steps + j] = [0, min_v + i * step, min_v + j * step]
            elif zero_axis == 1:
                data[i * steps + j] = [min_v + i * step, 0, min_v + j * step]
            else:
                data[i * steps + j] = [min_v + i * step, min_v + j * step, 0]
    return data


def show_numbers_mesh(images, steps=15):
    '''Visualize the mesh latent space'''
    f, ax = plt.subplots(steps, steps, figsize=(8.8,9))
    for i in range(steps):
        for j in range(steps):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].imshow(images[i * steps + j].squeeze(), cmap="gray")
    f.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def create_one_hot(n_clusters, cluster_i, step=10):
    '''Create one hot encodings with specific size and current cluster'''
    assert cluster_i > -1 and cluster_i < n_clusters

    clusters_temp = np.zeros([n_clusters])
    clusters_temp[cluster_i] = 1
    return np.tile(clusters_temp, [step, 1])


def create_one_hot_all_clusters(n_clusters, step=10):
    '''Creat one hot encodings for each step'''
    clusters = np.zeros([step, n_clusters])
    for i in range(step):
        if step <= n_clusters:
            clusters[i][i] = 1
        else:
            clusters[i][np.random.randint(step, n_clusters, size=1)[0]] = 1
    return clusters


def sample_gumbel_np(shape, eps=1e-20):
    '''Sample from gumbel distribution'''
    U = np.random.uniform(0, 1, size=shape)
    return -np.log(-np.log(U + eps) + eps)


def gumbel_softmax_sample_np(logits, temperature):
    '''Sample from softmax gumbel distribution'''
    y = sample_gumbel_np(logits.shape)
    y_t = y / temperature
    y_t_exp = np.exp(y_t)

    return y_t_exp / np.sum(y_t_exp)


def sample_clusters(step=10, temperature=0.5, hard=False):
    '''Sample clusters use either softmax gumbel or hard'''
    logits = np.zeros([step, n_clusters])
    clusters = gumbel_softmax_sample_np(logits, temperature)
    if hard:
        clusters = np.rint(y)
    return clusters


def get_pretrain_data(X_train, y_train, n_cluster=10, n_ex_per_cluster=100):
    '''Create pre training data with defined numbers of clusters and examples per cluster'''
    rndperm = np.random.permutation(n_ex_per_cluster)
    for i in range(n_cluster):
        extract = np.where(np.all(y_train == create_one_hot(n_cluster, i, step=1), axis=1))[0]
        if i == 0:
            X_train_pretrain = (X_train[extract])[rndperm]
            y_train_pretrain = (y_train[extract])[rndperm]
        else:
            X_train_pretrain = np.vstack((X_train_pretrain, (X_train[extract])[rndperm]))
            y_train_pretrain = np.vstack((y_train_pretrain, (y_train[extract])[rndperm]))
    return X_train_pretrain, y_train_pretrain
