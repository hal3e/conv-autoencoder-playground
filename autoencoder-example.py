# %% Import modules
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
# %%

# %% Prepare data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print("Image Shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# Get 10 unique numbers from the validation set
unique_labels, indices = np.unique(y_validation, return_index=True)
unique_x = X_validation[indices]
# %%


# Function that shows input images
def show_numbers(images):
    plt.figure(figsize=(200, 200))
    f, ax = plt.subplots(1, len(images))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(images[i].squeeze(), cmap="gray")
    plt.show()


def show_numbers_ns(images):
    f, ax = plt.subplots(1, len(images))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(images[i], cmap="gray")
    plt.show()


def linear_interp(a, b, step = 10):
    assert a.shape == b.shape
    cc = np.zeros(shape=[step, a.shape[0], a.shape[1], a.shape[2]])
    for c, i in zip(np.linspace(0, 1, step), range(len(cc))):
        cc[i] = a + (b - a) * c

    return cc

# Show unique numbers from the dataset
show_numbers(unique_x)


# %% Build the model
# Arguments used for tf.truncated_normal, the weights and biases initializer
mu = 0
sigma = 0.1

c1_strides = [1, 2, 2, 1]
padding = 'SAME'

c1_W = tf.Variable(tf.truncated_normal([3, 3, 1, 5], mean=mu, stddev=sigma))
c1_b = tf.Variable(tf.zeros(5))

c2_W = tf.Variable(tf.truncated_normal([3, 3, 5, 10], mean=mu, stddev=sigma))
c2_b = tf.Variable(tf.zeros(10))

c3_W = tf.Variable(tf.truncated_normal([3, 3, 10, 15], mean=mu, stddev=sigma))
c3_b = tf.Variable(tf.zeros(15))


def aencoder(x):
    # Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x5.
    conv1 = tf.nn.conv2d(x, c1_W, c1_strides, padding) + c1_b

    # Activation.
    conv1 = tf.nn.tanh(conv1)

    # Layer 2: Convolutional. Output = 10x10x10.
    conv2 = tf.nn.conv2d(conv1, c2_W, c1_strides, padding) + c2_b

    # Activation.
    conv2 = tf.nn.tanh(conv2)

    # Layer 3: Convolutional. Output = 10x10x10.
    conv3 = tf.nn.conv2d(conv2, c3_W, [1, 3, 3, 1], 'VALID') + c3_b

    # Activation.
    latent_space = tf.nn.tanh(conv3)

    return latent_space


ct1_W = tf.Variable(tf.truncated_normal([3, 3, 10, 15], mean=mu, stddev=sigma))
ct1_b = tf.Variable(tf.zeros(10))

ct2_W = tf.Variable(tf.truncated_normal([3, 3,  5, 10], mean=mu, stddev=sigma))
ct2_b = tf.Variable(tf.zeros(5))

ct3_W = tf.Variable(tf.truncated_normal([3, 3,  1, 5], mean=mu, stddev=sigma))
ct3_b = tf.Variable(tf.zeros(1))


def adecoder(ls):
    # Up-sampling
    batch_internal = tf.shape(ls)[0]
    conv_t_1 = tf.nn.conv2d_transpose(ls, ct1_W, [batch_internal, 7, 7, 10], [1, 3, 3, 1], 'VALID') + ct1_b

    # Activation.
    conv_t_1 = tf.nn.tanh(conv_t_1)

    # Up-sampling
    conv_t_2 = tf.nn.conv2d_transpose(conv_t_1, ct2_W, [batch_internal, 14, 14, 5], c1_strides, padding) + ct2_b

    # Activation.
    conv_t_2 = tf.nn.tanh(conv_t_2)

    # Up-sampling
    conv_t_3 = tf.nn.conv2d_transpose(conv_t_2, ct3_W, [batch_internal, 28, 28, 1], c1_strides, padding) + ct3_b

    # Activation.
    out = tf.nn.tanh(conv_t_3)

    return out


def autoencoder(x):
    return adecoder(aencoder(x))


x = tf.placeholder(tf.float32, (None, 28, 28, 1))
model = autoencoder(x)
encoder = aencoder(x)

lsp = tf.placeholder(tf.float32, (None, 2, 2, 15))
decoder = adecoder(lsp)

cost = tf.reduce_sum(tf.pow(model - x, 2))  # minimize squared error
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)  # construct an optimizer

sess = tf.Session()
saver = tf.train.Saver()

num_examples = len(X_train)
print("Model built!")
# %%

# Load model
saver.restore(sess, "./model/model.ckpt")

# %% Train the model
EPOCHS = 250
BATCH_SIZE = 256
sess.run(tf.global_variables_initializer())
print("Training...")
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        l, _ = sess.run([cost, train_op], feed_dict={x: batch_x})
    m, lspace = sess.run([model, encoder], feed_dict={x: unique_x})
    show_numbers(m)
    print("EPOCH {} ...".format(i+1))
    print("Loss = {:.3f}".format(l))
    print()
# %%

# %% Save tf model
save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in file: %s" % save_path)
# %%

# %% Latent space linear interpolation
m, lspace = sess.run([model, encoder], feed_dict={x: unique_x})
lin_intrp = linear_interp(lspace[0], lspace[5], 10)
show_numbers(sess.run(decoder, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[3], lspace[2], 10)
show_numbers(sess.run(decoder, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[3], lspace[8], 10)
show_numbers(sess.run(decoder, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[9], lspace[5], 10)
show_numbers(sess.run(decoder, feed_dict={lsp: lin_intrp}))
# %%

rndperm = np.random.permutation(10000)
decomp_x = X_train[rndperm].squeeze().reshape(len(rndperm), 28*28)
decomp_y = y_train[rndperm]

decomp_x.shape
decomp_y.shape

tsne = TSNE(n_components=2, perplexity=75, n_iter=750)
tsne_results = tsne.fit_transform(decomp_x)

pca_results = PCA().fit_transform(decomp_x)

# Save TSNE and PCA
with open('tsne-pca.pkl', 'wb') as handle:
    pickle.dump((tsne_results, pca_results, decomp_x, decomp_y), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load TSNE and PCA
with open('tsne-pca.pkl', 'rb') as handle:
    tsne_results, pca_results, decomp_x, decomp_y = pickle.load(handle)

# %%
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=decomp_y, cmap='Spectral', s=8)
sp2 = ax[1].scatter(pca_results[:, 0], pca_results[:, 1], c=decomp_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%


# Run it through the autoencoder
x_encoded = sess.run(encoder, feed_dict={x: X_train[rndperm]})
decomp_x = x_encoded.reshape(len(x_encoded), 60)
decomp_x.shape

tsne_results_2 = tsne.fit_transform(decomp_x)

pca_results_2 = PCA().fit_transform(decomp_x)

# Save TSNE_2 and PCA_2
with open('tsne-pca-2.pkl', 'wb') as handle:
    pickle.dump((tsne_results_2, pca_results_2, decomp_x, decomp_y), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load TSNE_2 and PCA_2
with open('tsne-pca-2.pkl', 'rb') as handle:
    tsne_results_2, pca_results_2, decomp_x, decomp_y = pickle.load(handle)

# %%
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].scatter(tsne_results_2[:, 0], tsne_results_2[:, 1], c=decomp_y, cmap='Spectral', s=8)
sp2 = ax[1].scatter(pca_results_2[:, 0], pca_results_2[:, 1], c=decomp_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%
