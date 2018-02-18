# %% Import modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from util import get_data, show_numbers, show_latent_space, linear_interp, sample_latent_space
# %%


# %% Get MNIST training data
X_train, y_train, unique_x = get_data()

# Show unique numbers from the dataset
print("\nUnique numbers:")
show_numbers(unique_x)
# %%


# %% Build the model
n_latent = 32

# Arguments used for tf.truncated_normal, the weights and biases initializer
mu_init = 0
sigma_init = 0.1

c1_strides = [1, 2, 2, 1]
padding = 'SAME'

c1_W = tf.Variable(tf.truncated_normal([3, 3, 1, 16], mean=mu_init, stddev=sigma_init))
c1_b = tf.Variable(tf.zeros(16))

c2_W = tf.Variable(tf.truncated_normal([3, 3, 16, 32], mean=mu_init, stddev=sigma_init))
c2_b = tf.Variable(tf.zeros(32))

c3_W = tf.Variable(tf.truncated_normal([3, 3, 32, 64], mean=mu_init, stddev=sigma_init))
c3_b = tf.Variable(tf.zeros(64))

de_W = tf.Variable(tf.truncated_normal(shape = (3*3*64, n_latent), mean=mu_init, stddev=sigma_init))
de_b = tf.Variable(tf.zeros(n_latent))


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
    conv3 = tf.nn.conv2d(conv2, c3_W, [1, 2, 2, 1], 'VALID') + c3_b

    conv3 = tf.nn.tanh(conv3)

    # Dense layer
    conv3_f = tf.contrib.layers.flatten(conv3)
    latent_space  = tf.matmul(conv3_f, de_W) + de_b

    return latent_space


ct1_W = tf.Variable(tf.truncated_normal([3, 3, 32, 64], mean=mu_init, stddev=sigma_init))
ct1_b = tf.Variable(tf.zeros(32))

ct2_W = tf.Variable(tf.truncated_normal([3, 3,  16, 32], mean=mu_init, stddev=sigma_init))
ct2_b = tf.Variable(tf.zeros(16))

ct3_W = tf.Variable(tf.truncated_normal([3, 3,  1, 16], mean=mu_init, stddev=sigma_init))
ct3_b = tf.Variable(tf.zeros(1))

dd_W = tf.Variable(tf.truncated_normal(shape = (n_latent, 3*3*64), mean=mu_init, stddev=sigma_init))
dd_b = tf.Variable(tf.zeros(3*3*64))


def adecoder(ls):
    # Up-sampling
    batch_internal = tf.shape(ls)[0]

    # Dense
    ls = tf.matmul(ls, dd_W) + dd_b
    ls = tf.reshape(ls, [batch_internal, 3, 3, 64])

    ls = tf.nn.tanh(ls)

    conv_t_1 = tf.nn.conv2d_transpose(ls, ct1_W, [batch_internal, 7, 7, 32], [1, 2, 2, 1], 'VALID') + ct1_b

    # Activation.
    conv_t_1 = tf.nn.tanh(conv_t_1)

    # Up-sampling
    conv_t_2 = tf.nn.conv2d_transpose(conv_t_1, ct2_W, [batch_internal, 14, 14, 16], c1_strides, padding) + ct2_b

    # Activation.
    conv_t_2 = tf.nn.tanh(conv_t_2)

    # Up-sampling
    conv_t_3 = tf.nn.conv2d_transpose(conv_t_2, ct3_W, [batch_internal, 28, 28, 1], c1_strides, padding) + ct3_b

    out = tf.nn.sigmoid(conv_t_3)
    return out


x = tf.placeholder(tf.float32, (None, 28, 28, 1))

lsp = aencoder(x)
dec_image = adecoder(lsp)

cost = tf.reduce_sum(tf.pow(dec_image - x, 2))  # minimize squared error

learning_rate = tf.placeholder(tf.float32)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # construct an optimizer

sess = tf.Session()
saver = tf.train.Saver()

num_examples = len(X_train)
print("Model built!")
# %%


# Load model
saver.restore(sess, "./model/ae/model.ckpt")


# %% Train the model
EPOCHS = 64
BATCH_SIZE = 256
# sess.run(tf.global_variables_initializer())
print("Training...")
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    loss_per_epoch = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        l, _ = sess.run([cost, train_op], feed_dict={x: batch_x, learning_rate: 0.0001})
        loss_per_epoch += l / (float(num_examples)/BATCH_SIZE)

    m = sess.run(dec_image, feed_dict={x: unique_x})
    show_numbers(m)
    print("EPOCH {} ...".format(i+1))
    print("Loss = {:.3f}".format(loss_per_epoch))
    print()
# %%


# %% Save tf model
save_path = saver.save(sess, "./model/ae/model.ckpt")
print("Model saved in file: %s" % save_path)
# %%


# %% Latent space visualization
lspace = sess.run(lsp, feed_dict={x: unique_x})

print(" latent space: 0")
show_latent_space(lspace[0])

print(" latent space: 3")
show_latent_space(lspace[3])

print(" latent space: 9")
show_latent_space(lspace[9])
# %%


# %% Latent space linear interpolation
lin_intrp = linear_interp(lspace[0], lspace[5], 10)
show_numbers(sess.run(dec_image, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[3], lspace[7], 10)
show_numbers(sess.run(dec_image, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[2], lspace[8], 10)
show_numbers(sess.run(dec_image, feed_dict={lsp: lin_intrp}))

lin_intrp = linear_interp(lspace[9], lspace[6], 10)
show_numbers(sess.run(dec_image, feed_dict={lsp: lin_intrp}))
# %%


# %% Latent space arithmetics
show_numbers(sess.run(dec_image, feed_dict={lsp: np.stack((lspace[0],lspace[4], lspace[9], lspace[6], lspace[7], lspace[5]))}))
show_numbers(sess.run(dec_image, feed_dict={lsp: np.stack((lspace[0] + lspace[4] - lspace[9], lspace[6] - lspace[7] + lspace[5]))}))
# %%


# %% Random sample new images
print(" samples: 1")
show_numbers(sess.run(dec_image, feed_dict={lsp: sample_latent_space(n_latent)}))

print(" samples: 2")
show_numbers(sess.run(dec_image, feed_dict={lsp: sample_latent_space(n_latent)}))

print(" samples: 3")
show_numbers(sess.run(dec_image, feed_dict={lsp: sample_latent_space(n_latent)}))
# %%


rndperm = np.random.permutation(10000)
decomp_x = X_train[rndperm].squeeze().reshape(len(rndperm), 28*28)
decomp_y = y_train[rndperm]

decomp_x.shape
decomp_y.shape

tsne = TSNE(n_components=2, perplexity=32, n_iter=750)
tsne_results = tsne.fit_transform(decomp_x)

pca_results = PCA().fit_transform(decomp_x)

# Save TSNE and PCA
with open('decomposition-data/ae-tsne-pca.pkl', 'wb') as handle:
    pickle.dump((tsne_results, pca_results, decomp_x, decomp_y), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load TSNE and PCA
with open('decomposition-data/ae-tsne-pca.pkl', 'rb') as handle:
    tsne_results, pca_results, decomp_x, decomp_y = pickle.load(handle)

# %%
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=decomp_y, cmap='Spectral', s=8)
sp2 = ax[1].scatter(pca_results[:, 0], pca_results[:, 1], c=decomp_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%


# Run it through the autoencoder
decomp_x = decomp_x.reshape(len(decomp_x), 28, 28, 1)
x_encoded = sess.run(lsp, feed_dict={x: decomp_x})
decomp_x = x_encoded.reshape(len(x_encoded), n_latent)

tsne_results_2 = tsne.fit_transform(decomp_x)

pca_results_2 = PCA().fit_transform(decomp_x)

# Save TSNE_2 and PCA_2
with open('decomposition-data/ae-tsne-pca-2.pkl', 'wb') as handle:
    pickle.dump((tsne_results_2, pca_results_2, decomp_x, decomp_y), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load TSNE_2 and PCA_2
with open('decomposition-data/ae-tsne-pca-2.pkl', 'rb') as handle:
    tsne_results_2, pca_results_2, decomp_x, decomp_y = pickle.load(handle)

# %%
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].scatter(tsne_results_2[:, 0], tsne_results_2[:, 1], c=decomp_y, cmap='Spectral', s=8)
sp2 = ax[1].scatter(pca_results_2[:, 0], pca_results_2[:, 1], c=decomp_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%
