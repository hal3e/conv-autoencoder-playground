# %% Import modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import tensorflow as tf
from util import get_data_label, show_numbers, show_latent_space, sample_latent_space, get_mesh_data, show_numbers_mesh, create_one_hot
# %%


# %% Get MNIST training data
X_train, y_train, unique_x, unique_c = get_data_label()

# Show unique numbers from the dataset
print("\nUnique numbers:")
show_numbers(unique_x)
# %%


# %% Build the model
n_latent = 3
n_clusters = 10

# Arguments used for tf.truncated_normal, the weights and biases initializer
mu_init = 0
sigma_init = 0.1

c1_strides = [1, 2, 2, 1]
padding = 'SAME'

c1_W = tf.Variable(tf.truncated_normal([3, 3, 1, 32], mean=mu_init, stddev=sigma_init))
c1_b = tf.Variable(tf.zeros(32))

c2_W = tf.Variable(tf.truncated_normal([3, 3, 32, 64], mean=mu_init, stddev=sigma_init))
c2_b = tf.Variable(tf.zeros(64))

c3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=mu_init, stddev=sigma_init))
c3_b = tf.Variable(tf.zeros(128))

mu_W = tf.Variable(tf.truncated_normal(shape = (3*3*128, n_latent), mean=mu_init, stddev=sigma_init))
mu_b = tf.Variable(tf.zeros(n_latent))

var_W = tf.Variable(tf.truncated_normal(shape = (3*3*128, n_latent), mean=mu_init, stddev=sigma_init))
var_b = tf.Variable(tf.zeros(n_latent))


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

    mu  = tf.matmul(conv3_f, mu_W) + mu_b
    var  = tf.matmul(conv3_f, var_W) + var_b

    # Sample latent space
    eps = tf.random_normal(tf.shape(var))
    z = mu + tf.multiply(eps, tf.exp(0.5 * var))

    return z, mu, var

ct1_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=mu_init, stddev=sigma_init))
ct1_b = tf.Variable(tf.zeros(64))

ct2_W = tf.Variable(tf.truncated_normal([3, 3,  32, 64], mean=mu_init, stddev=sigma_init))
ct2_b = tf.Variable(tf.zeros(32))

ct3_W = tf.Variable(tf.truncated_normal([3, 3,  1, 32], mean=mu_init, stddev=sigma_init))
ct3_b = tf.Variable(tf.zeros(1))

dd_W = tf.Variable(tf.truncated_normal(shape = (n_latent + n_clusters, 3*3*128), mean=mu_init, stddev=sigma_init))
dd_b = tf.Variable(tf.zeros(3*3*128))


def adecoder(ls):
    # Up-sampling
    batch_internal = tf.shape(ls)[0]

    # Dense
    ls = tf.matmul(ls, dd_W) + dd_b
    ls = tf.reshape(ls, [batch_internal, 3, 3, 128])

    ls = tf.nn.tanh(ls)

    conv_t_1 = tf.nn.conv2d_transpose(ls, ct1_W, [batch_internal, 7, 7, 64], [1, 2, 2, 1], 'VALID') + ct1_b

    # Activation.
    conv_t_1 = tf.nn.tanh(conv_t_1)

    # Up-sampling
    conv_t_2 = tf.nn.conv2d_transpose(conv_t_1, ct2_W, [batch_internal, 14, 14, 32], c1_strides, padding) + ct2_b

    # Activation.
    conv_t_2 = tf.nn.tanh(conv_t_2)

    # Up-sampling
    conv_t_3 = tf.nn.conv2d_transpose(conv_t_2, ct3_W, [batch_internal, 28, 28, 1], c1_strides, padding) + ct3_b

    out = tf.nn.sigmoid(conv_t_3)
    return out

x = tf.placeholder(tf.float32, (None, 28, 28, 1))
cluster = tf.placeholder(tf.float32, (None, n_clusters))

z, mu_, var_ = aencoder(x)
z_cluster = tf.concat([z, cluster], axis=1)

dec_image = adecoder(z_cluster)

z_sampled = tf.placeholder(tf.float32, (None, n_latent))

ls_sampled = tf.concat([z_sampled, cluster], axis=1)

dec_sampled = adecoder(ls_sampled)

latent_loss = -0.5 * tf.reduce_sum(1.0 + var_ - tf.square(mu_) - tf.exp(var_))
img_loss = tf.reduce_sum(tf.pow(dec_image - x, 2))
cost = tf.reduce_mean(img_loss + latent_loss)

learning_rate = tf.placeholder(tf.float32)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # construct an optimizer

sess = tf.Session()
saver = tf.train.Saver()

num_examples = len(X_train)
print("Model built!")
# %%


# Load model
saver.restore(sess, "./model/cvae/model.ckpt")


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
        l, _ = sess.run([cost, train_op], feed_dict={x: batch_x, cluster: batch_y, learning_rate: 0.001})
        loss_per_epoch += l / (float(num_examples)/BATCH_SIZE)

    d_img, r_img = sess.run([dec_image, dec_sampled], feed_dict={x: unique_x, cluster: unique_c, z_sampled: sample_latent_space(n_latent, step=10)})
    show_numbers(d_img)
    show_numbers(r_img)
    print("EPOCH {} ...".format(i+1))
    print("Loss = {:.3f}".format(loss_per_epoch))
    print()
# %%


# %% Save tf model
save_path = saver.save(sess, "./model/cvae/model.ckpt")
print("Model saved in file: %s" % save_path)
# %%


# %% Latent space visualization
lspace = sess.run(mu_, feed_dict={x: unique_x})

print(" latent space: 0")
show_latent_space(lspace[0])

print(" latent space: 3")
show_latent_space(lspace[3])

print(" latent space: 9")
show_latent_space(lspace[9])
# %%


# %% Random sample new images based on different clusters
for i in range(n_clusters):
    print(" samples cluster: {}".format(i))
    r_img = sess.run(dec_sampled, feed_dict={z_sampled: sample_latent_space(n_latent, step=10), cluster: create_one_hot(n_clusters, i, step=10)})
    show_numbers(r_img)
# %%


# %% Visualize samples from latent space
show_numbers_mesh(sess.run(dec_sampled, feed_dict={z_sampled: get_mesh_data(zero_axis=1), cluster: create_one_hot(n_clusters, 0, 15*15)}))

show_numbers_mesh(sess.run(dec_sampled, feed_dict={z_sampled: get_mesh_data(zero_axis=1), cluster: create_one_hot(n_clusters, 5, 15*15)}))

show_numbers_mesh(sess.run(dec_sampled, feed_dict={z_sampled: get_mesh_data(zero_axis=1), cluster: create_one_hot(n_clusters, 9, 15*15)}))
# %%


# %% Visualize latent distribution
rndperm = np.random.permutation(10000)
encode_x = X_train[rndperm]
encode_y = np.where(y_train[rndperm] == 1)[1]

encoded_mu = sess.run(mu_, feed_dict={x: encode_x})

f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(encoded_mu[:, 0], encoded_mu[:, 1], encoded_mu[:, 2], c=encode_y, cmap='Spectral', s=8)

ax2 = f.add_subplot(1, 2, 2)
sp2 = ax2.scatter(encoded_mu[:, 0], encoded_mu[:, 1], c=encode_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%


# %% Visualize two clusters
extract = np.where((encode_y == 9) | (encode_y == 0))
encoded_mu_c = encoded_mu[extract]
encode_y_c = encode_y[extract]
encoded_mu_c.shape
f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(encoded_mu_c[:, 0], encoded_mu_c[:, 1], encoded_mu_c[:, 2], c=encode_y_c, cmap='Spectral', s=8)

ax2 = f.add_subplot(1, 2, 2)
sp2 = ax2.scatter(encoded_mu_c[:, 0], encoded_mu_c[:, 1], c=encode_y_c, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%
