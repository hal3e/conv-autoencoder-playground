# %% Import modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import tensorflow as tf
# %%

# %% Prepare data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels

assert(len(X_train) == len(y_train))

print("Image Shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))

# Get 10 unique numbers from the validation set
unique_labels, indices = np.unique(y_train, return_index=True)
unique_x = X_train[indices]
# %%


# Function that shows input images
def show_numbers(images):
    f, ax = plt.subplots(1, len(images))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(images[i].squeeze(), cmap="gray")
    plt.show()


def show_numbers_ls(lspace):
    plt.imshow([lspace], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def linear_interp(a, b, step = 10):
    assert a.shape == b.shape
    cc = np.zeros(shape=[step, a.shape[0]])
    for c, i in zip(np.linspace(0, 1, step), range(len(cc))):
        cc[i] = a + (b - a) * c

    return cc

def sample_ls(step = 10):
    zs = np.zeros(shape=[step,n_latent])
    for i in range(len(zs)):
        zs[i] = np.random.normal(0, 1, n_latent)

    return zs

# Show unique numbers from the dataset
show_numbers(unique_x)


# %% Build the model
n_latent = 3

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

    eps = tf.random_normal(tf.shape(var))
    z = mu + tf.multiply(eps, tf.exp(0.5 * var))

    # Sample latenst space
    return z, mu, var


ct1_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=mu_init, stddev=sigma_init))
ct1_b = tf.Variable(tf.zeros(64))

ct2_W = tf.Variable(tf.truncated_normal([3, 3,  32, 64], mean=mu_init, stddev=sigma_init))
ct2_b = tf.Variable(tf.zeros(32))

ct3_W = tf.Variable(tf.truncated_normal([3, 3,  1, 32], mean=mu_init, stddev=sigma_init))
ct3_b = tf.Variable(tf.zeros(1))

dd_W = tf.Variable(tf.truncated_normal(shape = (n_latent, 3*3*128), mean=mu_init, stddev=sigma_init))
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

z, mu_, var_ = aencoder(x)
dec_image = adecoder(z)

z_sampled = tf.placeholder(tf.float32, (None, n_latent))
dec_sampled = adecoder(z_sampled)

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
saver.restore(sess, "./model/vae/model.ckpt")

# %% Train the model
EPOCHS = 16
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

    d_img, r_img = sess.run([dec_image, dec_sampled], feed_dict={x: unique_x, z_sampled: sample_ls()})
    show_numbers(d_img)
    show_numbers(r_img)
    print("EPOCH {} ...".format(i+1))
    print("Loss = {:.3f}".format(loss_per_epoch))
    print()
# %%

# %% Save tf model
save_path = saver.save(sess, "./model/vae/model.ckpt")
print("Model saved in file: %s" % save_path)
# %%


# %% Latent space visualization
lspace = sess.run(mu_, feed_dict={x: unique_x})

print(" latent space: 0")
show_numbers_ls(lspace[0])

print(" latent space: 3")
show_numbers_ls(lspace[3])

print(" latent space: 9")
show_numbers_ls(lspace[9])
# %%

# %% Random sample new images
r_img = sess.run(dec_sampled, feed_dict={z_sampled: sample_ls()})
print(" samples: 1")
show_numbers(r_img)

r_img = sess.run(dec_sampled, feed_dict={z_sampled: sample_ls()})
print(" samples: 2")
show_numbers(r_img)

r_img = sess.run(dec_sampled, feed_dict={z_sampled: sample_ls()})
print(" samples: 3")
show_numbers(r_img)
# %%


rndperm = np.random.permutation(10000)
encode_x = X_train[rndperm]
encode_y = y_train[rndperm]

encode_x = sess.run(mu_, feed_dict={x: encode_x})

# %%
f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(encode_x[:, 0], encode_x[:, 1], encode_x[:, 2], c=encode_y, cmap='Spectral', s=8)

ax2 = f.add_subplot(1, 2, 2)
sp2 = ax2.scatter(encode_x[:, 0], encode_x[:, 1], c=encode_y, cmap='Spectral', s=8)
f.colorbar(sp2)
plt.show()
# %%
