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

def sample_z(step = 10):
    zs = np.zeros(shape=[step, n_latent])
    for i in range(len(zs)):
        zs[i] = np.random.normal(0, 1, n_latent)

    return zs


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def sample_gumbel_np(shape, eps=1e-20):
    U = np.random.uniform(0, 1, size=shape)
    return -np.log(-np.log(U + eps) + eps)


def gumbel_softmax_sample_np(logits, temperature):
    y = sample_gumbel_np(logits.shape)
    y_t = y / temperature
    y_t_exp = np.exp(y_t)

    return y_t_exp / np.sum(y_t_exp)


def sample_y(step=10, temperature=0.5, hard=False):
    logits = np.zeros([step, n_classes])
    y = gumbel_softmax_sample_np(logits, temperature)
    if hard:
        y = np.rint(y)
    return y


def show_numbers_and_classes(lsp_mu, lsp_cs):
    for i in range(len(lsp_mu)):
        print(" latent space: {}".format(i))
        show_numbers_ls(lsp_mu[i])
        show_numbers_ls(lsp_cs[i])


def create_one_hot(n_classes, class_i, step=10):
    assert class_i > -1 and class_i < n_classes

    y_temp = np.zeros([n_classes])
    y_temp[class_i] = 1
    return np.tile(y_temp, [step, 1])


def create_one_hot_all_classes(n_classes, step=10):
    y = np.zeros([step, n_classes])
    for i in range(step):
        if step <= n_classes:
            y[i][i] = 1
        else:
            y[i][np.random.randint(5, size=1)[0]] = 1
    return y


def show_samples_from_classes():
    for i in range(n_classes):
        print(" samples class: {}".format(i))
        r_img = sess.run(dec_sampled, feed_dict={z_sampled: sample_z(), y_sampled: create_one_hot(n_classes, i)})
        show_numbers(r_img)


# # %%
# n_classes = 10
# a = np.round(sample_y(step=1, temperature=0.01),2)
# plt.bar(np.arange(n_classes), a.squeeze())
# plt.show()
# # %%

# Show unique numbers from the dataset
# show_numbers(unique_x)


# %% Build the model
tf.reset_default_graph()

n_latent = 32
n_classes = 10
w_gauss = 1.5

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

c_W = tf.Variable(tf.truncated_normal(shape = (3*3*128, n_classes), mean=mu_init, stddev=sigma_init))
c_b = tf.Variable(tf.zeros(n_classes))


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

    # Categorical sample
    c_logits =  tf.matmul(conv3_f, c_W) + c_b
    c_soft = tf.nn.softmax(c_logits)
    c_log = tf.log(c_soft + 1e-20)

    y = gumbel_softmax(c_logits, tau, hard=False)

    # Sample latenst space
    return z, y, mu, var, c_logits, c_soft, c_log


ct1_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=mu_init, stddev=sigma_init))
ct1_b = tf.Variable(tf.zeros(64))

ct2_W = tf.Variable(tf.truncated_normal([3, 3,  32, 64], mean=mu_init, stddev=sigma_init))
ct2_b = tf.Variable(tf.zeros(32))

ct3_W = tf.Variable(tf.truncated_normal([3, 3,  1, 32], mean=mu_init, stddev=sigma_init))
ct3_b = tf.Variable(tf.zeros(1))

zd_W = tf.Variable(tf.truncated_normal(shape = (n_latent, 3*3*128), mean=mu_init, stddev=sigma_init))
zd_b = tf.Variable(tf.zeros(3*3*128))

yd_W = tf.Variable(tf.truncated_normal(shape = (n_classes, 3*3*128), mean=mu_init, stddev=sigma_init))
yd_b = tf.Variable(tf.zeros(3*3*128))


def adecoder(z_ls, y_ls):
    # Up-sampling
    batch_internal = tf.shape(z_ls)[0]

    # Dense
    z_ls_expn = tf.matmul(z_ls, zd_W) + zd_b
    y_ls_expn = tf.matmul(y_ls, yd_W) + yd_b

    ls = z_ls_expn + y_ls_expn

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


tau = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, (None, 28, 28, 1))

z_, y_, mu_, var_, c_logits_, c_soft_, c_log_= aencoder(x)
dec_image = adecoder(z_, y_)

z_sampled = tf.placeholder(tf.float32, (None, n_latent))
y_sampled = tf.placeholder(tf.float32, (None, n_classes))
dec_sampled = adecoder(z_sampled, y_sampled)

gauss_KL = -0.5 * tf.reduce_sum(1.0 + var_ - tf.square(mu_) - tf.exp(var_)) * w_gauss
categorical_KL = tf.reduce_sum(c_soft_ * (c_log_ - tf.log(1.0 / n_classes)))


latent_loss = gauss_KL + categorical_KL
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
saver.restore(sess, "./model/ucvae/model.ckpt")

# %% Train the model
EPOCHS = 64
BATCH_SIZE = 256

decrease_tau = False
tau_ = 0.3
tau_decrease = EPOCHS * 0.5

# sess.run(tf.global_variables_initializer())
print("Training...")
for i in range(EPOCHS):
    if i <= tau_decrease and decrease_tau:
        tau_ = 1.0 - (i / tau_decrease) * 0.7
    X_train, y_train = shuffle(X_train, y_train)
    loss_per_epoch = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        l, _ = sess.run([cost, train_op], feed_dict={x: batch_x, learning_rate: 0.00001, tau: tau_})

        loss_per_epoch += l / (float(num_examples)/BATCH_SIZE)

    d_img, r_img = sess.run([dec_image, dec_sampled], feed_dict={x: unique_x, tau:0.05, z_sampled: sample_z(step=10), y_sampled: create_one_hot_all_classes(n_classes, step=10)})
    show_numbers(d_img)
    show_numbers(r_img)
    print("EPOCH: {}, Loss: {:.3f}, Tau: {:.3f}".format(i+1, loss_per_epoch, tau_))
    print()
# %%

# %% Save tf model
save_path = saver.save(sess, "./model/ucvae/model.ckpt")
print("Model saved in file: %s" % save_path)
# %%


# %% Latent space visualization
lsp_mu, lsp_cs = sess.run([mu_, c_soft_], feed_dict={x: unique_x})
show_numbers_and_classes(lsp_mu, lsp_cs)
# %%


# %% Random sample new images
show_samples_from_classes()
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
