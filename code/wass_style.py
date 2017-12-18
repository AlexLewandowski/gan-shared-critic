
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf
import numpy.random as npr

from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Uniform
from edward.models import Categorical, InverseGamma, Mixture, \
 Normal
from scipy.stats import multivariate_normal as mnormal
import tensorflow.contrib.distributions as tfd
import shutil

import glob
from PIL import Image

ed.set_seed(42)


# In[2]:

M = 64  # batch size during training
leak = 0.2  # leak parameter for leakyrelu
num_iter = 40000
DIR = "../../../data/bags2shoes"

IMG_DIR = "../plots/bags2shoes_shared_rev_5"

if os.path.exists(IMG_DIR):
    shutil.rmtree(IMG_DIR)

os.makedirs(IMG_DIR)
os.makedirs(IMG_DIR+'/trainA/')
os.makedirs(IMG_DIR+'/trainB/')
os.makedirs(IMG_DIR+'/trainBA/')
os.makedirs(IMG_DIR+'/trainAB/')
os.makedirs(IMG_DIR+'/trainABA/')
os.makedirs(IMG_DIR+'/trainBAB/')


# In[3]:

if len(glob.glob(DIR + '/*.npy')) == 0:
    filelist_1 = glob.glob(DIR + '/bags/*.jpg')
    filelist_2 = glob.glob(DIR + '/shoes/*.jpg')
    xs = np.array([np.array(Image.open(fname)) for fname in filelist_1])
    ys = np.array([np.array(Image.open(fname)) for fname in filelist_2])
    np.save(DIR + '/xs.npy', xs)
    np.save(DIR + '/ys.npy', ys)
else:
    xs = np.load(DIR + '/xs.npy')
    ys = np.load(DIR + '/ys.npy')


# In[4]:
print(os.curdir)
print(DIR)
print(glob.glob(DIR))


# In[4]:

y_ph = tf.placeholder(tf.float32, [M, 64, 64, 3])
x_ph = tf.placeholder(tf.float32, [M, 64, 64, 3])
phase = tf.placeholder(tf.bool)
drop = tf.placeholder(tf.float32)


# In[5]:

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    plt.title(str(samples))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    return fig


def leakyrelu(x, alpha=leak):
    return tf.maximum(x, alpha * x)


# In[6]:

def encoder(y, phase, drop):
    h = tf.layers.conv2d(y, 32, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, 64, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, 128, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 128, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 64, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 32, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 3, 5, padding='same')
    return tf.nn.tanh(h)


def decoder(x, phase, drop):
    h = tf.layers.conv2d(x, 32, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, 64, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, 128, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 128, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 64, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 32, 5, padding='same')
    h = slim.batch_norm(h, center=True, scale=True, is_training=phase)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d_transpose(h, 3, 5, padding='same')
    return tf.nn.tanh(h)


def discriminative_network(y):
    h = tf.layers.conv2d(y, 32, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 64, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 64, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 64, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 128, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 128, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.layers.conv2d(h, 128, 5, activation=leakyrelu, padding='same')
#    h = tf.layers.dropout(h,drop)
    h = tf.reshape(h, [M, -1])
    logit = slim.fully_connected(h, 1, activation_fn=None)
    return logit


with tf.variable_scope("Gen"):
    yf = decoder(x_ph, phase, drop)
    xf = encoder(y_ph, phase, drop)


# In[7]:

optimizer = tf.train.AdamOptimizer(0.00001, 0.5, 0.9)
optimizer_d = tf.train.AdamOptimizer(0.00001, 0.5, 0.9)
# optimizer = tf.train.RMSPropOptimizer(0.00005)
# optimizer_d = tf.train.RMSPropOptimizer(0.00005)


inference = ed.sharedWGANInference(
    data={yf: y_ph, xf: x_ph}, discriminator=discriminative_network)

inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer_d, n_iter=50000, n_print=100)


# In[8]:

# Initialize The Session
sess = ed.get_session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
i = 0
j = 0


# In[9]:

rndint_y = np.random.choice(len(ys), M, replace=False)
rndint_x = np.random.choice(len(xs), M, replace=False)
y_batch = ys[rndint_y, :, :, :]/127.5 - 1.
x_batch = xs[rndint_x, :, :, :]/127.5 - 1.

plot(x_batch[0:16, ])
