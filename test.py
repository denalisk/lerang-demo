'''Demo program for tensorflow'''

import tensorflow as tf
import os as os
import matplotlib.pyplot as plt
from numpy import array as np_array
from skimage import data, transform, color

def load_data(data_directory):
    directories = [directory for directory in os.listdir(data_directory) 
    if os.path.isdir(os.path.join(data_directory, directory))]

    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = '''C:\\Users\\Skirsch\\Desktop\\Projects\\Tensor1\\Assets'''
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)


# Normalize the size of all the images to 28x28
images28 = [transform.resize(image, (28, 28)) for image in images]

# Convert the image28 variable back to an array and make all the images greyscale
images28 = np_array(images28)
images28 = color.rgb2gray(images28)


# Make a histogram
# plt.hist(labels, 62)

# plt.show()

### PUT IN COMMENTY BITS WITH VISUALIZING THE DATA

# \\================// The actual learning portion! \\====================//

# Here you could define a graph, but it seems unnecessary? 
# with tf.Graph().as_default()

# initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# flatten our 28x28 data into 1-dimensional array
images_flat = tf.contrib.layers.flatten(x)

# build out a neural network layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# define the loss function we will use to evaluate our model
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# define an optimizer to backpropagate our evaluations and adjust the model
train_op = tf.train.AdamOptimizer(learning_rate = .001).minimize(loss)

# take our logits and convert them back to label indexes
correct_prediction = tf.argmax(logits, 1)

# create a definition for an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Holder variables
RUN_RANGE = 201
LOSS_LOG_INTERVAL = 10

# Set up a session to run 
tf.set_random_seed(6666)
sesh = tf.Session()

sesh.run(tf.global_variables_initializer())

for i in range(RUN_RANGE):
    print('Iteration ', i)
    _, accuracy_val = sesh.run([train_op, accuracy], feed_dict = {x: images28, y: labels})
    if i % LOSS_LOG_INTERVAL == 0:
        print('Current loss: ', loss)
    print('Finished Iteration ', i)


# Things to do: 
# -Print out results
# -Run over test data
# -print out pretty bits along the way

# Slides:
# Basic problems with machine learning: representing the world as data
# Vectors and tensors
# Neural network
# Basic loss function and back propagation
# The goal of the sample neural net

# THen, out of commented code:
# random images of the signs
# histogram of the sign types
# Pictures of the normalization steps (square and greyscale)
# then, go over the code


