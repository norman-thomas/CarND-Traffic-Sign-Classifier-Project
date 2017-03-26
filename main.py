

import numpy as np
import cv2
import tensorflow as tf
import pickle

from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

print('Loading images...')
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)

image_shape = (32, 32, 3)

n_classes = len(set(y_train))
n_channels = 3

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

def convert_to_hsv(img):
    '''converts an image from RGB to HSV color space'''
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def increase_contrast(img):
    hsv = convert_to_hsv(img)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    clahe.apply(v)
    return cv2.merge((h, s, v))

def normalize(img):
    '''normalizes pixel values from a [0,255] to [-0.5,+0.5] range'''
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return (img - 0.5)

def preprocess(dataset):
    '''do the whole preprocessing by converting color space and normalizing values with one function call'''
    normalized = []
    for img in dataset:
        contrast = increase_contrast(img)
        norm = normalize(img)
        normalized.append(norm)
    return np.array(normalized)

print('Normalizing data...')
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

# LeNet taken from classroom
def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    µ = 0
    σ = 0.05

    def conv_layer(x, dimensions, stride, pool=False):
        print('prev conv layer shape:', x.shape.as_list())
        print('new shape:', dimensions)
        print('--')
        height, width, output_depth = dimensions
        _, h, w, d = x.shape.as_list()
        filter_w = 1 + w - width
        filter_h = 1 + h - height
        conv_W = tf.Variable(tf.truncated_normal(shape=(filter_h, filter_w, d, output_depth), mean=µ, stddev=σ))
        conv_b = tf.Variable(tf.zeros(output_depth))
        conv = tf.nn.conv2d(x, conv_W, strides=(1, stride, stride, 1), padding='VALID') + conv_b
        conv = tf.nn.relu(conv)
        if pool:
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return conv

    conv1 = conv_layer(x, (28, 28, 16), 1) # Input = 32x32x3.  Output = 28x28x16
    conv2 = conv_layer(conv1, (14, 14, 32), 1)     # Input = 28x28x16. Output = 14x14x32
    conv3 = conv_layer(conv2, (10, 10, 64), 1)     # Input = 14x14x32. Output = 

    fc0 = flatten(conv3)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(6400, 120), mean=µ, stddev=σ))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=µ, stddev=σ))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=µ, stddev=σ))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, n_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


rate = 0.001
BATCH_SIZE = 128
EPOCHS = 12

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")
