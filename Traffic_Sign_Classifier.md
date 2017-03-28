
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = (32, 32, 3)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
n_channels = 3

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
```


```python
classes = range(n_classes)
y_train_list = y_train.tolist()
examples = [ y_train_list.index(c) for c in classes ]
X_examples = [ X_train[i] for i in examples ]
```


```python
import math
from mpl_toolkits.axes_grid1 import ImageGrid

def display_traffic_signs(dataset):
    n_cols = 11 if len(dataset) > 10 else len(dataset)
    width = math.ceil(n_cols * 1.5)
    fig1 = plt.figure(1, (width, width * math.ceil(len(dataset)/n_cols)))
    grid1 = ImageGrid(fig1, 111,
                     nrows_ncols=(math.ceil(len(dataset)/n_cols), n_cols),
                     axes_pad=0.1,
                    )

    for index, img in enumerate(dataset):
        grid1[index].imshow(img)
        
display_traffic_signs(X_examples)
```


![png](output_10_0.png)


### Histogram of Labels


```python
def histogram(data, bins):
    plt.figure(figsize=(16,6))
    plt.hist(data, bins=bins, rwidth=.8)
    plt.ylabel('count')
    plt.xlabel('class')

histogram(y_train, n_classes)
```


![png](output_12_0.png)


As can be seen from the histogram, the number of samples per category is not well distributed in our training data.

----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import numpy as np
import cv2

def normalize(img):
    '''normalizes pixel values from a [0,255] to [-0.5,+0.5] range'''
    result = np.zeros(shape=img.shape)
    result = cv2.normalize(img, result, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return result - 0.5

def preprocess(dataset):
    '''do the whole preprocessing by converting color space and normalizing values with one function call'''
    normalized = []
    for img in dataset:
        normalized.append(normalize(img))
    return np.array(normalized)
```


```python
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)
```


```python
def unpreprocess(dataset):
    return [ ((d + 0.5) * 255).astype(np.uint8) for d in dataset ]

X_examples = [ X_train[i] for i in examples ]
X_examples = unpreprocess(X_examples)
display_traffic_signs(X_examples)
```


![png](output_19_0.png)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

# LeNet taken from classroom + own modifications
def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    µ = 0
    σ = 0.05

    def conv_layer(num, x, dimensions, stride):
        _, h, w, d = x.shape.as_list()
        height, width, output_depth = dimensions
        
        filter_w = 1 + w - width
        filter_h = 1 + h - height
        conv_W = tf.Variable(tf.truncated_normal(shape=(filter_h, filter_w, d, output_depth), mean=µ, stddev=σ), name='conv{}_weights'.format(num))
        conv_b = tf.Variable(tf.zeros(output_depth), name='conv{}_bias'.format(num))
        conv = tf.nn.conv2d(x, conv_W, strides=(1, stride, stride, 1), padding='VALID', name='conv{}_layer'.format(num))
        conv = tf.add(conv, conv_b, name='conv{}_add'.format(num))
        conv = tf.nn.relu(conv, name='conv{}_act'.format(num))
        # removed pooling, without worked better for me
        # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return conv

    conv1 = conv_layer(1, x,     (28, 28, 16), 1) # Input = 32x32x3.  Output = 28x28x16
    conv2 = conv_layer(2, conv1, (14, 14, 32), 1) # Input = 28x28x16. Output = 14x14x32
    conv3 = conv_layer(3, conv2, (10, 10, 64), 1) # Input = 14x14x32. Output = 10x10x64

    fc0 = flatten(conv3)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(6400, 120), mean=µ, stddev=σ))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1, name='fc1')
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=µ, stddev=σ))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2, name='fc2')
    fc2    = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=µ, stddev=σ))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return (conv1, conv2, conv3, logits)
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

import tensorflow as tf

x = tf.placeholder(tf.float32, (None, 32, 32, n_channels), name='placeholder_x')
y = tf.placeholder(tf.int32, (None), name='placeholder_y')
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
```


```python
RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 128
KEEP_PROB = 0.75

conv_net_layers = LeNet(x)
logits = conv_net_layers[-1]
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
training_operation = optimizer.minimize(loss_operation)
```


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
from sklearn.utils import shuffle

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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} of {}...".format(i+1, EPOCHS))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic_signs')
    print("Model saved")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Training...
    
    EPOCH 1 of 5...
    Training Accuracy = 0.958
    Validation Accuracy = 0.925
    
    EPOCH 2 of 5...
    Training Accuracy = 0.980
    Validation Accuracy = 0.938
    
    EPOCH 3 of 5...
    Training Accuracy = 0.988
    Validation Accuracy = 0.958
    
    EPOCH 4 of 5...
    Training Accuracy = 0.994
    Validation Accuracy = 0.963
    
    EPOCH 5 of 5...
    Training Accuracy = 0.992
    Validation Accuracy = 0.963
    
    Model saved
    Test Accuracy = 0.945


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import numpy as np
import matplotlib.image as mpimg

def load_images_from_dir(folder):
    filenames = os.listdir(folder)
    # annoyance with macOS' Finder creating ".DS_Store" files
    filenames = list(filter(lambda fn: fn != '.DS_Store', filenames))
    labels = np.array([ int(filename.split('_')[0]) for filename in filenames ])
    return np.array([ mpimg.imread(folder + filename) for filename in filenames ]), labels

X_germany, y_germany = load_images_from_dir('images/de/')
X_international, y_international = load_images_from_dir('images/int/')
```


```python
display_traffic_signs(X_germany)
```


![png](output_31_0.png)



```python
display_traffic_signs(X_international)
```


![png](output_32_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

X_germany = preprocess(X_germany * 255)
X_international = preprocess(X_international * 255)
```

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

saver = tf.train.Saver()
predict = tf.argmax(logits, 1)
predict_top5 = tf.nn.top_k(logits, k=5)
top5_germany = None
top5_international = None
result_germany = None
result_international = None

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, './traffic_signs')
    top5_germany = sess.run(predict_top5, feed_dict={x: X_germany, keep_prob: 1})
    top5_international = sess.run(predict_top5, feed_dict={x: X_international, keep_prob: 1})
    result_germany = sess.run(predict, feed_dict={x: X_germany, keep_prob: 1})
    result_international = sess.run(predict, feed_dict={x: X_international, keep_prob: 1})
    print(result_germany, '<--->', y_germany)
    print(result_international, '<--->', y_international)
```

    [11 11 13 14 17 18 25 28  5 38 38  3 40 40  1] <---> [11 11 13 14 17 18 25 28  2 38 38  3 40 40  4]
    [13 13 13  2 22 23  1 29 12 21] <---> [13 13 13 14 22 23 25 26 40  8]



```python
import csv
signnames = {}
with open('signnames.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row) != 2 or not row[0].isdigit():
            continue
        c = int(row[0])
        signnames[c] = row[1]
```


```python
def diff(result, expected):
    '''finds diverging array elements'''
    positions = []
    for i, z in enumerate(zip(result, expected)):
        a, b = z
        if a != b:
            positions.append(i)
    return positions

def correctness(label, result, expected):
    '''calculates and prints percentage of matching array elements and returns array index of diverging ones'''
    d = diff(result, expected)
    print('{:.1f}% correct ({}), errors at: {}'.format((1 - len(d)/len(expected)) * 100, label, d))
    return d

d_germany = correctness('Germany', result_germany, y_germany)
d_international = correctness('International', result_international, y_international)
```

    86.7% correct (Germany), errors at: [8, 14]
    50.0% correct (International), errors at: [3, 6, 7, 8, 9]



```python
def print_errors(d, rs, ys):
    for i in d:
        print('mistook "{}" for a "{}"'.format(signnames[ys[i]], signnames[rs[i]]))

errors_germany = [ X_germany[i] for i in d_germany ]
display_traffic_signs(unpreprocess(errors_germany))
print_errors(d_germany, result_germany, y_germany)
```

    mistook "Speed limit (50km/h)" for a "Speed limit (80km/h)"
    mistook "Speed limit (70km/h)" for a "Speed limit (30km/h)"



![png](output_39_1.png)



```python
errors_international = [ X_international[i] for i in d_international ]
display_traffic_signs(unpreprocess(errors_international))
print_errors(d_international, result_international, y_international)
```

    mistook "Stop" for a "Speed limit (50km/h)"
    mistook "Road work" for a "Speed limit (30km/h)"
    mistook "Traffic signals" for a "Bicycles crossing"
    mistook "Roundabout mandatory" for a "Priority road"
    mistook "Speed limit (120km/h)" for a "Double curve"



![png](output_40_1.png)


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
for i, z in enumerate(zip(X_germany, y_germany, top5_germany.values, top5_germany.indices)):
    x, y, vals, inds = z
    error = '❌ ' if y != inds[0] else '' # mark errors
    print('{}Web image for [{:2d}] "{}" had the following probability distribution:'.format(error, y, signnames[y]))
    for v, j in zip(vals, inds):
        bars = '█' * int(v/2)
        print('\t{:25s}[{:2d}] {} = {:.1f}%'.format(bars, j, signnames[j], v))
    print()
    
print(top5_germany)
```

    Web image for [11] "Right-of-way at the next intersection" had the following probability distribution:
    	████████████████         [11] Right-of-way at the next intersection = 33.0%
    	█████                    [30] Beware of ice/snow = 10.2%
    	█                        [27] Pedestrians = 3.7%
    	█                        [18] General caution = 2.7%
    	                         [21] Double curve = -0.2%
    
    Web image for [11] "Right-of-way at the next intersection" had the following probability distribution:
    	███████████              [11] Right-of-way at the next intersection = 22.3%
    	██                       [18] General caution = 5.8%
    	██                       [30] Beware of ice/snow = 4.4%
    	█                        [27] Pedestrians = 2.9%
    	                         [26] Traffic signals = -2.9%
    
    Web image for [13] "Yield" had the following probability distribution:
    	████████████             [13] Yield = 25.6%
    	█                        [35] Ahead only = 3.6%
    	█                        [ 9] No passing = 3.2%
    	█                        [15] No vehicles = 2.9%
    	                         [25] Road work = 1.7%
    
    Web image for [14] "Stop" had the following probability distribution:
    	███████████████          [14] Stop = 30.7%
    	██                       [17] No entry = 5.8%
    	██                       [25] Road work = 4.9%
    	██                       [ 5] Speed limit (80km/h) = 4.0%
    	                         [ 3] Speed limit (60km/h) = 1.2%
    
    Web image for [17] "No entry" had the following probability distribution:
    	████████████████████     [17] No entry = 41.5%
    	████                     [14] Stop = 8.9%
    	█                        [ 0] Speed limit (20km/h) = 2.7%
    	                         [31] Wild animals crossing = 0.5%
    	                         [12] Priority road = 0.4%
    
    Web image for [18] "General caution" had the following probability distribution:
    	███████████              [18] General caution = 22.5%
    	█████                    [27] Pedestrians = 11.1%
    	███                      [26] Traffic signals = 7.6%
    	██                       [11] Right-of-way at the next intersection = 5.1%
    	                         [28] Children crossing = 1.1%
    
    Web image for [25] "Road work" had the following probability distribution:
    	██████                   [25] Road work = 12.9%
    	█                        [ 1] Speed limit (30km/h) = 3.2%
    	                         [18] General caution = 1.6%
    	                         [ 5] Speed limit (80km/h) = 1.2%
    	                         [11] Right-of-way at the next intersection = 0.6%
    
    Web image for [28] "Children crossing" had the following probability distribution:
    	███████                  [28] Children crossing = 15.6%
    	███                      [29] Bicycles crossing = 6.5%
    	                         [24] Road narrows on the right = 1.3%
    	                         [22] Bumpy road = 0.0%
    	                         [36] Go straight or right = -0.2%
    
    ❌ Web image for [ 2] "Speed limit (50km/h)" had the following probability distribution:
    	██                       [ 5] Speed limit (80km/h) = 5.3%
    	█                        [36] Go straight or right = 3.0%
    	█                        [ 3] Speed limit (60km/h) = 2.4%
    	                         [20] Dangerous curve to the right = -0.7%
    	                         [23] Slippery road = -1.0%
    
    Web image for [38] "Keep right" had the following probability distribution:
    	██████████████████████████████████[38] Keep right = 69.4%
    	████                     [34] Turn left ahead = 9.3%
    	████                     [36] Go straight or right = 8.3%
    	██                       [20] Dangerous curve to the right = 4.6%
    	                         [25] Road work = -5.2%
    
    Web image for [38] "Keep right" had the following probability distribution:
    	██████████████           [38] Keep right = 29.8%
    	██                       [34] Turn left ahead = 5.3%
    	██                       [36] Go straight or right = 4.0%
    	█                        [20] Dangerous curve to the right = 2.3%
    	                         [25] Road work = 1.3%
    
    Web image for [ 3] "Speed limit (60km/h)" had the following probability distribution:
    	███                      [ 3] Speed limit (60km/h) = 7.8%
    	██                       [ 5] Speed limit (80km/h) = 5.3%
    	█                        [36] Go straight or right = 3.3%
    	                         [35] Ahead only = 0.4%
    	                         [ 2] Speed limit (50km/h) = -1.1%
    
    Web image for [40] "Roundabout mandatory" had the following probability distribution:
    	██                       [40] Roundabout mandatory = 4.8%
    	█                        [ 7] Speed limit (100km/h) = 2.4%
    	                         [ 5] Speed limit (80km/h) = 0.8%
    	                         [11] Right-of-way at the next intersection = 0.7%
    	                         [ 6] End of speed limit (80km/h) = 0.5%
    
    Web image for [40] "Roundabout mandatory" had the following probability distribution:
    	█████████                [40] Roundabout mandatory = 18.8%
    	██                       [ 8] Speed limit (120km/h) = 4.2%
    	█                        [38] Keep right = 3.7%
    	                         [18] General caution = 0.8%
    	                         [37] Go straight or left = -0.1%
    
    ❌ Web image for [ 4] "Speed limit (70km/h)" had the following probability distribution:
    	█                        [ 1] Speed limit (30km/h) = 2.9%
    	                         [ 8] Speed limit (120km/h) = 0.2%
    	                         [ 5] Speed limit (80km/h) = -0.3%
    	                         [ 4] Speed limit (70km/h) = -0.3%
    	                         [ 0] Speed limit (20km/h) = -0.6%
    
    TopKV2(values=array([[  3.30248871e+01,   1.02083426e+01,   3.68725514e+00,
              2.72066951e+00,  -2.44085565e-01],
           [  2.23214378e+01,   5.78904915e+00,   4.40951061e+00,
              2.86835957e+00,  -2.94923711e+00],
           [  2.55759239e+01,   3.56149459e+00,   3.18440199e+00,
              2.94299769e+00,   1.66370988e+00],
           [  3.07285728e+01,   5.84630775e+00,   4.85900784e+00,
              4.04801941e+00,   1.20331931e+00],
           [  4.14990883e+01,   8.89694405e+00,   2.72331834e+00,
              5.35437286e-01,   4.40323293e-01],
           [  2.24827957e+01,   1.10563402e+01,   7.58152437e+00,
              5.10150766e+00,   1.14290392e+00],
           [  1.29028282e+01,   3.23334980e+00,   1.55917406e+00,
              1.20130789e+00,   5.82556844e-01],
           [  1.55929184e+01,   6.48433018e+00,   1.31732213e+00,
              7.55004212e-03,  -2.24898100e-01],
           [  5.34135532e+00,   3.01597023e+00,   2.44795060e+00,
             -7.27056742e-01,  -1.01828694e+00],
           [  6.93816071e+01,   9.25492382e+00,   8.26457691e+00,
              4.64256334e+00,  -5.15487719e+00],
           [  2.98265228e+01,   5.32497597e+00,   4.00307369e+00,
              2.28643417e+00,   1.31301486e+00],
           [  7.84087610e+00,   5.27362967e+00,   3.26444411e+00,
              3.66740882e-01,  -1.06014252e+00],
           [  4.77126169e+00,   2.39094234e+00,   8.42364430e-01,
              6.69439256e-01,   5.22003233e-01],
           [  1.88409824e+01,   4.19342327e+00,   3.71623564e+00,
              8.26615036e-01,  -5.22410795e-02],
           [  2.90787888e+00,   2.24912345e-01,  -2.88656622e-01,
             -3.16560149e-01,  -6.41943872e-01]], dtype=float32), indices=array([[11, 30, 27, 18, 21],
           [11, 18, 30, 27, 26],
           [13, 35,  9, 15, 25],
           [14, 17, 25,  5,  3],
           [17, 14,  0, 31, 12],
           [18, 27, 26, 11, 28],
           [25,  1, 18,  5, 11],
           [28, 29, 24, 22, 36],
           [ 5, 36,  3, 20, 23],
           [38, 34, 36, 20, 25],
           [38, 34, 36, 20, 25],
           [ 3,  5, 36, 35,  2],
           [40,  7,  5, 11,  6],
           [40,  8, 38, 18, 37],
           [ 1,  8,  5,  4,  0]], dtype=int32))


---

## Step 4: Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess, feed_dict={x: image_input, keep_prob: 1})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```


```python
with tf.Session() as sess:
    saver.restore(sess, './traffic_signs')
    img = np.array([ X_train[0] ])
    layer = tf.get_default_graph().get_tensor_by_name("conv1_layer:0")
    outputFeatureMap(img, layer)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-29-1a9b772f5d21> in <module>()
          3     img = np.array([ X_train[0] ])
          4     layer = tf.get_default_graph().get_tensor_by_name("conv1_layer:0")
    ----> 5     outputFeatureMap(img, layer)
    

    <ipython-input-25-28abe45665aa> in outputFeatureMap(image_input, tf_activation, activation_min, activation_max, plt_num)
         13     # Note: x should be the same name as your network's tensorflow data placeholder variable
         14     # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    ---> 15     activation = tf_activation.eval(session=sess, feed_dict={x: image_input, keep_prob: 1})
         16     featuremaps = activation.shape[3]
         17     plt.figure(plt_num, figsize=(15,15))


    TypeError: unhashable type: 'numpy.ndarray'


### Question 9

Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images


**Answer:**

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
