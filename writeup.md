# Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_12_0.png "uneven traffic sign type distribution"
[image2]: ./output_10_0.png "traffic sign sample from each category"
[image3]: ./output_19_0.png "preprocessed traffic sign sample from each category"
[image4]: ./output_31_0.png "new test images (Germany)"
[image5]: ./output_32_0.png "new test images (International)"
[image6]: ./output_39_1.png "Errors (Germany)"



## Rubric Points

### Data Set Summary & Exploration

The training set has 34,799 road signs, the test set has 12,630 (Code cell 2).
Those images are very unevenly distributed over the 43 categories as shown in the histogram below.

![Histogram (from code cell 6)][image1]

The uneven distribution is also true for the validation and test sets. This bears the problem that for pure probabiliy reasons the model will inevitably lean towards the more numerous traffic signs as opposed to the rare ones with less than 250 training samples.

Looking at the actual image data, all images have a size of `32`x`32` pixels and are color images. The size of the traffic sign within the image varies, but the most obvious difference is the brightness of the photos (see code cell 5). While some are very well lit, some other are almost completely black. In total, there are 43 different kinds of traffic signs.

![Traffic sign categories (from code cell 5)][image2]

### Design and Test a Model Architecture

#### Preprocessing

Before feeding the images to the neural network, the input data needs to be normalized. As the images consists of pixels with 3 color channels, each containing values between `0` and `255`, they need to be normalized so that their absolute values are smaller than `1` and their mean value is around `0`. I decided to normalize the values to a range of `[-0.5, +0.5]` with the help of the OpenCV function `cv2.normalize` using the parameter `cv2.NORM_MINMAX`, which happens to help with unbalanced brightness in an image as well. The result is that very dark images are brightened. All the preprocessing is done in code cell 7 and 8.

Here's what the images look like after preprocessing, and for comparison again the original images before preprocessing:

After:
![After Preprocessing][image3]

Before:
![Before Preprocessing][image2]

I decided not to convert to grayscale as I felt that colors do provide valuable information when classifying traffic signs, especially in low resolutions like the 32x32 pixels the images are provided in. Such cases would for example be the "general caution" and "traffic light ahead" signs, both triangular with red borders and white backgrounds. But from afar, without color, indistinguishable. Only the 3 colors representing the traffic light set it apart from the black exclamation mark in the center. Other such similar traffic signs exist in Germany, albeit not included in this data set.

#### Augmentation

I saw the suggestion to use augmentation on the review page. However, I decided not to include that step as it seems fragile to me, i.e. introducing gray or black borders when rotating, etc., and because the current solution provided > 93% accuracy for all data set segments.

#### Model

As the project introduction in the classroom suggested starting with LeNet, I did just that by using the LeNet solution we had from the previous lab. Initially results were not as good as I had anticipated, with epoch 1 validation accuracy being mostly below 50%. Up to 15 epochs usually yielded an accuracy of about 70-80%. By tweaking some parameters I managed to squeeze 90-91% out of it, but still not stable and good enough for the required > 93%. I then introduced another convolutional layer and adjusted the layer dimensions. Another valuable change was reducing sigma `σ` from `0.1` to `0.05`. The default learning rate of `0.001` worked well for me.

I removed any pooling as I had the feeling it reduced too much data for the little data each 32x32 image had and, somewhere in the process, decreased the accuracy. Another modification I made, was to use dropouts in two fully connected layers, in order to improve the reliability of the model. (code cell 10)

This resulted in the following architecture.

| Layer         	      	|     Description	                         					| 
|:---------------------:|:---------------------------------------------:| 
| Input         	      	| 32x32x3 RGB image                      							| 
| Convolution 3x3      	| 1x1 stride, valid padding, outputs 28x28x16	  |
| RELU			             		|												                                   |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 14x14x32			|
| RELU			             		|												                                   |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 10x10x64			|
| RELU			             		|												                                   |
| Fully connected		     | flattened previous, outputs 1x6400   									|
| Fully connected		     | outputs 1x120    								                    	|
| RELU			             		|												                                   |
| dropout             		|	keep 75%			                                   |
| Fully connected		     | outputs 1x84     								                    	|
| RELU			             		|												                                   |
| dropout             		|	keep 75%			                                   |
| Fully connected		     | result layer, outputs 1x43     								       |


I didn't need to split training, validation and test data anymore as they already came provided seperately in three different Pickle files.

The training of the model happens in code cell 14. I picked only 5 epochs, because the model did not improve considerably beyond that. As for the batch size, I used `128`, which gave good results. Increasing the size would speed up learning, but gave worse results. Batch sizes like `64` were slower, but did not improve the accuracy, so the final `128` were a good balance between the learning speed and a high accuracy.

I checked the model's accuracy with the `evaluate` function from the classroom.

My final model results (using code cell 13 and 14) were:
* training set accuracy of 0.992
* validation set accuracy of 0.963
* test set accuracy of 0.945

### Test a Model on New Images

#### Image Selection

I decided to test my model with 15 images of German traffic signs found on Google Image Search and Goole Street View. 

![German test images][image4]

Furthermore, for the fun of it, I wanted to see whether my model would be able to recognize untrained traffic signs. For that, I downloaded 10 more traffic sign photos from the UK, US, China, Japan and an electronic traffic sign from Germany. I picked signs that belong to the categories trained and, for a human driver, would be recognizable even when encountered for the first time.

![International test images][image5]

All images were cropped and resized by me. But, apparently, they mostly fill out the frame, whereas the training and test data's traffic signs are usually smaller compared to the 32x32 frame. Additionally, some of my images are a bit tilted whereas upon first glance the provided dataset contains frontal images.

#### Prediction Results

As calculated in code cell 21, the accuracy of the model is as follows:

```
86.7% correct (Germany)
50.0% correct (International)
```

The wrongly classified German traffic signs are, quite surprisingly (code cell 22):

![Errors (Germany)][image6]

My guess is that the left sign (speed limit 50km/h) is photographed from the lower left and therefore a bit distorted. The right sign (speed limit 70km/h) may be too tighly cropped.

As for the international traffic signs, I am positively surprised that a 50% accuracy was achieved.

In code cell 24, a bar chart displays the top 5 probability distributions of the German traffic signs I tested. The erroneous ones are marked with a ❌. The model generally favors one traffic sign class by a large margin. In three cases, though, two of which were wrongly predicted, the probability is below 5% for each class. The roundabout was ultimately correct.

```
❌ Web image for [ 2] "Speed limit (50km/h)" had the following probability distribution:
	██                       [ 5] Speed limit (80km/h) = 5.3%
	█                        [36] Go straight or right = 3.0%
	█                        [ 3] Speed limit (60km/h) = 2.4%
	                         [20] Dangerous curve to the right = -0.7%
	                         [23] Slippery road = -1.0%

❌ Web image for [ 4] "Speed limit (70km/h)" had the following probability distribution:
	█                        [ 1] Speed limit (30km/h) = 2.9%
	                         [ 8] Speed limit (120km/h) = 0.2%
	                         [ 5] Speed limit (80km/h) = -0.3%
	                         [ 4] Speed limit (70km/h) = -0.3%
	                         [ 0] Speed limit (20km/h) = -0.6%

Web image for [40] "Roundabout mandatory" had the following probability distribution:
	██                       [40] Roundabout mandatory = 4.8%
	█                        [ 7] Speed limit (100km/h) = 2.4%
	                         [ 5] Speed limit (80km/h) = 0.8%
	                         [11] Right-of-way at the next intersection = 0.7%
	                         [ 6] End of speed limit (80km/h) = 0.5%
```

The downloaded test images resulted in a 86.7% accuracy, which is lower than the provided test set's accuracy of 94.5%. As mentioned above, I suspect that the images I downloaded are cropped tigher around the traffic signs, which may impact the result. So to me it seems that the model I trained overfitted the provided data and hence yields lower accuracy provided different data. In any case, I wish there were more German traffic sign data sets with a more even distribution of traffic sign classes.
