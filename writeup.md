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

The training of the model happens in code cell 14. I picked only 5 epochs, because the model did not improve considerably beyond that.

I checked the model's accuracy with the ` evaluate` function from the classroom.


---


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
