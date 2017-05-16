#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visual.png "Visualization"
[image2]: ./original.png "Original"
[image3]: ./normalized.png "Normalized"
[image4]: ./rotatednormalized.png "Augmented Normalized"
[image5]: ./german/1.png "Traffic Sign 1"
[image6]: ./german/4.png "Traffic Sign 2"
[image7]: ./german/9.png "Traffic Sign 3"
[image8]: ./german/13.png "Traffic Sign 4"
[image9]: ./german/18.png "Traffic Sign 5"
[image10]: ./german/25.png "Traffic Sign 6"
[image11]: ./german/35.png "Traffic Sign 7"
[image12]: ./test_images.png "Probability"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bradbenvenuti/traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot of each class/label and an example image.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images to have mean zero and equal variance so that it is easier for the network to train.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]


I decided to generate additional data because the model seemed to be overfitting due to limited training data for certain classes.

I found the classes that needed more data and then took each of the training images for that label and rotated it with 3 random angles.

Here is an example of an original image and an augmented image:

![alt text][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          		|             									|
| Max Pooling			| 2x2 stride,  outputs 5x5x16         			|
| Flatten				|												|
| Fully Connected		|												|
| RELU		            |												|
| Dropout	          	|												|
| Fully Connected		|												|
| RELU              	|												|
| Dropout	          	|												|
| Fully Connected		|												|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with a learning rate of 0.001. I used 25 epochs with a batch size of 128.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.952
* test set accuracy of 0.938

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The architecture I chose was the LeNet architecture. This worked well for classifying the MNIST data, so it seemed like a good place to start.

* What were some problems with the initial architecture? I was initially getting validation accuracy of only ~80. So my model was overfitting.

* How was the architecture adjusted and why was it adjusted? I added normalization, dropout, and data augmentation to achieve a validation accuracy of >95.

* Which parameters were tuned? How were they adjusted and why? I tried different # epochs and adjusted the training rate in trial and error.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]
![alt text][image11]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image12]

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. This is not as good as the test set which had an accuracy of 94.1.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image12]

The model was very confident in the images that it labeled correctly, with a probability of >99%. For Image 5, it was also very confident, but had chosen the wrong image. For image 4, it was only 60% confident it had the correct label.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
