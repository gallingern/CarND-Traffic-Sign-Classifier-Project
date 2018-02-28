# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization_train.png "Visualization Train"
[image2]: ./examples/visualization_valid.png "Visualization Validation"
[image3]: ./examples/visualization_test.png "Visualization Test"
[image4]: ./examples/equalize.png "Histogram Equalization"
[image5]: ./examples/grayscale.png "Grayscaling"
[image6]: ./examples/clahe.png "CLAHE"
[image7]: ./examples/test1.png "Test Images"
[image8]: ./examples/test2.png "Test Images Processed"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gallingern/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a series of bar charts showing how the data is distributed among the different classes/labels for the training, validation, and testing datasets.

![Training Data][image1]
![Validation Data][image2]
![Testing Data][image3]

While the size if the datasets differs, the distribution is relatively uniform.  This is an optimal condition that will aid our testing.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing

Before training my model, I used several preprocessing techniques to make it easier for my modal to fit my data.  I first tried the very basic approach of grayscale to reduce the color complexity from 3 to 1, and normalization to center the data around zero.  This improved my model some, but didn't achieve the accuracy I was looking for.  After some research, I discovered the scikit-image exposure library.  This library's functions histogram equalization and contrast limited adaptive histogram equalization (CLAHE) were big improvements.   These functions balance the lighting and enhance details, though when combined with normalization the accuracy was actually lower.  So in the end my preprocessing pipeline was the following:

 * Histogram Equalization
 * Grayscale
 * Contrast Limited Adaptive Histogram Equalization (CLAHE)
 * Shuffling (to randomize the data)

Here is an example of a traffic sign image after each step:

![Equalization][image4]
![Grayscaling][image5]
![CLAHE][image6]

##### Image Augmentation

I decided to generate additional data to improve training accuracy and prevent overfitting.

To add more data to the the data set, I experimented with the following techniques:
 * Rotation
 * Translation
 * Shearing

In the end I used only rotation, creating three additional copies of each image rotated a random amount between -5 and +5 degrees.  In my experimentation I found that more augmentation generally improved training accuracy, but also greatly increased training time.

I increased the number of training images from 34,799 to 139,196.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 grayscale image						| 
| Convolution			| 1x1 stride, valid padding, outputs 28x28x6	|
| ELU					| exponential linear unit						|
| Max pooling			| 2x2 stride, outputs 14x14x6					|
| Convolution			| 1x1 stride, valid padding, outputs 10x10x16	|
| ELU					| exponential linear unit						|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| Dropout				| 50%											|
| ELU					| exponential linear unit						|
| Fully connected		| outputs 84									|
| ELU					| exponential linear unit						|
| Fully connected		| outputs 43									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the example Lenet code which included the Adam Optimizer.  I played with tuning the hyperparameters but ended up coming back to close to the default of:

 * EPOCHS = 10
 * BATCH_SIZE = 128
 * rate = 0.002

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.4%
* test set accuracy of 93.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * I started with the Lenet example because the handwriting recognition problem was similar and it was an architecture that I already understood.
* What were some problems with the initial architecture?
    * After updating the dimensions, the only problem with the architecture was that the accuracy was too low.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * I changed the activation to ELU which I found increased my training accuracy.  I then played around with using dropout in different locations to prevent overfitting.  I ended up only using it once after the first fully connected layer at 50%.  Too much dropout decreased my accuracy too much.
* Which parameters were tuned? How were they adjusted and why?
    * Epochs: I increased epochs but found that the solution had converged by 10 epochs so there was no gain by increasing
    * Learning rate: I played with this a bit, I found that it was most effective between .002 and .001.  I settled on .002.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Convolution is an excellent choice for image processing because weights are reused.  Since images often have similar features throughout (edges), this makes convolutional layers particularly powerful.
    * Dropout layers are really useful in preventing overfitting by preventing your model from memorizing the data.

If a well known architecture was chosen:
* What architecture was chosen?
    * Lenet
* Why did you believe it would be relevant to the traffic sign application?
    * It was used as the handwriting classifying example and this is a similar application
* How does the final model's accuracy on the validation and test set provide evidence that the model is working well?
    * The drop from a validation to test accuracy of only 2% is evidence that the model is not overfit and works well on new data.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Test Images][image7]

And after processing:

![Test Images Processed][image8]

The 100 km/h sign has proved hard to classify.  It is often classified as a different speed limit, which makes sense since the are all numbers inside of circles.  Similarly the last three signs would often be confused with other triangular signs.  The stop sign is fairly distinct, but wasn't always perfectly classified in my testing.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image					| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)	| Speed limit (60km/h)							| 
| Stop					| Stop											|
| Bumpy road			| Bumpy road									|
| Road work				| Dangerous curve to the right					|
| Pedestrians			| Road work										|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This isn't great, but was also inconsistent when I would re-run it.  It would usually predict some kind of speed limit sign for the first sign, even if not the correct one, which is close.  And it would sometimes get the last two signs correct but didn't in my final run.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Image 1: Speed limit (100km/h) 100%

| Probability			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 100.0%				| Speed limit (60km/h)							| 
| 0.0%					| Speed limit (80km/h)							|
| 0.0%					| Speed limit (120km/h)							|
| 0.0%					| Vehicles over 3.5 metric tons prohibited		|
| 0.0%					| Ahead only									|

Certainty: 100%

The model is wrong here, though the top three are all similar looking speed limit signs.

Image 2: Stop 79%

| Probability			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 79.0%					| Stop											| 
| 17.6%					| Speed limit (70km/h)							|
| 2.1%					| Speed limit (60km/h)							|
| 1.2%					| Speed limit (80km/h)							|
| 0.1%					| Speed limit (120km/h)							|

Certainty: 79%

This one is correct and the model is fairly sure, though the next 4 guesses are all signs that are very dissimilar.

Image 3: Bumpy road 100%

| Probability			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 100.0%				| Bumpy road									| 
| 0.0%					| Bicycles crossing								|
| 0.0%					| Children crossing								|
| 0.0%					| No entry										|
| 0.0%					| Dangerous curve to the right					|

Certainty: 100%

Dead on here, no problems.

Image 4: Road work 0%

| Probability			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 90.8%					| Dangerous curve to the right					| 
| 8.9%					| No passing for vehicles over 3.5 metric tons	|
| 0.1%					| Priority road									|
| 0.1%					| Speed limit (80km/h)							|
| 0.0%					| Roundabout mandatory							|

Model is 90% certain, but wrong and the correct sign isn't in the top 5, not great.

Image 5: Pedestrians 20%

| Probability			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 42.6%					| Road work										| 
| 24.4%					| Road narrows on the right						|
| 20.0%					| Pedestrians									|
| 6.6%					| Beware of ice/snow							|
| 4.0%					| Wild animals crossing							|

Certainty: 20%

Model's third choice is correct with a low certainty, this is not a great fit.
