Sixth Week
=====
[TOC]
10、Advice for Applying Machine Learning
------------------------------------------------------------

### 10.1  Deciding What to Try Next

When we use the trained model to predict unknown data, we find that there is a large error. What can we do next?

1. Obtain more training samples-usually effective, but the cost is higher, the following method may also be effective, consider using the following methods first.

2. Try to reduce the number of features

3. Try to get more features

4. Try to add polynomial features

5. Try to reduce the degree of regularization $\lambda$

6. Try to increase the degree of regularization $\lambda$

We should not randomly choose one of the above methods to improve our algorithm, but use some machine learning diagnostic methods to help us know which of the above methods are effective for our algorithm.

### 10.2  Evaluating a Hypothesis

![](../images/f49730be98810b869951bbe38b6319ba.png)

When we determine the parameters of the learning algorithm, we consider selecting parameters to minimize the training error. Some people think that it is a good thing to get a very small training error, but we already know that it is only because this hypothesis has a small training error does not mean that it must be a good hypothesis function. And we also learned examples of overfitting hypothesis functions, so this is not applicable to the new training set.

So, how can you tell if a hypothetical function is overfitting? For this simple example, we can plot the hypothetical function $h(x)$, and then observe the graphical trend, but for the general case of more than one feature variable, there are problems like there are many feature variables, and we want to pass drawing hypothetical functions to observe will become difficult or impossible.
	
Therefore, we need another method to evaluate our hypothesis function overfitting test.
In order to test whether the algorithm is overfitting, we divide the data into a training set and a test set, usually 70% of the data is used as the training set, and the remaining 30% of the data is used as the test set. It is very important that both the training set and the test set contain various types of data. Usually, we have to "shuffle" the data and then divide it into a training set and a test set.

![](../images/9c769fd59c8a9c9f92200f538d1ab29c.png)

Test set evaluation after letting our model learn its parameters through the training set, applying the model to the test set, we have two ways to calculate the error:

1. For the linear regression model, we use the test data to calculate the cost function $J$
2. For the logistic regression model, we can use the test data set to calculate the cost function:

The ratio of misclassification, for each test set sample, calculate:

![](../images/751e868bebf4c0bf139db173d25e8ec4.png)

Then average the calculation results.

### 10.3 Model Selection and Train_Validation_Test Sets

Suppose we want to choose between 10 binomial models of different degrees:

![](../images/1b908480ad78ee54ba7129945015f87f.jpg)

Obviously, the higher degree polynomial model can adapt to our training data set, but adapting to the training data set does not mean that it can be generalized to the general situation. We should choose a model that is more suitable for the general situation. We need to use a cross-validation set to help choose the model。
	
That is: use 60% of the data as the training set, 20% of the data as the cross-validation set, and 20% of the data as the test set

![](../images/7cf1cd9c123a72ca4137ca515871689d.png)

The method of model selection is:

1. Train 10 models using the training set

2. Use 10 models to calculate the cross-validation error (value of cost function) for the cross-validation set

3. Choose the model with the smallest cost function value

4. Use the model selected in step 3 to calculate the generalization error on the test set (the value of the cost function)

### 10.4  Diagnosing Bias vs. Variance 

When you run a learning algorithm, if the algorithm's performance is not ideal, then most of the two cases occur: either the deviation is relatively large, or the variance is relatively large. In other words, the situation is either under-fitting or over-fitting. So which of the two cases is related to deviation, which is related to variance, or is it related to both? It is very important to understand this point, because it is possible to determine which of these two situations is occurring. In fact, it is a very effective indicator, guiding the most effective method and way to improve the algorithm.

![](../images/20c6b0ba8375ca496b7557def6c00324.jpg)

We usually help the analysis by plotting the cost function error of the training set and the cross-validation set on the same chart as the degree of the polynomial:
![](../images/bca6906add60245bbc24d71e22f8b836.png)


![](../images/64ad47693447761bd005243ae7db0cca.png)

For the training set, when $d$ is smaller, the model fits lower and the error is larger; as $d$ increases, the fit increases and the error decreases.
The
For the cross-validation set, when $d$ is small, the model fits low and the error is large; but as $d$ increases, the error tends to decrease first and then increase, and the turning point is that our model begins to over-fit When combining training data sets.
The
If our cross-validation set has a large error, how do we determine whether it is variance or bias? According to the above chart, we know:

![](../images/25597f0f88208a7e74a3ca028e971852.png)

When training set error and cross-validation set error are approximate: deviation/underfitting
When the error of the cross-validation set is much larger than the error of the training set: variance/overfitting

### 10.5 Regularization and Bias_Variance 


In the process of training the model, we usually use some regularization methods to prevent overfitting. However, the degree of regularization may be too high or too small, that is, when we choose the value of λ, we also need to think about the problem similar to the degree of the polynomial model just selected.

![](../images/2ba317c326547f5b5313489a3f0d66ce.png)

We choose a series of $\lambda$ values that we want to test, usually values between 0-10 that show a double relationship (eg: $0,0.01,0.02,0.04,0.08,0.15,0.32,0.64,1.28, 2.56,5.12,10$ 12 in total). We also divide the data into training set, cross-validation set and test set.

![](../images/8f557105250853e1602a78c99b2ef95b.png)

The method to select $\lambda$ is:

1. Use training set to train 12 models with different degrees of regularization
2. Use 12 models to calculate the cross-validation error for the cross-validation set respectively
3. Choose the model with the smallest cross-validation error
4. Use the model selected in step 3 to calculate the generalization error for the test set. We can also plot the cost function error and the value of λ of the training set and cross-validation set models on a chart:

![](../images/38eed7de718f44f6bb23727c5a88bf5d.png)

• When $\lambda$ is small, the error of the training set is small (overfitting) and the error of the cross-validation set is large
The
• With the increase of $\lambda$, the training set error continues to increase (underfitting), while the cross-validation set error decreases first and then increases    

### 10.6  Learning Curves


The learning curve is a very good tool. I often use the learning curve to judge whether a learning algorithm is in the problem of deviation or variance. The learning curve is a good **sanity check** for learning algorithms. The learning curve is a graph that plots the training set error and cross-validation set error as a function of the number of training set samples ($m$).
	
That is, if we have 100 rows of data, we start with 1 row of data and gradually learn more rows of data. The idea is: when training fewer rows of data, the trained model will be able to adapt perfectly to less training data, but the trained model will not be able to adapt well to the cross-validation set data or test set data.

![](../images/969281bc9b07e92a0052b17288fb2c52.png)

![](../images/973216c7b01c910cfa1454da936391c6.png)

How to use the learning curve to identify high deviation/underfitting: As an example, we try to adapt the following data with a straight line. It can be seen that no matter how large the training set is, there will not be much improvement:

![](../images/4a5099b9f4b6aac5785cb0ad05289335.jpg)

In other words, in the case of high deviation/underfitting, adding data to the training set may not necessarily help.
The
How to use the learning curve to identify high variance/overfitting: Suppose we use a very high-degree polynomial model, and the regularization is very small. It can be seen that when the cross-validation set error is much larger than the training set error, the training set increases Multiple data can improve the effect of the model.

![](../images/2977243994d8d28d5ff300680988ec34.jpg)

In other words, in the case of high variance/overfitting, adding more data to the training set may improve the algorithm effect.
### 10.7 Deciding What to Do Next Revisited

Let's go back to the original example again and find the answer there. This is our previous example. Reviewing the six optional next steps proposed in 1.1, let's take a look at how we should choose under what circumstances:

1. Get more training samples-solve high variance

2. Try to reduce the number of features-to solve the high variance

3. Try to get more features-solve high deviations

4. Attempt to increase polynomial characteristics-solve high deviations

5. Try to reduce the degree of regularization λ-solve high deviations

6. Try to increase the degree of regularization λ-solve the high variance
Neural network variance and deviation:
![](../images/c5cd6fa2eb9aea9c581b2d78f2f4ea57.png)

The use of a smaller neural network is similar to the case of fewer parameters, which is likely to cause high deviation and underfitting, but the calculation cost is smaller. The use of a larger neural network is similar to the case of more parameters, which is likely to cause high variance and excessive Fitting, although computationally expensive, can be adjusted by means of regularization to better suit the data.
	
It is usually better to choose a larger neural network and use regularization processing than to use a smaller neural network.
	
For the selection of the number of hidden layers in the neural network, the number of layers is usually gradually increased from one layer. In order to make a better choice, the data can be divided into a training set, a cross-validation set, and a test set, for different hidden layer layers. Number of neural networks to train neural networks,
Then select the neural network with the least cost of the cross-validation set.



11.Machine Learning System Design
--------------------------------------------------------

### 11.1 Prioritizing What to Work On
Our first decision is how to select and express the feature vector $x$. We can choose a list consisting of the 100 most frequently appeared words in spam, according to whether these words appear in the mail, to obtain our feature vector (appears as 1, does not appear as 0), the size is 100×1.

In order to build this classifier algorithm, we can do many things, for example:
1. Collect more data, so that we have more samples of spam and non-spam

2. Develop a series of complex features based on mail routing information

3. Develop a series of complex features based on the body text of the email, including consideration of truncation

4. Develop complex algorithms to detect deliberate typos (write **watch** as **w4tch**)

### 11.2 Error Analysis 
	
The recommended method for building a learning algorithm is:
	
1. Start with a simple algorithm that can be implemented quickly, implement the algorithm, and test the algorithm with cross-validation data
	

2. Draw a learning curve and decide whether to add more data, or add more features, or other options
	
3. Perform error analysis: manually check the samples that produce prediction errors in our algorithm in the cross-validation set, and see if these samples have a systematic trend

What people often do is: spend a lot of time on constructing algorithms, constructing the simple methods they think. Therefore, don't worry about your algorithm being too simple or too imperfect, but implement your algorithm as quickly as possible. When you have an initial implementation, it will become a very powerful tool to help you decide what to do next. Because we can first look at the errors caused by the algorithm, through error analysis to see what mistakes he made, and then decide the optimization method. Another thing is: suppose you have a fast and imperfect algorithm implementation, and there is a numerical evaluation data, which will help you try new ideas and quickly discover whether these ideas you try can improve the performance of the algorithm , So that you will make decisions faster, what to give up in the algorithm, and what error analysis to absorb can help us systematically choose what to do.

### 11.3 Error Metrics for Skewed Classes


In the previous course, I mentioned error analysis and the importance of setting error metrics. That is, set a certain real number to evaluate your learning algorithm and measure its performance, with the evaluation of the algorithm and the error metric. One important thing to note is that using an appropriate error metric can sometimes have a very subtle effect on your learning algorithm. This important thing is the problem of skewed classes. The case of class skew is manifested in the fact that there are many samples of the same kind in our training set, with few or no samples of other classes.
	
For example, we want to use an algorithm to predict whether a cancer is malignant. In our training set, only 0.5% of instances are malignant tumors. Suppose we write a non-learning algorithm that predicts that the tumor is benign in all cases, then the error is only 0.5%. However, the neural network algorithm we obtained through training has an error of 1%. At this time, the size of the error cannot be regarded as the basis for judging the effect of the algorithm.

**Precision**  and **Recall**  We divide the results predicted by the algorithm into four cases:

1. **True Positive** (**TP**): the prediction is true, the actual is true
2. **True Negative** (**TN**): the prediction is false, the actual is false
3. **False Positive** (**FP**): the prediction is true, the actual is false
4. **False Negative** (**FN**): the prediction is false, the actual is true

Then: precision rate=**TP/(TP+FP)**. For example, among all the patients we predicted to have malignant tumors, the actual percentage of patients with malignant tumors, the higher the better.
	
Recall rate=**TP/(TP+FN)**. For example, among all patients with malignant tumors, the percentage of patients with malignant tumors successfully predicted, the higher the better.
	
In this way, for the algorithm we just predicted that the patient's tumor is benign, the recall rate is 0.

|            |              | **predict value**   |             |
| ---------- | ------------ | ------------ | ----------- |
|            |              | **Positive** | **Negtive** |
| **actual value** | **Positive** | **TP**       | **FN**      |
|            | **Negtive**  | **FP**       | **TN**      |

### 11.4 Trading Off Precision and Recall 

Continue to use the example just predicted the nature of the tumor. If the result of our algorithm is between 0-1, we use a threshold of 0.5 to predict true and false.
![](../images/ad00c2043ab31f32deb2a1eb456b7246.png)

**Precision=TP/(TP+FP)**
For example, among all the patients we predicted to have malignant tumors, the actual percentage of patients with malignant tumors, the higher the better.

**Recall=TP/(TP+FN)** In all patients with malignant tumors, the percentage of patients with malignant tumors successfully predicted, the higher the better。

If we want to predict true (the tumor is malignant) only if we are very confident, that is, we want a higher precision, we can use a threshold greater than 0.5, such as 0.7, 0.9. In doing so, we will reduce the number of false predictions of patients as malignant tumors, but at the same time increase the number of failures to predict tumors as malignant.	

	
If we want to improve the recall rate and try to make all patients who are likely to be malignant tumors to be further examined and diagnosed, we can use a threshold smaller than 0.5, such as 0.3.
	
We can draw the graph of the relationship between recall and precision under different thresholds. The shape of the curve varies according to the data:

![](../images/84067e23f2ab0423679379afc6ed6caf.png)

We hope there is a way to help us choose this threshold. One method is to calculate **F1 Score**, the calculation formula is:

${{F}_{1}}Score:2\frac{PR}{P+R}$

We choose the threshold that makes **F1** the highest value.