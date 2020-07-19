First week 
=====
[TOC]
Introduction
------------------

### 1.1 welcome

### 1.2 What is machine learning?

Tom defines machine learning by saying that, a well posed learning problem is defined as follows. He says, a computer program is said to learn from experience **E**, with respect to some task **T**, and some performance measure **P**, if its performance on **T** as measured by **P** improves with experience **E**.


### 1.3 Supervised Learning 

Let's say you want to predict housing prices. A while back, a student collected data sets from the Institute of Portland Oregon. And let's say you plot a data set and it looks like this. Here on the horizontal axis, the size of different houses in square feet,and on the vertical axis, the price of different houses in thousands of dollars.Given this data, let's say you have a friend who owns a house that is, say 750 square feet and hoping to sell the houseand they want to know how much they can get for the house.

So how can the learning algorithm help you? One thing a learning

![](../images/2d99281dfc992452c9d32e022ce71161.png)
algorithm might be able to do is put a straight line through the data or to fit a straight line to the data and, based on that, it looks like maybe the house can be sold for maybe about $\$150,000$. But maybe this isn't the only learning algorithm you can use. There might be a better one. For example, instead of sending a straight line to the data, we might decide that it's better to fit a quadraticfunction or a second-order polynomial to this data. And if you do that, and make a prediction here, then it looks like, well,maybe we can sell the house for closer to $\$200,000$. One of the things we'll talk about later is how to choose and how to decide do you want to fit a straight line to the data or do you want to fit the quadratic function to the data and there's no fair picking whichever one gives your friend the better house to sell. But each of these would be a fine example of a learning algorithm. So this is an example of a supervised learning algorithm.

And the term supervised learning refers to the fact that we gave the algorithm a data set in which the "right answers" were given. That is, we gave it a data set of houses in which for every example in this data set, we told it what is the right price so what is the actual price that,that house sold for and the toss of the algorithm was to just produce more of these right answers such as for this new house, you know, that your friend may be trying to sell. To define with a bit more terminology this is also called a **regression problem** and by regression.problem I mean we're trying to predict a continuous value output. Namely the price.

So technically I guess prices can be rounded off to the nearest cent. So maybe prices are actually discrete values, but usually we think of the price of a house as a real number, as a scalar value, as a continuous value number and the term regression refers to the fact that we're trying to predict the sort of continuous values attribute.

![](../images/4f80108ebbb6707d39b7a6da4d2a7a4e.png)
Here's another supervised learning example, some friends and I were actually working on this earlier. Let's see you want to look at medical records and try to predict of a breast cancer as malignant or benign. If someone discovers a breast tumor, a lump in their breast, a malignant tumor is a tumor that is harmful and dangerous and a benign tumor is a tumor that is harmless. So obviously people care a lot about this.Let's see a collected data set and suppose in your data set you have on your horizontal axis the size of the tumor and on the vertical axis I'm going to plot one or zero, yes or no, whether or not these are examples of tumors we've seen before are malignant–which is one–or zero if not malignant or benign. So let's say our data set looks like this where we saw a tumor of this size that turned out to be benign. One of this size, one of this size. And so on.And sadly we also saw a few malignant tumors, one of that size, one of that size, one of that size... So on. So this example... I have five examples of benign tumors showndown here, and five examples of malignant tumors shown with a vertical axis value of one. And let's say we have a friend who tragically has a breast tumor, and let's say her breast tumor size is maybe somewhere around this value.

The machine learning question is, can you estimate what is the probability, what is the chance that a tumor is malignant versus benign? To introduce a bit more terminology this is an example of a classification problem. The term classification refers to the fact that here we're trying to predict a discrete value output: zero or one, malignant or benign. And it turns out that in classification problems sometimes you can have more than two values for the two possible values for the output.

Supervised learning is that, in supervised learning, **in every example in our data set, we are told what is the "correct answer" that we would have quite liked the algorithms have predicted on that example.** Such as the price of the house, or whether a tumor is malignant or benign. We also talked about the regression problem. And by regression,that means that our goal is to predict a continuous valued output. And we talked about the classification problem, where the goal is to predict a discrete value output.

### 1.4 Unsupervised Learning

![](../images/0c93b5efd5fd5601ed475d2c8a0e6dcd.png)

![](../images/94f0b1d26de3923fc4ae934ec05c66ab.png)

In Unsupervised Learning, we're given data that looks different than data that looks like this that doesn't have any labels or that all has the same label or really no labels. So we're given the data set and we're not told what to do with it and we're not told what each data point is.Instead we're just told, here is a data set.

Unsupervised Learning algorithm might decide that the data lives in two different clusters.**Unsupervised Learning, which is a learning setting where you give the algorithm a ton of data and just ask it to find structure in the data for us.**

二、Linear Regression with One Variable
-------------------------------------------------------

### 2.1 Model Representation

Reference video: 2 - 1 - Model Representation (8 min).mkv

We're going to use a data set of housing prices from the city of Portland, Oregon. And here I'm going to plot my data set of a number of houses that were different sizes that were sold for a range of different prices. One thing we could do is fit a model. Maybe fit a straight line to this data.

![](../images/8e76e65ca7098b74a2e9bc8e9577adfc.png)

in supervised learning, we have a data set and this data set is called a training set. So for housing prices example, we have a training set of different housing prices and our job is to learn from this data how to predict prices of the houses.

![](../images/44c68412e65e62686a96ad16f278571f.png)


$m$ denote the number of training examples.

$x$ denote the input variables often also called the features.

$y$ denote my output variables or the target variable

$\left( x,y \right)$ denote a single training example

$({{x}^{(i)}},{{y}^{(i)}})$ denote $i$ training example

$h$ represents the solution or function of the learning algorithm, also called hypothesis（**hypothesis**）

![](../images/ad0718d6e5218be6e6fce9dc775a38e6.png)
Here's how this supervised learning algorithm works.We saw that with the training set like our training set of housing prices and we feed that to our learning algorithm. Is the job of a learning algorithm to then output a function which by convention is usually denoted lowercase h and h stands for hypothesis And what the job of the hypothesis is a function that takes as input the size of a house like maybe the size of the new house your friend's trying to sell so it takes in the value of x and it tries to output the estimat value of y for the corresponding house. So h is a function that maps from x's to y's.
Designing a learning algorithm, the next thing we need to decide is how do we represent this hypothesis h.
One possible expression is：$h_\theta \left( x \right)=\theta_{0} + \theta_{1}x$ because there is only one feature / input variable, such a problem is called a univariate linear regression problem.

### 2.2 Cost Function

Reference video: 2 - 2 - Cost Function (8 min).mkv
In linear regression we have a training set like
that shown here. 

![](../images/d385f8a293b254454746adee51a027d4.png)

In linear regression we have a training set like this，$m$was the number of training examples，So maybe $m = 47$. And the form of the hypothesis, which we use to make predictions, is this linear function. $h_\theta \left( x \right)=\theta_{0}+\theta_{1}x$。

All we have to do now is choose the appropriate （**parameters**）$\theta_{0}$ 和 $\theta_{1}$ for our model，In the case of the housing price problem, it is the slope of the straight line and the intercept on the $ y $ axis.

The parameter we choose determines the accuracy of the straight line we get relative to our training set. The gap between the value predicted by the model and the actual value in the training set (the blue line in the figure below) is **modeling error**。

![](../images/6168b654649a0537c67df6f2454dc9ba.png)

Our goal is to choose the model parameters that can minimize the sum of squared modeling errors。 Minimize cost function $J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$.

We draw a contour map, the three coordinates are$\theta_{0}$和$\theta_{1}$ 和$J(\theta_{0}, \theta_{1})$：

![](../images/27ee0db04705fb20fab4574bb03064ab.png)
It can be seen that there is a point in the three-dimensional space that minimizes $J(\theta_{0}, \theta_{1})$.


### 2.3  Cost Function - Intuition I 

Reference video: 2 - 3 - Cost Function - Intuition I (11 min).mkv

![](../images/10ba90df2ada721cf1850ab668204dc9.png)

![](../images/2c9fe871ca411ba557e65ac15d55745d.png)

### 2.4 Cost Function - Intuition II

Reference video: 2 - 4 - Cost Function - Intuition II (9 min).mkv

![](../images/0b789788fc15889fe33fb44818c40852.png)
The cost function looks like a contour plot, and it can be seen that there is a point in three-dimensional space that minimizes$J(\theta_{0}, \theta_{1})$
![](../images/86c827fe0978ebdd608505cd45feb774.png)

### 2.5 Gradient Descent 

Reference video: 2 - 5 - Gradient Descent (11 min).mkv

Gradient descent is an algorithm for finding the minimum value of the function. We will use the gradient descent algorithm to find the minimum value of the cost function $J(\theta_{0}, \theta_{1})$.
The idea behind gradient descent is: at the beginning we randomly choose a combination of parameters $\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$ calculate the cost function, and then we look for the next parameter combination that can make the cost function value drop the most. We continue to do this until we find a **local minimum** , because we have not tried all the parameter combinations, so we are not sure whether the local minimum we get is the **global minimum**.If different initial parameter combinations are selected, different local minimum values may be found.

![](../images/db48c81304317847870d486ba5bb2015.jpg)

The formula of the **batch gradient descent** algorithm is:
![](../images/7da5a5f635b1eb552618556f1b4aac1a.png)
Where $a$ is the **learning rate**, which determines how big the steps we take in the direction that can make the cost function drop the most. At the same time, all the parameters minus the learning rate multiplied by the derivative of the cost function.

### 2.6 Gradient Descent Intuition

Reference video:: 2 - 6 - Gradient Descent Intuition (12 min).mkv
The gradient descent algorithm is as follows：

${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)$
Description: Assign a value to $\theta $, so that$J\left( \theta  \right)$ will proceed in the fastest direction of gradient descent, iterate all the way, and finally get the local minimum. Where $ a $ is the **learning rate** , which determines how much steps we take in the direction that can make the cost function decrease the most.

![](../images/ee916631a9f386e43ef47efafeb65b0f.png)

![](../images/0c31b42f1ee2b0703decf4e6c55d61d1.wmf)

For the purpose of derivation, it can basically be said that the tangent of this red dot is such a red straight line, which is exactly tangent to the function. Let us look at the slope of this red straight line, which is just the curve of the function. The tangent of this line, the slope of this line is exactly the height of this triangle divided by this horizontal length. Now, this line has a positive slope, which means it has a positive derivative, so I get the new ${\theta_{1}}$，${\theta_{1}}$ after update equals ${\theta_{1}}$ minus a positive number multiplied by $a$.

${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)$
If $\alpha$ is too small, the result is that it can only move a little bit like a baby, trying to reach the lowest point, so it takes many steps to reach the lowest point, so if If $\alpha$ is too small, it may be very slow, because it will move a little, and it will take many steps to reach the global lowest point.
If $\alpha$ is too large, then the gradient descent method may cross the lowest point, and may even fail to converge. The next iteration moves a big step, over once, over again, over the lowest point again and again, until you find that in fact It is getting farther and farther from the lowest point, so if $\alpha$ is too large, it will cause failure to converge or even diverge.

In the gradient descent method, when we are close to the local minimum, the gradient descent method will automatically take a smaller amplitude, because when we are close to the local minimum, it is clear that the derivative is equal to zero at the local minimum, so when we are close to the local At the lowest point, the derivative value will automatically become smaller and smaller, so the gradient descent will automatically take a smaller amplitude. This is the method of gradient descent. So there is actually no need to reduce $\alpha$.


### 2.7 Gradient Descent for LinearRegression 

Reference video: 2 - 7 - GradientDescentForLinearRegression (6 min).mkv
The comparison of gradient descent algorithm and linear regression algorithm is shown in the figure:
![](../images/5eb364cc5732428c695e2aa90138b01b.png)

The key to applying the gradient descent method to the linear regression problem is to find the derivative of the cost function:

$\frac{\partial }{\partial {{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$  时：$\frac{\partial }{\partial {{\theta }_{0}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$

$j=1$  时：$\frac{\partial }{\partial {{\theta }_{1}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

Then the algorithm is rewritten as:

**Repeat {**

​                ${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}$

​                ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

​               **}**
The algorithm we just used is sometimes called batch gradient descent. In fact, in machine learning, the algorithm is not usually named, but the name "**batch gradient descent**" refers to all the training samples used in each step of gradient descent. In gradient descent, when calculating the differential derivative term, we need to perform a summation operation, so in each individual gradient descent, we must finally calculate such a thing, this item requires training for all Sample summation. Therefore, the name of the batch gradient descent method shows that we need to consider all this "batch" training samples, and in fact, sometimes there are other types of gradient descent methods, not this "batch" type, without considering the entire training set Instead, only focus on a small subset of the training set each time. 