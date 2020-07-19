Second Week
=====

[TOC]

4.Linear Regression with Multiple Variables
-------------------------------------------------------------

### 4.1 Multiple Features

Now we add more features to the housing price model to form a model with multiple variables. The features in the model are$\left( {x_{1}},{x_{2}},...,{x_{n}} \right)$。

![](../images/591785837c95bca369021efa14a8bb1c.png)


$n$ denotes the number of features
${x^{\left( i \right)}}$ represents the  $i$ training instance, the $i$ row in the feature matrix, and is a **vector**.

${x}^{(2)}\text{=}\begin{bmatrix} 1416\\\ 3\\\ 2\\\ 40 \end{bmatrix}$，
${x}_{j}^{\left( i \right)}$ represents the $j$ feature of the $i$ row in the feature matrix, which is the $j$ feature of the $i$ training instance characteristics.

$x_{2}^{\left( 2 \right)}=3,x_{3}^{\left( 2 \right)}=2$，

The multivariate support hypothesis $h$ is expressed as：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$，

There are $n+1$ parameters and $n$ variables in this formula. In order to make the formula more simplified, the introduction of $x_{0}=1$ will convert the formula into:
$h_{\theta} \left( x \right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$


At this time, the parameter in the model is a $n+1$ dimension vector, any training instance is also a $n+1$ dimension vector, and the dimension of the feature matrix $m*(n+1)$ . Therefore, the formula can be simplified to: $h_{\theta} \left( x \right)={\theta^{T}}X$, where the superscript $T$ represents the matrix transpose.


### 4.2 Gradient Descent for Multiple Variables

Reference video : 4 - 2 - Gradient Descent for Multiple Variables (5 min).mkv

Similar to univariate linear regression, in multivariate linear regression, we also build a cost function, then this cost function is the sum of squares of all modeling errors：$J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ ，

$h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$ ，

The batch gradient descent algorithm for multivariate linear regression is:

![](../images/41797ceb7293b838a3125ba945624cf6.png)



![](../images/6bdaff07783e37fcbb1f8765ca06b01b.png)

After deriving the derivative

![](../images/dd33179ceccbd8b0b59a5ae698847049.png)

when $n>=1$，
${{\theta }_{0}}:={{\theta }_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)}$

${{\theta }_{1}}:={{\theta }_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{1}^{(i)}$

${{\theta }_{2}}:={{\theta }_{2}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{2}^{(i)}$

We start to randomly select a series of parameter values, calculate all prediction results, and then give all parameters a new value, and so on until convergence.

Calculate the cost function
$J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {h_{\theta}}\left( {x^{(i)}} \right)-{y^{(i)}} \right)}^{2}}}$

${h_{\theta}}\left( x \right)={\theta^{T}}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

**Python** code

```python
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
```

### 4.3  Gradient Descent in Practice I - Feature Scaling

Reference video: 4 - 3 - Gradient Descent in Practice I - Feature Scaling (9 min).mkv

When we face multi-dimensional feature problems, we must ensure that these features have similar scales, which will help the gradient descent algorithm to converge faster.

Taking the housing price problem as an example, suppose we use two features, the size of the house and the number of rooms. The value of the size is 0-2000 square feet, and the value of the number of rooms is 0-5. Coordinate, drawing a contour map of the cost function can see that the image will appear very flat, the gradient descent algorithm requires very many iterations to converge.

![](../images/966e5a9b00687678374b8221fdd33475.jpg)

The solution is to try to scale the scale of all features to between -1 and 1. As shown:

![](../images/b8167ff0926046e112acf789dba98057.png)

The easiest way is to make: ${{x}_{n}}=\frac{{{x}_{n}}-{{\mu}_{n}}}{{{s}_{n }}}$, where ${\mu_{n}}$ is the average and ${s_{n}}$ is the standard deviation.


### 4.4 Gradient Descent in Practice II - Learning Rate

Reference video: 4 - 4 - Gradient Descent in Practice II - Learning Rate (9 min).mkv

The number of iterations required for the convergence of the gradient descent algorithm differs depending on the model. We cannot predict in advance. We can draw a graph of the number of iterations and the cost function to observe when the algorithm tends to converge.

![](../images/cd4e3df45c34f6a8e2bb7cd3a2849e6c.jpg)

There are also some methods for automatically testing whether convergence is achieved, such as comparing the change value of the cost function with a certain threshold (for example, 0.001), but it is usually better to look at the chart above

Each iteration of the gradient descent algorithm is affected by the learning rate. If the learning rate $a$ is too small, the number of iterations required to achieve convergence will be very high; if the learning rate $\alpha$ is too large, each iteration may not decrease A small cost function may cross the local minimum and cause failure to converge.

We can usually consider trying some learning rates:
$\alpha=0.01，0.03，0.1，0.3，1，3，10$

### 4.5  Features and Polynomial Regression

Reference video : 4 - 5 - Features and Polynomial Regression (8 min).mkv

Such as the problem of housing price prediction

![](../images/8ffaa10ae1138f1873bc65e1e3657bd4.png)

$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}\times{frontage}+{\theta_{2}}\times{depth}$ 

${x_{1}}=frontage$（Frontage width），${x_{2}}=depth$（Longitudinal depth），$x=frontage*depth=area$（area），then：${h_{\theta}}\left( x \right)={\theta_{0}}+{\theta_{1}}x$。
Linear regression is not suitable for all data, sometimes we need curves to adapt to our data, such as a quadratic model：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}$
or cubic model： $h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}$ 

![](../images/3a47e15258012b06b34d4e05fb3af2cf.jpg)

Usually we need to observe the data before deciding what kind of model we are going to try. In addition, we can make:
${{x}_{2}}=x_{2}^{2},{{x}_{3}}=x_{3}^{3}$，thereby transforming the model into a linear regression model.

According to the graphical characteristics of the function, we can also make:

${{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta}_{2}}{{(size)}^{2}}$

or:

${{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta }_{2}}\sqrt{size}$

### 4.6 Normal Equation

4 - 6 - Normal Equation (16 min).mkv

So far, we have been using gradient descent algorithm, but for some linear regression problems, the normal equation method is a better solution. Such as:

![](../images/a47ec797d8a9c331e02ed90bca48a24b.png)

The normal equation is to find the parameter that minimizes the cost function by solving the following equation：$\frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0$ 。
Suppose our training set feature matrix is $X$ (including ${{x}_{0}}=1$) and our training set result is vector $y$, then use the normal equation to solve vector $\theta ={{\left( {X^T}X \right)}^{-1}}{X^{T}}y$
The superscript **T** represents the matrix transpose, and the superscript -1 represents the inverse of the matrix. If the matrix $A={X^{T}}X$, then: ${{\left( {X^T}X \right)}^{-1}}={A^{-1}}$

The following represents data as an example:

![](../images/261a11d6bce6690121f26ee369b9e9d1.png)


![](../images/c8eedc42ed9feb21fac64e4de8d39a06.png)

Solve the parameters using the normal equation method:

![](../images/b62d24a1f709496a6d7c65f87464e911.jpg)

Note: For those irreversible matrices (usually because the features are not independent, such as including both the size in feet and the size in meters, the number of features may be greater than the number of training sets), the normal equation method Is not usable.

Comparison of gradient descent and normal equation:

| gradient descent              | normal equation                                     |
| ---------------- | ---------------------------------------- |
| need to choose learning rate $\alpha$  | no need                                      |
| multiple iterations           | calculated in one operation                                   |
| It can be better applied when the number of features is large | Need to calculate ${{\left( {{X}^{T}}X \right)}^{-1}}$ If the number of features n is large, the operation cost is high, because the calculation time complexity of the matrix inverse is $ O\left( {{n}^{3}} \right)$, generally speaking, it is still acceptable when $n$ is less than 10000 |
| Suitable for all types of models       | Only applicable to linear models, not to other models such as logistic regression models                  |

To sum up, as long as the number of characteristic variables is not large, the standard equation is a good alternative method for calculating the parameter $\theta$. Specifically, as long as the number of feature variables is less than 10,000, I usually use the standard equation method instead of the gradient descent method.


The **python** implementation of the normal equation:

```python
import numpy as np
    
 def normalEqn(X, y):
    
   theta = np.linalg.inv(X.T@X)@X.T@y #X.T@Xequal X.T.dot(X)
    
   return theta
```