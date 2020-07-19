Third Week
=====
[TOC]

6 Logistic Regression
---------------------------------

### 6.1 Classification

In the classification problem, we try to predict whether the result belongs to a certain class (for example, right or wrong). Examples of classification problems are: judging whether an email is spam; judging whether a financial transaction is fraudulent; before we also talked about examples of tumor classification problems, distinguishing whether a tumor is malignant or benign.

![](../images/a77886a6eff0f20f9d909975bb69a7ab.png)

我们从二元的分类问题开始讨论。

We refer to the two classes that the **dependent variable** may belong to as **negative class** and  **positive class**, then the dependent variable $y\ in {0,1 \\}$, where 0 means negative class and 1 means positive class.

![](../images/f86eacc2a74159c068e82ea267a752f7.png)

![](../images/e7f9a746894c4c7dfd10cfcd9c84b5f9.png)
If we want to use a linear regression algorithm to solve a classification problem, for classification, the value of $y$ is 0 or 1, but if you are using linear regression, then it is assumed that the output value of the function may be much greater than 1, or much less than 0 , Even if the labels $y$ of all training samples are equal to 0 or 1. Although we know that the label should take the value 0 or 1, it will feel strange if the value obtained by the algorithm is much greater than 1 or much less than 0. So the algorithm we will study next is called the logistic regression algorithm. The nature of this algorithm is: its output value is always between 0 and 1.

### 6.2 Hypothesis Representation

Recalling the breast cancer classification problem mentioned at the beginning, we can use linear regression to find a straight line suitable for the data:

![](../images/29c12ee079c079c6408ee032870b2683.jpg)

According to the linear regression model, we can only predict continuous values. However, for classification problems, we need to output 0 or 1, and we can predict:

when ${h_\theta}\left( x \right)>=0.5$ predict $y=1$。

when ${h_\theta}\left( x \right)<0.5$ predict $y=0$ 。

For the data shown in the figure above, such a linear model seems to be able to complete the classification task well. If we observe a very large size malignant tumor, add it as an example to our training set, which will allow us to obtain a new line.

![](../images/d027a0612664ea460247c8637b25e306.jpg)

At this time, it is not appropriate to use 0.5 as a threshold to predict whether the tumor is benign or malignant. It can be seen that the linear regression model is not suitable for solving such problems because its predicted value can exceed the range of [0,1].

We introduce a new model, logistic regression. The output variable range of this model is always between 0 and 1.
The hypothesis of the logistic regression model is $h_\theta \left( x \right)=g\left(\theta^{T}X \right)$

$X$ represents the feature vector
$g$ stands for **logistic function** is a commonly used logical function is **S** shape function (**Sigmoid function**), the formula is: $g\left( z \right)=\frac{1 }{1+{{e}^{-z}}}$.


**python** code

```python
import numpy as np
    
def sigmoid(z):
    
   return 1 / (1 + np.exp(-z))
```

The image of this function is:

![](../images/1073efb17b0d053b4f9218d4393246cc.jpg)

Taken together, we get the hypothesis of the logistic regression model：

$g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。


The role of $h_\theta \left( x \right)$ is to calculate the probability of output variable=1 for the given input variable **estimated probablity**, which is $h_\theta \left (x \right)=P\left( y=1|x;\theta \right)$


### 6.3 Decision Boundary

![](../images/6590923ac94130a979a8ca1d911b68a3.png)

In logistic regression, we predict:

When ${h_\theta}\left( x \right)>=0.5$, predict $y=1$.

When ${h_\theta}\left( x \right)<0.5$, predict $y=0$.

According to the **S** shape function image drawn above, we know that 

When $z=0$ $g(z)=0.5$

When $z>0$ $g(z)>0.5$

When $z<0$ $g(z)<0.5$

and $z={\theta^{T}}x$, that is:
${\theta^{T}}x>=0$, predict $y=1$
When ${\theta^{T}}x<0$, predict $y=0$

Now suppose we have a model:

![](../images/58d098bbb415f2c3797a63bd870c3b8f.png)

And the parameter $\theta$ is a vector [-3 1 1]. Then when $-3+{x_1}+{x_2} \geq 0$, that is ${x_1}+{x_2} \geq 3$, the model will predict $y=1$.
We can draw a straight line ${x_1}+{x_2} = 3$, this line is the boundary of our model, separating the area predicted as 1 from the area predicted as 0.

![](../images/f71fb6102e1ceb616314499a027336dc.jpg)

If our data presents such a distribution, what model is suitable?

![](../images/197d605aa74bee1556720ea248bab182.jpg)

Because a curve is needed to separate the region of $y=0$ and the region of $y=1$, we need the quadratic feature: ${h_\theta}\left( x \right)=g\left( {\theta_0 }+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}+{\theta_{3}}x_{1}^{2}+{\theta_{4}}x_{2 }^{2} \right)$ is [-1 0 0 1 1], then the decision boundary we get is exactly a circle with a circle point at the origin and a radius of 1.

We can use very complex models to adapt to decision boundaries of very complex shapes.

### 6.4 Cost Function


![](../images/f23eebddd70122ef05baa682f4d6bd0f.png)

For linear regression models, the cost function we define is the sum of squared errors of all models. In theory, we can also use this definition for the logistic regression model, but the problem is that when we put ${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$ When brought into the cost function defined in this way, the cost function we get will be a **non-convex function**

![](../images/8b94e47b7630ac2b0bcb10d204513810.jpg)

This means that our cost function has many local minimums, which will affect the gradient descent algorithm to find the global minimum.

The cost function of linear regression is：$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ 。
我们重新定义逻辑回归的代价函数为：$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}$，

![](../images/54249cb51f0086fa6a805291bf2639f1.png)

The relationship between ${h_\theta}\left( x \right)$ and $Cost\left( {h_\theta}\left( x \right),y \right)$ is shown below:

![](../images/ffa56adcc217800d71afdc3e0df88378.jpg)

The characteristics of the $Cost\left( {h_\theta}\left( x \right),y \right)$ function constructed in this way are: when the actual $y=1$ and ${h_\theta}\left( x \right)$ is 1 when the error is 0, when $y=1$ but ${h_\theta}\left( x \right)$ is not 1, the error follows ${h_\theta}\left( x \right)$ becomes smaller and larger; when the actual $y=0$ and ${h_\theta}\left( x \right)$ is also 0, the cost is 0, when $y=0$ but ${ When h_\theta}\left( x \right)$ is not 0, the error becomes larger as ${h_\theta}\left( x \right)$ becomes larger.


Simplify the constructed $Cost\left( {h_\theta}\left( x \right),y \right)$ as follows: 
$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$
bring in the cost function to get:
$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$
$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

**Python** code

```python
import numpy as np
    
def cost(theta, X, y):
    
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```

After obtaining such a cost function, we can use the gradient descent algorithm to find the parameters that can minimize the cost function. The algorithm is:

**Repeat** {
$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)$
(**simultaneously update all** )
}


After derivation, we get:

**Repeat** {
$\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}$ 
**(simultaneously update all** )
}

We define the cost function of a single training sample. The content of convexity analysis is beyond the scope of this course, but it can be proved that the algebraic value function we choose will give us a convex optimization problem. The cost function $J(\theta)$ will be a convex function, and there is no local optimal value.


$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$
considering that：
${h_\theta}\left( {{x}^{(i)}} \right)=\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}}$
then：
${{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)$
$={{y}^{(i)}}\log \left( \frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)$
$=-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^T}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^T}{{x}^{(i)}}}} \right)$

So：
$\frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)=\frac{\partial }{\partial {\theta_{j}}}[-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^{T}}{{x}^{(i)}}}} \right)]}]$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\frac{-x_{j}^{(i)}{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}{1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}]$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{{y}^{(i)}}\frac{x_j^{(i)}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}]$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}x_j^{(i)}-x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}+{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}\left( 1\text{+}{{e}^{{\theta^T}{{x}^{(i)}}}} \right)-{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}x_j^{(i)}}$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}-{h_\theta}\left( {{x}^{(i)}} \right)]x_j^{(i)}}$
$=\frac{1}{m}\sum\limits_{i=1}^{m}{[{h_\theta}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_j^{(i)}}$

Note: Although the obtained gradient descent algorithm looks the same as the linear regression gradient descent algorithm, here ${h_\theta}\left( x \right)=g\left( {\theta^T}X \right)$ is different from linear regression, so it is actually different. In addition, before running the gradient descent algorithm, feature scaling is still very necessary.

一些梯度下降算法之外的选择：
除了梯度下降算法以外，还有一些常被用来令代价函数最小的算法，这些算法更加复杂和优越，而且通常不需要人工选择学习率，通常比梯度下降算法要更加快速。这些算法有：**共轭梯度**（**Conjugate Gradient**），**局部优化法**(**Broyden fletcher goldfarb shann,BFGS**)和**有限内存局部优化法**(**LBFGS**) 
Some alternatives to gradient descent algorithms:
In addition to the gradient descent algorithm, there are some algorithms that are often used to minimize the cost function. These algorithms are more complex and superior, and usually do not require manual selection of the learning rate, and are usually faster than the gradient descent algorithm. These algorithms are: **Conjugate Gradient** , **Local Optimization Method** (**Broyden fletcher goldfarb shann, BFGS**) and **Limited Memory Local Optimization Method** ( **LBFGS**)


### 6.5 Simplified Cost Function and Gradient Descent

This is the cost function of logistic regression:
![](../images/eb69baa91c2fc6e7dd8ebdf6c79a6a6f.png)

This formula can be combined into:

$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$
the cost function of logistic regression:
$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$
$=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$
According to this cost function, what should we do to fit the parameters? We want to try to find the parameter $\theta $ that minimizes $J\left( \theta \right)$ as much as possible.
$\underset{\theta}{\min }J\left( \theta  \right)$ 
So we want to minimize this item, which will give us a certain parameter $\theta $.
If we give a new sample, if a certain feature $x$, we can use the parameter $\theta $ that fits the training sample to output the prediction of the hypothesis.

In addition, our hypothetical output is actually this probability value: $p(y=1|x;\theta)$, which is about the probability of $x$ with $\theta $ as the parameter, $y=1$, you It can be considered that our assumption is to estimate the probability of $y=1$, so the next step is to figure out how to minimize the cost function $J\left( \theta \right)$ as a statement about $\theta $ Function, so that we can fit the parameter $\theta $ to the training set.


The method to minimize the cost function is to use the **gradient descent** method . This is our cost function:
$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

If we want to minimize the value of this function about $\theta$, this is the template of the gradient descent method we usually use.


![Want ${{\min }_\theta}J(\theta )$：](../images/171031235527.png)

We have to update each parameter repeatedly, using this formula to update, that is, subtracting the learning rate $\alpha$ by itself and multiplying the subsequent differential term. After derivation, you get:


![Want ：](../images/171031235719.png)


If you calculate it, we will get this equation:
${\theta_j}:={\theta_j}-\alpha \frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}}){x_{j}}^{(i)}}$
I write it here and sum the latter formula from $i=1$ to $m$, which is actually the prediction error multiplied by $x_j^{(i)}$, so you take this partial derivative term $\frac{\partial }{\partial {\theta_j}}J\left( \theta \right)$ back to the original formula, we can write the gradient descent algorithm as follows:
${\theta_j}:={\theta_j}-\alpha \frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{( i)}})-{{y}^{(i)}}){x_{j}}^{(i)}}$

So, if you have $n$ features, that is: ![](../images/0171031235044.png), the parameter vector $\theta $ includes ${\theta_{0}}$ ${\theta_{ 1}}$ ${\theta_{2}}$ all the way to ${\theta_{n}}$, then you need to use this formula:

${\theta_j}:={\theta_j}-\alpha \frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{( i)}})-{{y}^{(i)}}){{x}_{j}}^{(i)}}$ to update all $\theta $ values at the same time.

Now, if you compare this update rule with the linear regression we used before, you will be surprised to find that this formula is exactly what we use for linear regression gradient descent.

So, are linear regression and logistic regression the same algorithm? To answer this question, we have to observe logistic regression to see what changes have occurred. In fact, the hypothetical definition has changed.

For the linear regression hypothesis function:

${h_\theta}\left( x \right)={\theta^T}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

And now the logical function assumes the function:

${h_\theta}\left( x \right)=\frac{1}{1+{{e}^{-{\theta^T}X}}}$

Therefore, even though the rules for updating parameters seem to be basically the same, the gradient descent of the logistic function is actually two completely different things from the gradient descent of the linear regression because the definition of the hypothesis has changed.

### 6.6 Advanced Optimization

Now let's look at what is gradient descent from another perspective. We have a cost function $J\left( \theta \right)$, and we want to minimize it, then all we need to do is write code when the input parameter $ \theta$, they will calculate two things: $J\left( \theta \right)$ and the partial derivative term when $J$ is equal to 0, 1 until $n$.

![](../images/394a1d763425c4ecf12f8f98a392067f.png)

Assuming that we have completed the code that can achieve these two things, all gradient descent does is repeatedly perform these updates.
Another way of thinking about gradient descent is: we need to write code to calculate $J\left( \theta \right)$ and these partial derivatives, and then insert these into gradient descent, and then it can be minimized for us This function.

For gradient descent, I think technically, you don't actually need to write code to calculate the cost function $J\left( \theta \right)$. You only need to write code to calculate the derivative terms, but if you want the code to be able to monitor the convergence of these $J\left( \theta \right)$, then we need to write our own code to calculate the cost function $J( \theta)$ and partial derivative terms $\frac{\partial }{\partial {\theta_j}}J\left( \theta \right)$. So, after writing the code that can calculate both, we can use gradient descent.

However, gradient descent is not the only algorithm we can use, there are other algorithms that are more advanced and more complex. If we can use these methods to calculate the cost function $J\left( \theta \right)$ and partial derivative terms $\frac{\partial }{\partial {\theta_j}}J\left( \theta \right)$ with two terms, then these algorithms are different methods for optimizing the cost function for us, **conjugate gradient method BFGS** (**variable scale method**) and **L-BFGS** (**limit variable scale Method**) are some of the more advanced optimization algorithms, they need a method to calculate $J\left( \theta \right)$, and a method to calculate the derivative term, and then use a more complex algorithm than gradient descent to minimize the cost function.


7、Regularization
--------------------------

### 7.1 The Problem of Overfitting
So far, we have learned several different learning algorithms, including linear regression and logistic regression, which can effectively solve many problems, However, when they are applied to some specific machine learning applications, they will encounter the problem of **over-fitting**, which may cause them to be very ineffective.

If we have a lot of features, the hypothesis we learned from learning may be able to adapt very well to the training set (the cost function may be almost 0), but it may not be generalized to new data.

The following figure is an example of a regression problem:

![](../images/72f84165fbf1753cd516e65d5e91c0d3.jpg)

The first model is a linear model, which is under-fitting and cannot adapt well to our training set; the third model is a fourth-order model that places too much emphasis on fitting the original data and loses the essence of the algorithm: prediction New data. We can see that if a new value is given to make a prediction, it will perform poorly and it is overfitting. Although it can adapt to our training set very well, it may not be effective when predicting new input variables. Good; and the model in the middle seems to be the most appropriate.

This problem also exists in the classification problem:

![](../images/be39b497588499d671942cc15026e4a2.jpg)

In terms of polynomials, the higher the degree of $x$, the better the fit, but the corresponding prediction ability may become worse.

The question is, what should we do if we find an overfitting problem?

1.Discard some features that cannot help us predict correctly. You can choose which features to keep manually, or use some model selection algorithms to help (eg **PCA**)

2. Regularization. Keep all the features, but reduce the size of the parameters

### 7.2 Cost Function


In the regression problem above, if our model is:
${h_\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}+{\theta_{4}}{x_{4}^4}$

We can see from the previous examples that it is those higher-order terms that lead to overfitting, so if we can make the coefficients of these higher-order terms close to 0, we can fit them well.
所以我们要做的就是在一定程度上减小这些参数$\theta $ 的值，这就是正则化的基本方法。我们决定要减少${\theta_{3}}$和${\theta_{4}}$的大小，我们要做的便是修改代价函数，在其中${\theta_{3}}$和${\theta_{4}}$ 设置一点惩罚。这样做的话，我们在尝试最小化代价时也需要将这个惩罚纳入考虑中，并最终导致选择较小一些的${\theta_{3}}$和${\theta_{4}}$。
So what we have to do is to reduce the value of these parameters $\theta $ to a certain extent, this is the basic method of regularization. We decided to reduce the size of ${\theta_{3}}$ and ${\theta_{4}}$, all we have to do is modify the cost function, in which ${\theta_{3}}$ and ${ \theta_{4}}$ Set a small penalty. In doing so, we also need to take this penalty into consideration when trying to minimize the cost, and ultimately lead to the selection of smaller ${\theta_{3}}$ and ${\theta_{4}}$.

The modified cost function is as follows：$\underset{\theta }{\mathop{\min }}\,\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}+1000\theta _{3}^{2}+10000\theta _{4}^{2}]}$

The cost of ${\theta_{3}}$ and ${\theta_{4}}$ selected by this cost function has a much smaller impact on the prediction results than before. If we have a lot of features, we don't know which features we want to punish. We will punish all the features and let the software with the most optimized cost function choose the degree of these penalties. The result is a simpler hypothesis that prevents overfitting:

$J\left( \theta  \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]}$

Among them, $\lambda $ is also called **regularization Parameter**. Note: According to common practice, we do not punish ${\theta_{0}}$. The possible comparison between the regularized model and the original model is shown below:

![](../images/ea76cc5394cf298f2414f230bcded0bd.jpg)
If the selected regularization parameter $\lambda$ is too large, all parameters will be minimized, resulting in the model becoming ${h_\theta}\left( x \right)={\theta_{0}}$ , Which is the situation shown by the red straight line in the figure above, causing underfitting.
So why does an additional item $\lambda =\sum\limits_{j=1}^{n}{\theta_j^{2}}$ reduce the value of $\theta $?
Because if we make the value of $\lambda$ large, in order to make **Cost Function** as small as possible, all the values of $\theta $ (excluding ${\theta_{0}}$) will be in reduce to a certain extent.

But if the value of $\lambda$ is too large, then $\theta $ (excluding ${\theta_{0}}$) will approach 0, so that we can only get one parallel to $x$ The straight line of the axis.

So for regularization, we have to take a reasonable value of $\lambda$, so that regularization can be better applied.

### 7.3 Regularized Linear Regression


For solving linear regression, we derived two learning algorithms: one based on gradient descent and one based on normal equations.

The cost function of regularized linear regression is:

$J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{[({{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta _{j}^{2}})]}$

If we want to use the gradient descent method to minimize this cost function, because we have not regularized $\theta_0​$, the gradient descent algorithm will be divided into two situations:

$Repeat$  $until$  $convergence${

​                                                   ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$ 

​                                                   ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$ 

​                                                             $for$ $j=1,2,...n$

​                                                   }

Adjusting the update formula when $ j=1,2,...,n$ in the above algorithm can be obtained:

${\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}​$ 
It can be seen that the change of the gradient descent algorithm for regularized linear regression is that each time the value of $\theta $ is reduced by an additional value based on the original algorithm update rules.

We can also use regular equations to solve the regularized linear regression model, as follows:

![](../images/71d723ddb5863c943fcd4e6951114ee3.png)

The size of the matrix in the figure is $(n+1)*(n+1)$.

### 7.4 Regularized Logistic Regression

For the logistic regression problem, we have learned two optimization algorithms in the previous course: we first learned to use gradient descent to optimize the cost function $J\left( \theta \right)$, and then we learned more advanced optimization Algorithms, these advanced optimization algorithms require you to design the cost function $J\left( \theta \right)$.


![](../images/2726da11c772fc58f0c85e40aaed14bd.png)

Calculating the derivative yourself is also for logistic regression. We also add a regularized expression to the cost function to get the cost function：

$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}$

**Python** code

```python
import numpy as np

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```

To minimize the cost function, by derivation, the gradient descent algorithm is:

$Repeat$  $until$  $convergence${

​                                                   ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$

​                                                  ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$

​                                                 $for$ $j=1,2,...n$

​                                                 }
Note: It looks the same as linear regression, but knows that ${h_\theta}\left( x \right)=g\left( {\theta^T}X \right)​$, so it is different from linear regression.

