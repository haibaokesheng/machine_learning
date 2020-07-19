

Seventh Week
=====
[TOC]
12.Support Vector Machines
-----------------------------------------

### 12.1 Optimization Objective

As with the learning algorithm we developed earlier, we start with the optimization goal. So, we start learning this algorithm. In order to describe the support vector machine, in fact, I will start from logistic regression to show how we can modify it little by little to get the essential support vector machine.

![](../images/3d12b07f13a976e916d0c707fd03153c.png)

Well, in logistic regression, we are already familiar with the hypothetical function form here, and the S-type excitation function on the right. However, in order to explain some mathematical knowledge. I will use $z$ for $\theta^Tx$.

Now consider what we want to do for logistic regression: if there is a sample of $y=1$, I mean whether it is in the training set or the test set, or in the cross-validation set, in short $y=1 $, now we want ${{h}_{\theta }}\left( x \right)$ to approach 1. Because we want to classify this sample correctly, this means that when ${{h}_{\theta }}\left( x \right)$ approaches 1, $\theta^Tx$ should be much greater than 0, where $>>$ means much greater than 0. This is because $z$ means $\theta^Tx$. When $z$ is much greater than 0, that is, to the right of the graph, it is not difficult to find that the output of logistic regression will approach 1 at this time. Conversely, if we have another sample, $y=0$. We want to assume that the output value of the function will approach 0, which corresponds to $\theta^Tx$, or that $z$ will be much smaller than 0, because the output value of the corresponding hypothesis function approaches 0.

![](../images/66facb7fa8eddc3a860e420588c981d5.png)

If you further observe the cost function of logistic regression, you will find that each sample $(x,y)$ will be the total cost function, and increase one of the items here. Therefore, for the total cost function, there are usually all training samples And, and there is also a $1/m$ term, but in logistic regression, this term here is the expression corresponding to a training sample. Now, if I substitute a fully defined hypothetical function here. Then, we will get that every training sample affects this item.

Now, ignore the term $1/m$, but this term affects this term in the total cost function.

Now, let's consider two situations together:

One is when $y$ is equal to 1; the other is when $y$ is equal to 0.

In the first case, assuming $y=1$, only the first item in the objective function will work, because when $y=1$, the $(1-y)$ item will be equal to 0. Therefore, when in the sample of $y=1$, that is, in $(x, y) $, we get $y=1$ $-\log(1-\frac{1}{1+e^{ -z}})$ This is the same as the previous slide.

I use $z$ to represent $\theta^Tx$, that is: $z= \theta^Tx$. Of course, in the cost function, $y$ is preceded by a minus sign. We just say this, if $y=1$ in the cost function, this term is also equal to 1. This is done to simplify the expression here. If you draw a function about $z$, you will see this curve in the lower left corner. We can also see that when $z$ increases, it is equivalent to $\theta^Tx$ increasing, The value corresponding to $z$ will become very small.For the entire cost function, the impact is also very small. This explains why logistic regression attempts to set $\theta^Tx$ very large when it observes positive samples $y=1$. Because, this term in the cost function will become very small.

Now to build a support vector machine, we start here:

We will start with this cost function, which is $-\log(1-\frac{1}{1+e^{-z}})$ and modify it little by little, let me take the $z=1$ here Point, I first draw the cost function to be used.

![](../images/b4b43ee98bff9f5e73d841af1fa316bf.png)

The new cost function will go horizontally from here to the right (outside of the picture), and then I draw a straight line that is very similar to logistic regression, but here is a straight line, which is the curve I draw with fuchsia, this is it Fuchsia curve. Well, here we are very close to the cost function used in logistic regression. It's just that it consists of two line segments, that is, the horizontal part on the right and the straight part on the left. Don't think too much about the slope of the straight part on the left. This is not very important. However, the new cost function we will use here is on the premise of $y=1$. You might think that this should do the same thing as logistic regression, but in fact, it will become more firm in the optimization problem afterwards, and it will bring computational advantages to support vector machines. For example, it is easier to calculate the problem of stock trading and so on.


At present, we have only discussed the case of $y=1$. Another case is when $y=0$. At this time, if you carefully observe the cost function, only the second term is left because the first term is eliminated. . If $y=0$, then this item is also 0. So the above expression leaves only the second term. Therefore, the cost of this sample or the contribution of the cost function. Will be represented by this item. And, if you use this term as a function of $z$, then you will get the horizontal axis $z$. Now that you have completed part of the support vector machine, similarly, we will replace this blue line in a similar way.

![](../images/ab372c9161375a4f7b6f0bd4a69560e9.png)

If we use a new cost function instead, that is, this horizontal straight line starting at 0, and then a diagonal line, as shown above. So, let me name these two equations now. The function on the left, which I call ${\cos}t_1{(z)}$, and the function on the right, I call it ${\cos}t_0{(z )}$. The subscript here refers to the corresponding cases of $y=1$ and $y=0$ in the cost function. After having these definitions, now, we start to build the support vector machine.

![](../images/59541ab1fda4f92d6f1b508c8e29ab1c.png)

This is how we use the cost function $J(\theta)$ in logistic regression. Maybe this equation doesn't look very familiar. This is because there was a minus sign outside the equation, but what I did here was to move the minus sign into the expression. This makes the equation look a little different. For support vector machines, essentially we want to replace this with ${\cos}t_1{(z)}$, which is ${\cos}t_1{(\theta^Tx)}$. Similarly, I Also replace this item with ${\cos}t_0{(z)}$, which is the cost ${\cos}t_0{(\theta^Tx)}$. The cost function ${\cos}t_1$ here is the line mentioned earlier. In addition, the cost function ${\cos}t_0$ is also the line introduced above. Therefore, for the support vector machine, we get the minimization problem here, namely:

![](../images/4ac1ca54cb0f2c465ab81339baaf9186.png)

Then, add regularization parameters. Now, according to the convention of support vector machines, in fact, our writing will be slightly different, and the parameter representation of the cost function will also be slightly different.

First of all, we have to remove the item $1/m$. Of course, this is only due to different habits when people use support vector machines compared to logistic regression, but what I mean here is: you know , What I'm going to do is just remove the item $1/m$, but this will also result in the same optimal value of ${{\theta }}$, okay, because $1/m$ is only a constant, so , You know that in this minimization problem, no matter whether there is $1/m$ in front, the final optimal value I get is ${{\theta }}$. What I mean here is to give you a sample first, assuming that there is a minimization problem: that is, the $u$ value when $(u-5)^2+1$ is required to obtain the minimum value, then the minimum value is: when Get the minimum value when $u=5$.


Now, if we want to multiply this objective function by a constant of 10, here minimization problem becomes: find the minimum value $u$ that makes $10×(u-5)^2+10$, however, making The minimum $u$ value here is still 5. Therefore, multiplying some constants by your minimization term will not change the value of $u$ when minimizing the equation. Therefore, what I did here is to delete the constant $m$. In the same way, I multiply the objective function by a constant $m$ and it will not change the value of ${{\theta }}$ when the minimum value is obtained.
The second conceptual change, we only refer to the following standard conventions when using support vector machines, not logistic regression. Therefore, for logistic regression, in the objective function, we have two items: the first is the cost of the training sample, and the second is our regularization term. We have to use this term to balance. This is equivalent to we want to minimize $A$ plus the regularization parameter $\lambda$, and then multiply it by other items $B$, right? $A$ here means the first item here, and I use **B** to mean the second item, but does not include $\lambda$, we are not optimizing $A+\lambda\times B$ here. What we have done is to optimize by setting different regular parameters $\lambda$. In this way, we can weigh the corresponding items to make the training samples fit better. That is, minimize $A$. It is still guaranteed that the regular parameter is small enough, that is, for the **B** item, but for the support vector machine, by convention, we will use a different parameter to replace the $\lambda$ used here to weigh the two. You know, the first item and the second item we use a different parameter according to the convention called $C$, and at the same time changed to the optimization goal, $C×A+B$.


Therefore, in logistic regression, if $\lambda$ is given, a very large value means that $B$ is given greater weight. And here, it corresponds to setting $C$ to a very small value, then, correspondingly, $B$ will be given a greater weight than $A$. Therefore, this is just a different way to control this trade-off or a different method, that is, to use parameters to decide whether to care more about the first optimization or the second optimization. Of course, you can also consider the parameter $C$ here as $1/\lambda$, which plays the same role as $1/\lambda$, and these two equations or these two expressions are not the same, because $C= 1/\lambda$, but this is not always the case. If $C=1/\lambda$, these two optimization goals should get the same value, the same optimal value ${{\theta }}$. Therefore, use them instead. So, I now delete $\lambda$ here and replace it with the constant $C$. Therefore, this gives us the entire optimization objective function in the support vector machine. Then minimize this objective function to get the parameter $C$ learned by **SVM**.

![](../images/5a63e35db410fdb57c76de97ea888278.png)

Finally, it is different from the probability of logistic regression output. Here, our cost function, when the cost function is minimized and the parameter ${{\theta }}$ is obtained, what the support vector machine does is directly predict whether the value of $y$ is equal to 1 or equal to 0. Therefore, this hypothetical function predicts 1. When $\theta^Tx$ is greater than or equal to 0, or equal to 0, the learning parameter ${{\theta }}$ is the form of the support vector machine hypothesis function. Well, this is the mathematical definition of support vector machine.


### 12.2 12 - 2 - Large Margin Intuition

![](../images/cc66af7cbd88183efc07c8ddf09cbc73.png)

This is the cost function of my support vector machine model. On the left, I have drawn the cost function ${\cos}t_1{(z)}$ about $z$. This function is used for positive samples, and it is here on the right. I drew the cost function ${\cos}t_0{(z)}$ about $z$. The horizontal axis represents $z$. Now let us consider what are the necessary conditions to minimize these cost functions. If you have a positive sample, $y=1$, then the cost function ${\cos}t_1{(z)}$ is equal to 0 only when $z>=1$.

In other words, if you have a positive sample, we would like $\theta^Tx>=1$, and conversely, if $y=0$, we observe that the function ${\cos}t_0{(z)}$ , It only has a function value of 0 in the range of $z<=-1$. This is an interesting property of support vector machines. In fact, if you have a positive sample $y=1$, in fact, we only require $\theta^Tx$ to be greater than or equal to 0, and the sample can be properly separated, because if $\theta^Tx$\ >0 is large, our model cost function value is 0, similarly, if you have a negative sample, you only need $\theta^Tx$\<=0 to separate the negative examples correctly, but support vector machines The requirements are higher, not only to correctly separate the input samples, that is, not only $\theta^Tx$\>0, we need to be much larger than 0, such as greater than or equal to 1, I also want this ratio to be 0 Much smaller, such as I want it to be less than or equal to -1, which is equivalent to embedding an additional safety factor, or a safe spacing factor, in the support vector machine.

Of course, logistic regression does something similar. But let's take a look at what results this factor will cause in support vector machines. Specifically, I will consider a special case next. We set this constant $C$ to a very large value. For example, let's assume that the value of $C$ is 100,000 or other very large numbers, and then observe the results of support vector opportunities.


![](../images/12ebd5973230e8fdf279ae09e187f437.png)

If $C$ is very large, when minimizing the cost function, we will hopefully find an optimal solution that makes the first term zero. Therefore, let us try to understand the optimization problem when the first term of the cost term is 0. For example, we can set $C$ to a very large constant, which will give us some intuitive feelings about the support vector machine model.

We have seen that entering a training sample label is $y=1​$, you want to make the first item 0, all you need to do is find a ${{\theta }}​$, making $\theta^Tx> =1​$, similarly, for a training sample, the label is $y=0​$, in order to make the value of the ${\cos}t_0{(z)}​$ function 0, we need $\theta^Tx <=-1​$. Therefore, now consider our optimization problem. Selecting the parameter so that the first item is equal to 0 will lead to the following optimization problem, because we will select the parameter so that the first item is 0, so the first item of this function is 0, so it is $C​$ multiplied by 0 plus Multiply the upper half by the second term. The first item here is $C$$ multiplied by 0, so it can be deleted because I know it is 0.

This will obey the following constraint: $\theta^Tx^{(i)}>=1$, if $y^{(i)}$ is equal to 1, $\theta^Tx^{(i)}< =-1$, if the sample $i$ is a negative sample, so when you solve this optimization problem, when you minimize this function about the variable ${{\theta }}$, you will get a very Interesting decision boundaries.

![](../images/b1f670fddd9529727aa16a559d49d151.png)

Specifically, if you look at such a data set with positive samples and negative samples, you can see that this data set is linearly separable. I mean, there is a straight line separating positive and negative samples. Of course, there are many different straight lines that can completely separate the positive and negative samples.

![](../images/01105c3afd1315acf0577f8493137dcc.png)

For example, this is a decision boundary that can separate positive samples from negative samples. But more or less, this doesn't seem very natural, right?

Or we can draw a worse decision boundary, which is another decision boundary, which can separate positive samples and negative samples, but only barely. These decision boundaries do not seem to be particularly good choices. Support vector machines will Choose this black decision boundary, compared to the decision boundary I painted in pink or green. This black one looks much better, and the black line seems to be a more robust decision-making world. It looks better in separating positive and negative samples. Mathematically speaking, what does this mean? This black line has a greater distance. This distance is called the **margin**.

![](../images/e68e6ca3275f433330a7981971eb4f16.png)

When drawing these two additional blue lines, we see a larger shortest distance between the black decision boundary and the training sample. However, the pink line and the blue line are very close to the training sample, and will perform worse than the black line when separating the sample. Therefore, this distance is called the spacing of the support vector machine, and this is the reason why the support vector machine is robust because it strives to separate samples with a maximum spacing. Therefore, support vector machines are sometimes called **large-space classifiers**, and this is actually the result of solving the optimization problem on the previous slide.

I know you may be wondering why solving the optimization problem from the previous slide produced this result? How did it produce this large-spacing classifier? I know I have not explained this.

![](../images/dd6239efad3d3ee7a89a28574d7795b3.png)

We set the regularization factor constant $C$ in this large-spacing classifier to be very large, I remember I set it to 100000, so for such a data set, maybe we will choose such a decision boundary, so that the maximum spacing Separate positive and negative samples. So in the process of minimizing the cost function, we hope to find the parameters that make the left term in the cost function as zero as possible in both cases of $y=1$ and $y=0$. If we find such a parameter, our minimization problem becomes:

![](../images/f4b6dee99cfb4352b3cac5287002e8de.png)

In fact, support vector machines are now more mature than this large-pitch classifier, especially when you use a large-pitch classifier, your learning algorithm will be affected by outliers. For example, we add an additional positive sample.

![](../images/b8fbe2f6ac48897cf40497a2d034c691.png)

Here, if you add this sample, in order to separate the samples by the maximum distance, maybe I will eventually get a decision boundary like this, right? It is this pink line, based on only one outlier and only one sample, that changed my decision-making circle from this black line to this pink line, which is really unwise. If the regularization parameter $C$ is set very large, this is actually what the support vector machine will do. It will change the decision-making circle from a black line to a pink line, but if $C$ is set smaller, **If you do not set C too big, you will eventually get this black line,**Of course if the data is Not linearly separable, if you have some positive samples here or you have some negative samples here, the support vector machine will also separate them appropriately. Therefore, the description of the large-spacing classifier only intuitively gives the case where the regularization parameter $C$ is very large, and at the same time, it is necessary to remind you that the role of $C$ is similar to $1/\lambda$, $\lambda$ Is the regularization parameter we used before. This is only the case where $C$ is very large, or equivalently $\lambda$ is very small. You will end up with decision circles like pink lines, but when applying support vector machines, **When $C$ is not very, very large, it can ignore the influence of some abnormal points and get a better decision boundary.** Even when your data is not linearly separable, support vector machines can give good results.

Recalling $C=1/\lambda$, therefore:

When $C$ is large, it is equivalent to $\lambda$ being small, which may lead to overfitting and high variance.

When $C$ is small, it is equivalent to $\lambda$ being large, which may cause low fitting and high deviation.


### 12.4 Kernels I 

Recall that we discussed previously that polynomial models with advanced numbers can be used to solve classification problems that cannot be separated by straight lines:

![](../images/529b6dbc07c9f39f5266bd0b3f628545.png)

In order to obtain the decision boundary shown in the figure above, our model may be${{\theta }_{0}}+{{\theta }_{1}}{{x}_{1}}+{{\theta }_{2}}{{x}_{2}}+{{\theta }_{3}}{{x}_{1}}{{x}_{2}}+{{\theta }_{4}}x_{1}^{2}+{{\theta }_{5}}x_{2}^{2}+\cdots $。

We can replace each item in the model with a series of new features $f$. For example:
${{f}_{1}}={{x}_{1}},{{f}_{2}}={{x}_{2}},{{f}_{3}}={{x}_{1}}{{x}_{2}},{{f}_{4}}=x_{1}^{2},{{f}_{5}}=x_{2}^{2}$

...get $h_θ(x)={{\theta }_{1}}f_1+{{\theta }_{2}}f_2+...+{{\theta }_{n}}f_n$. However, besides combining the original features, is there a better way to construct $f_1,f_2,f_3$? We can use the kernel function to calculate new features.

Given a training sample $x$, we use each feature of $x$ and our pre-selected **landmarks**$l^{(1)},l^{(2) },l^{(3)}$ to select new features $f_1,f_2,f_3$.

![](../images/2516821097bda5dfaf0b94e55de851e0.png)

for example ${{f}_{1}}=similarity(x,{{l}^{(1)}})=e(-\frac{{{\left\| x-{{l}^{(1)}} \right\|}^{2}}}{2{{\sigma }^{2}}})$

Among them: ${{\left\| x-{{l}^{(1)}} \right\|}^{2}}=\sum{_{j=1}^{n}}{{( {{x}_{j}}-l_{j}^{(1)})}^{2}}$, which is between all the features in the instance $x$ and the landmark $l^{(1)}$ Of the distance. The $similarity(x,{{l}^{(1)}})$ in the above example is the kernel function, specifically, here is a **Gaussian kernel function** (**Gaussian Kernel**).

What is the role of these landmarks? If the distance between a training sample $x$ and the landmark $l$ is approximately 0, the new feature $f$ is approximately $e^{-0}=1$, if the training sample $x$ and the landmark $l$ If the distance between them is far, then $f$ is similar to $e^{-(a larger number)}=0$.

Suppose our training sample contains two features [$x_{1}$ $x{_2}$], given the landmark $l^{(1)}$ and different values of $\sigma$, see the figure below:

![](../images/b9acfc507a54f5ca13a3d50379972535.jpg)

The coordinates of the horizontal plane in the figure are $x_{1}$, $x_{2}$ and the vertical axis represents $f$. It can be seen that only when $x$ and $l^{(1)}$ coincide, $f$ has the maximum value. With the change of $x$, the rate of change of $f$ value is controlled by $\sigma^2$.

In the figure below, when the sample is at the magenta dot position, because it is closer to $l^{(1)}$, but it is closer to $l^{(2)}$ and $l^{(3)}$ It is farther away, so $f_1$ is close to 1, and $f_2$,$f_3$ is close to 0. So $h_θ(x)=θ_0+θ_1f_1+θ_2f_2+θ_1f_3>0$, so predict $y=1$. Similarly, we can find that for green points closer to $l^{(2)}$, we also predict $y=1$, but for blue-green points, because they are farther from the three landmarks, we predict $ y=0$.

![](../images/3d8959d0d12fe9914dc827d5a074b564.jpg)

In this way, the range represented by the red closed curve in the figure is the decision boundary obtained by us based on a single training sample and the landmark we selected. When predicting, the features we use are not the features of the training sample itself, but New features calculated by kernel function $f_1,f_2,f_3$

### 12.5  Kernels II

How to choose a landmark?

We usually choose the number of landmarks according to the number of training sets, that is, if there are $m$ samples in the training set, we select $m$ landmarks, and let: $l^{(1)}=x^{(1 )},l^{(2)}=x^{(2)},.....,l^{(m)}=x^{(m)}$. The advantage of this is that the new feature we get now is based on the distance between the original feature and all other features in the training set, namely:

![](../images/eca2571849cc36748c26c68708a7a5bd.png)

![](../images/ea31af620b0a0132fe494ebb4a362465.png)

Below we apply the kernel function to the support vector machine and modify our support vector machine hypothesis as follows:

Given $x$, calculate the new feature $f$, when $θ^Tf>=0$, predict $y=1$, otherwise the opposite.

Modify the cost function accordingly：$\sum{_{j=1}^{n=m}}\theta _{j}^{2}={{\theta}^{T}}\theta $，

$min C\sum\limits_{i=1}^{m}{[{{y}^{(i)}}cos {{t}_{1}}}( {{\theta }^{T}}{{f}^{(i)}})+(1-{{y}^{(i)}})cos {{t}_{0}}( {{\theta }^{T}}{{f}^{(i)}})]+\frac{1}{2}\sum\limits_{j=1}^{n=m}{\theta _{j}^{2}}$
In the specific implementation process, we also need to make some slight adjustments to the final regularization term, in calculating $\sum{_{j=1}^{n=m}}\theta _{j}^{2}={ {\theta}^{T}}\theta $, we use $θ^TMθ$ instead of $θ^Tθ$, where $M$ is a matrix that differs according to the kernel function we choose. The reason for this is to simplify the calculation.

In theory, we can also use kernel functions in logistic regression, but the above method of using $M$ to simplify the calculation is not applicable to logistic regression, so the calculation will be very time-consuming.

In addition, the support vector machine can also use no kernel function, which is also known as **linear kernel function** (**linear kernel**), when we do not use very complex functions, or our training set features When there are many and very few samples, this support vector machine without kernel function can be used.

The following are the effects of the two parameters $C$ and $\sigma$ of the support vector machine:

$C=1/\lambda$

When $C$ is large, it is equivalent to $\lambda$ being small, which may lead to overfitting and high variance;

When $C$ is small, it is equivalent to $\lambda$ being large, which may cause low fitting and high deviation;

When $\sigma$ is large, it may cause low variance and high deviation;

When $\sigma$ is small, it may cause low deviation and high variance.

### 12.6 Using An SVM

In addition to the Gaussian kernel function, we have other options, such as:

(**Polynomial Kerne**l)

(**String kernel**)

(**chi-square kernel**)

(**histogram intersection kernel**)

The goal of these kernel functions is also to construct new features based on the distance between the training set and the landmark. These kernel functions need to meet Mercer's theorem before they can be correctly processed by the optimization software of the support vector machine.

Multi-class classification problem

Suppose we use the one-to-many method introduced earlier to solve a multi-class classification problem. If there are $k$ classes, we need $k$ models and $k$ parameter vectors ${{\theta }}$. We can also train $k$ support vector machines to solve multi-class classification problems. But most support vector machine software packages have built-in multi-class classification function, we just use it directly.

Although you don’t write your own SVM optimization software, you also need to do a few things:

1. The selection of the parameter $C$ is proposed. We discussed the nature of the error/variance in this regard in the previous video.

2. You also need to select the kernel parameters or similar functions you want to use. One of the options is: we choose the concept that does not require any kernel parameters and there is no kernel parameters. It is also called a linear kernel function. Therefore, if someone says that he uses linear kernel **SVM** (support vector machine), it means that he uses **SVM** (support vector machine) without kernel function.

From the logistic regression model, we got the support vector machine model. Between the two, how should we choose?

**Here are some commonly used guidelines:**

$n$ is the number of features, and $m$ is the number of training samples.

(1) If compared to $m$, $n$ is much larger, that is, the amount of data in the training set is not enough to support us to train a complex nonlinear model, we choose logistic regression model or support vector machine without kernel function .

(2) If $n$ is small and $m$ is of medium size, for example, $n$ is between 1-1000 and $m$ is between 10-10000, use the support vector machine of Gaussian kernel function.

(3) If $n$ is small and $m$ is large, for example, $n$ is between 1-1000, and $m$ is greater than 50000, the chance of using support vectors is very slow, the solution is to create, add more Multiple features, then use logistic regression or support vector machines without kernel functions.

It is worth mentioning that the neural network may perform better in the above three cases, but the training of the neural network may be very slow. The main reason for selecting the support vector machine is that its cost function is a convex function, and there is no local The minimum value.