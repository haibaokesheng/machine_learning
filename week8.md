Week 8
=====
[TOC]
13.Clustering
----------------------

### 13.1  Unsupervised Learning Introduction

In a typical supervised learning, we have a labeled training set. Our goal is to find a decision boundary that can distinguish between positive and negative samples. In supervised learning here, we have a series of labels, and we need to plan accordingly Combine a hypothetical function. The difference is that in unsupervised learning, our data is not accompanied by any labels, and the data we get is like this

![](../images/6709f5ca3cd2240d4e95dcc3d3e808d5.png)

Here we have a series of points, but no labels. Therefore, our training set can be written as only $x^{(1)}$, $x^{(2)}$….. up to $x^{(m)}$. We don't have any tags $y$. Therefore, the points drawn on the graph have no label information. That is to say, in unsupervised learning, we need to input a series of unlabeled training data into an algorithm, and then we tell the algorithm to quickly find the given structure of the data for us. We may need an algorithm to help us find a structure. The data on the graph looks like it can be divided into two separate sets of points (called clusters). An algorithm that can find these sets of points I circle is called a clustering algorithm.

![](../images/6709f5ca3cd2240d4e95dcc3d3e808d5.png)

This will be the first unsupervised learning algorithm we introduced. Of course, hereafter we will also mention other types of unsupervised learning algorithms, which can find other types of structures or other patterns for us, not just clusters.

![](../images/ff180f091e9bad9ac185248721437526.png)


### 13.2 K-Means Algorithm

**K-means** is the most popular clustering algorithm. The algorithm accepts an unlabeled data set and then clusters the data into different groups.

**K-means** is an iterative algorithm. Suppose we want to cluster the data into n groups. The method is:

First select $K$ random points, called **cluster center**；

For each piece of data in the data set, according to the distance of $K$ center points, it is associated with the nearest center point, and all points associated with the same center point are grouped together.

Calculate the average value of each group and move the center point associated with the group to the average position.

Repeat steps 2-4 until the center point no longer changes.

Here is an example of clustering:

![](../images/ff1db77ec2e83b592bbe1c4153586120.jpg)

1 iteration

![](../images/acdb3ac44f1fe61ff3b5a77d5a4895a1.jpg)

3 iteration

![](../images/fe6dd7acf1a1eddcd09da362ecdf976f.jpg)

10 iteration

Use $μ^1$,$μ^2$,...,$μ^k$ to represent the clustering center, use $c^{(1)}$,$c^{(2)}$,. ..,$c^{(m)}$ to store the index of the cluster center nearest to the $i$th instance data, the pseudo code of the **K-mean** algorithm is as follows:

```
Repeat {

for i = 1 to m

c(i) := index (form 1 to K) of cluster centroid closest to x(i)

for k = 1 to K

μk := average (mean) of points assigned to cluster k

}
```

The algorithm is divided into two steps, the first **for** loop is the assignment step, that is: for each sample $i$, calculate the class it should belong to. The second **for** loop is the movement of the clustering center, that is: for each class $K$, the centroid of the class is recalculated.

The **K-means** algorithm can also be conveniently used to divide the data into many different groups, even in the absence of very distinct groups. The data set shown in the figure below is composed of two features, height and weight. The **K-means** algorithm is used to divide the data into three categories, which are used to help determine the three sizes of T-shirts to be produced.

![](../images/fed50a4e482cf3aae38afeb368141a97.png)

### 13.3  Optimization Objective

The problem of K-means minimization is to minimize the sum of the distances between all data points and the associated cluster center points, so
The cost function of K-means is:

$$J(c^{(1)},...,c^{(m)},μ_1,...,μ_K)=\dfrac {1}{m}\sum^{m}_{i=1}\left\| X^{\left( i\right) }-\mu_{c^{(i)}}\right\| ^{2}$$

Where ${{\mu }_{{{c}^{(i)}}}}$ represents the clustering center point closest to ${{x}^{(i)}}$.
Our optimization goal is to find $c^{(1)}$,$c^{(2)}$,...,$c^{(m)}$ and $μ that minimize the cost function ^1$,$μ^2$,...,$μ^k$:

![](../images/8605f0826623078a156d30a7782dfc3c.png)

Recall what was just given:
**K-means** Iterative algorithm, we know that the first loop is used to reduce the cost caused by $c^{(i)}$, and the second loop is used to reduce ${{\ The cost incurred by mu }_{i}}$. The iterative process must be that each iteration is reducing the cost function, otherwise there is an error.

### 13.4  Random Initialization

Before running the K-means algorithm, we must first initialize all clustering centers randomly. Here is how to do it:

1. We should choose $K<m$, that is, the number of clustering center points should be less than the number of all training set instances

2. Randomly select $K$ training instances, and then make $K$ clustering centers equal to these $K$ training instances, respectively

One problem with **K-means** is that it may stay at a local minimum, which depends on initialization.

![](../images/d4d2c3edbdd8915f4e9d254d2a47d9c7.png)

In order to solve this problem, we usually need to run the **K-mean** algorithm multiple times, re-initialize each time randomly, and finally compare the results of multiple runs of **K-mean**, and select the result with the smallest cost function . This method is still feasible when the $K$ is small (2--10), but if the $K$ is large, this may not improve significantly.

### 13.5 Choosing the Number of Clusters


There is no so-called best method for selecting the number of clusters, which usually needs to be selected manually according to different problems. When choosing, think about our motivation for clustering using the **K-means** algorithm, and then choose the number of standard clusters that best serve the purpose.

When people are discussing how to choose the number of clusters, one method that may be mentioned is called the "elbow rule". Regarding the "elbow rule", all we need to do is to change the value of $K$, which is the total number of clustering categories. We use a cluster to run the **K-means** clustering method. This means that all data will be divided into a cluster, and then the cost function or the distortion function $J$ is calculated. $K$ represents the clustering number.

![](../images/f3ddc6d751cab7aba7a6f8f44794e975.png)

We may get a curve similar to this. Like a person's elbow. This is what the "Elbow Rule" does. Let's look at such a picture. It looks as if there is a clear elbow there. It's like a human arm. If you extend your arm, this is your shoulder joint, elbow joint, and hand. This is the "elbow rule". You will find this mode, its distortion value will drop rapidly, from 1 to 2, from 2 to 3, you will reach an elbow point at 3. After this, the distortion value drops very slowly. It looks like it is correct to use 3 clusters for clustering. This is because that point is the elbow point of the curve, and the distortion value drops quickly. $K= After 3$, it drops slowly, so we choose $K=3$. When you apply the "elbow rule", if you get a graph like the one above, then this will be a reasonable method for selecting the number of clusters.


1.Summary of similarity/distance calculation methods

(1). **Minkowski distance**/（Euclidean distance：$p=2$) 

$dist(X,Y)={{\left( {{\sum\limits_{i=1}^{n}{\left| {{x}_{i}}-{{y}_{i}} \right|}}^{p}} \right)}^{\frac{1}{p}}}$

(2). **Jaccard**：

$J(A,B)=\frac{\left| A\cap B \right|}{\left|A\cup B \right|}$

(3). **cosine similarity**：

The angle between the $n$ dimension vectors $x$ and $y$ is denoted as $\theta$. According to the cosine theorem, the cosine value is:

$cos (\theta )=\frac{{{x}^{T}}y}{\left|x \right|\cdot \left| y \right|}=\frac{\sum\limits_{i=1}^{n}{{{x}_{i}}{{y}_{i}}}}{\sqrt{\sum\limits_{i=1}^{n}{{{x}_{i}}^{2}}}\sqrt{\sum\limits_{i=1}^{n}{{{y}_{i}}^{2}}}}$
(4). Pearson correlation coefficient：
${{\rho }_{XY}}=\frac{\operatorname{cov}(X,Y)}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{E[(X-{{\mu }_{X}})(Y-{{\mu }_{Y}})]}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{\sum\limits_{i=1}^{n}{(x-{{\mu }_{X}})(y-{{\mu }_{Y}})}}{\sqrt{\sum\limits_{i=1}^{n}{{{(x-{{\mu }_{X}})}^{2}}}}\sqrt{\sum\limits_{i=1}^{n}{{{(y-{{\mu }_{Y}})}^{2}}}}}$

The Pearson correlation coefficient is the angle cosine after the $x$ and $y$ coordinate vectors are translated to the origin respectively.

2.Measures of clustering

(1). Uniformity:$p$
Similar to the accuracy rate, a cluster contains only one category of samples, which meets uniformity. In fact, it can also be regarded as the correct rate (the ratio of the number of correctly classified samples in each cluster to the total number of samples in the cluster)

(2). Completeness:$r$

Similar to the recall rate, the samples of the same category are classified into the same cluster, then the integrity is satisfied; the sum of the number of correctly classified samples in each cluster accounts for the total number of samples of this type

(3). **V-measure**:

Weighted average of uniformity and completeness

$V = \frac{(1+\beta^2)*pr}{\beta^2*p+r}$

(4). Contour coefficient

Contour coefficient of sample $i$: $s(i)$

In-cluster dissimilarity: Calculate the average distance between sample $i$ and other samples in the same cluster as $a(i)$, which should be as small as possible.

Contour coefficient: The closer the value of $s(i)$ to 1, the more reasonable the sample $i$ clustering, and the closer to -1, the sample $i$ should be classified into another cluster, which is approximately 0, indicating the sample $i$ Should be on the boundary; the average value of $s(i)$ for all samples is used as the contour coefficient of the clustering result.

$s(i) = \frac{b(i)-a(i)}{max\{a(i),b(i)\}}$

(5). **ARI**

The data set $S$ has a total of $N$ elements. The two clustering results are:

$X=\{{{X}_{1}},{{X}_{2}},...,{{X}_{r}}\},Y=\{{{Y}_{1}},{{Y}_{2}},...,{{Y}_{s}}\}$

The number of elements in $X$ and $Y$ are:

$a=\{{{a}_{1}},{{a}_{2}},...,{{a}_{r}}\},b=\{{{b}_{1}},{{b}_{2}},...,{{b}_{s}}\}$

${{n}_{ij}}=\left| {{X}_{i}}\cap {{Y}_{i}} \right|$

$ARI=\frac{\sum\limits_{i,j}{C_{{{n}_{ij}}}^{2}}-\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)\cdot \left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]/C_{n}^{2}}{\frac{1}{2}\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)+\left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]-\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)\cdot \left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]/C_{n}^{2}}$



14 Dimensionality Reduction
------------------------------------

### 14.1 Motivation I Data Compression


![](../images/2373072a74d97a9f606981ffaf1dd53b.png)

Suppose we do not know two features: $x_1$: length: expressed in centimeters; $x_2$: the length of the same object in inches.

So, this gives us a highly redundant representation, maybe not two separate features $x_1$ and $x_2$, these two basic length measures, maybe what we want to do is reduce the data to one dimension, only one number Measure this length. This example seems a bit contrived. The centimeter-inch example here is actually not so impractical, and there is no difference between the two.

Reduce data from two dimensions to one dimension:
If we want to use two different instruments to measure the size of some things, one of the instruments measures the unit in inches, and the other instrument measures the result in centimeters. We hope to use the measured result as a feature of our machine learning. The problem now is that the results of two instruments measuring the same thing are not completely equal (due to error, accuracy, etc.), and both are used as features to repeat somewhat. Therefore, we hope to reduce this two-dimensional data to one dimension.


![](../images/2c95b316a3c61cf076ef132d3d50b51c.png)

I have been studying helicopter autopilot for many years. And so on. If you want to measure-if you want to do, you know, do a survey or do these tests for different pilots-you may have a feature: $x_1$, this may be their skill (helicopter pilot), maybe $x_2 $ May be a hobby of the pilot. This is to indicate whether they like to fly, maybe these two characteristics will be highly correlated. What you really care about may be the direction of this red line, and the different characteristics that determine the ability of the pilot.

![](../images/8274f0c29314742e9b4f15071ea7624a.png)

Reduce data from 3D to 2D:
In this example, we want to reduce a three-dimensional feature vector to a two-dimensional feature vector. The process is similar to the above, we project the three-dimensional vector onto a two-dimensional plane, forcing all the data to be on the same plane, down to the two-dimensional feature vector.

![](../images/67e2a9d760300d33ac5e12ad2bd5523c.jpg)

Such a process can be used to reduce any dimension of data to any desired dimension, for example, to reduce a 1000-dimensional feature to 100 dimensions.

As we have seen, in the end, this will allow us to make some of our learning algorithms run late, but we will mention it in a later video.

### 14.2  Motivation II Visualization


In many of its learning problems, if we can visualize the data, we can find a better solution, and dimensionality reduction can help us.

![](../images/789d90327121d3391735087b9276db2a.png)

If we have data about many different countries, each feature vector has 50 features (such as **GDP**, **GDP** per capita, average lifespan, etc.). It is impossible to visualize this 50-dimensional data. Using the dimensionality reduction method to reduce it to 2 dimensions, we can visualize it.

![](../images/ec85b79482c868eddc06ba075465fbcf.png)

The problem with this is that the dimensionality reduction algorithm is only responsible for reducing the number of dimensions, and the meaning of the newly generated features must be discovered by ourselves.

### 14.3 Principal Component Analysis Problem Formulation

Principal component analysis (**PCA**) is the most common dimensionality reduction algorithm.

In **PCA**, what we have to do is to find a direction vector (**Vector direction**). When we project all the data onto this vector, we hope that the projected average mean square error can be as much as possible. The ground is small. The direction vector is a vector passing through the origin, and the projection error is the length of the vertical line from the feature vector to the direction vector.

![](../images/a93213474b35ce393320428996aeecd9.jpg)

The following gives a description of the principal component analysis problem:

The problem is to reduce the $n$ dimension data to the $k$ dimension. The goal is to find the vector $u^{(1)}$,$u^{(2)}$,...,$u^{(k )}$ minimizes the total projection error. Comparison of principal component analysis and linear review:

Principal component analysis and linear regression are two different algorithms. Principal component analysis minimizes projected error (**Projected Error**), while linear regression attempts to minimize prediction error. The purpose of linear regression is to predict results, while principal component analysis does not make any predictions.

![](../images/7e1389918ab9358d1432d20ed20f8142.png)

In the figure above, the error on the left is linear regression (projected perpendicular to the horizontal axis), and the error on the right is principal component analysis error (projected perpendicular to the red line).

**PCA** reduces $n$ features to $k$, which can be used for data compression. If a 100-dimensional vector can be expressed in 10 dimensions at the end, the compression rate is 90%. In the same image processing field, **KL transform** uses **PCA** for image compression. But **PCA** must ensure that after the dimensionality reduction, the loss of data characteristics is minimal.

One of the major benefits of **PCA** technology is the dimensionality reduction of data. We can sort the importance of the newly obtained "principal" vector, take the most important part in front as needed, and save the following dimensions, which can achieve dimensionality reduction to simplify the model or compress the data. . At the same time, the information of the original data is kept to the greatest extent.

A big advantage of **PCA** technology is that it is completely parameter-free. In the calculation process of **PCA**, there is no need to manually set parameters or intervene in the calculation based on any empirical model. The final result is only related to the data and is independent of the user.

However, this can also be seen as a disadvantage. If the user has certain prior knowledge of the observed object and masters some characteristics of the data, but cannot intervene in the processing process through parameterization and other methods, the expected effect may not be obtained and the efficiency is not high.

### 14.4 Principal Component Analysis Algorithm

**PCA** reduce $n$ dimension to $k$ dimension:

The first step is to normalize the mean. We need to calculate the mean of all features, and then let $x_j= x_j-μ_j$. If the features are on different orders of magnitude, we also need to divide it by the standard deviation $σ^2$.

The second step is to calculate the **covariance matrix** $Σ$:
$\sum=\dfrac {1}{m}\sum^{n}_{i=1}\left( x^{(i)}\right) \left( x^{(i)}\right) ^{T}$


The third step is to calculate the **eigenvector** of the covariance matrix $Σ$:


![](../images/0918b38594709705723ed34bb74928ba.png)
$$Sigma=\dfrac {1}{m}\sum^{n}_{i=1}\left( x^{(i)}\right) \left( x^{(i)}\right) ^{T}$$

![](../images/01e1c4a2f29a626b5980a27fc7d6a693.png)

For a $n×n$ dimension matrix, the $U$ in the above formula is a matrix of direction vectors with the smallest projection error from the data. If we want to reduce the data from $n$ dimension to $k$ dimension, we only need to select the first $k$ vectors from $U$ to obtain a matrix of dimension $n×k$, we use $U_{reduce }$, and then obtain the required new feature vector by the following calculation$z^{(i)}$:
$$z^{(i)}=U^{T}_{reduce}*x^{(i)}$$


Where $x$ is in the dimension of $n×1$, so the result is the dimension of $k×1$. Note that we do not process variance features.

### 14.5 Choosing The Number Of Principal Components 


The main component analysis is to reduce the average mean square error of the projection:

The variance of the training set is：$\dfrac {1}{m}\sum^{m}_{i=1}\left\| x^{\left( i\right) }\right\| ^{2}$

We want to choose the smallest possible value of $k$ when the ratio of the average mean square error to the variance of the training set is as small as possible.

If we want this ratio to be less than 1%, it means that 99% of the original data deviations have been retained. If we choose to retain 95% of the deviations, we can significantly reduce the dimension of the features in the model.

We can first make $k=1$, and then perform the main component analysis to obtain $U_{reduce}$ and $z$, and then calculate whether the ratio is less than 1%. If not, let $k=2$, and so on, until you find the smallest $k$ value that can make the ratio less than 1% (because there is usually some correlation between the features).


### 14.6 Reconstruction from Compressed Representation

In the previous video, I talked about **PCA** as the compression algorithm. There you may need to compress 1000-dimensional data to 100-dimensional features, or have three-dimensional data compressed to a two-dimensional representation. So, if this is a compression algorithm, you should be able to go back to this compressed representation and back to an approximation of your original high-dimensional data.

So, given $z^{(i)}$, this may be 100-dimensional, how come back to your original representation of $x^{(i)}$, which may be a 1000-dimensional array?

![](../images/0a4edcb9c0d0a3812a50b3e95ef3912a.png)

**PCA** algorithm, we may have such a sample. As shown in the sample $x^{(1)}$, $x^{(2)}$. What we do is that we project these samples onto this one-dimensional plane in the figure. Then now we need to use only one real number, such as $z^{(1)}$, after specifying the positions of these points, they are projected onto this 3D surface. Given a point $z^{(1)}$, how can we go back to this original two-dimensional space? $x$ is 2 dimensions, $z$ is 1 dimension, $z=U^{T}_{reduce}x$, the opposite equation is: $x_{appox}=U_{reduce}\cdot z$,$ x_{appox}\approx x$. As shown:

![](../images/66544d8fa1c1639d80948006f7f4a8ff.png)

As you know, this is a pretty similar to the original data. So, this is how you return from the low-dimensional representation $z$ to the uncompressed representation. One of the data we get is your original data $x$, we also call this process to reconstruct the original data.

When we think of trying to reconstruct the initial value of $x$ from compression representation. So, given an unlabeled data set, you now know how to apply **PCA**, your high-dimensional feature $x$ and the low-dimensional representation mapped to it $z$. In this video, I hope you now know how to take these low-dimensional representations $z$ and map them to a backup to approximate your original high-dimensional data.


### 14.7 Advice for Applying PCA

Suppose we are doing some computer vision machine learning on a 100×100 pixel image, that is, there are a total of 10,000 features.

1. The first step is to use principal component analysis to compress the data to 1000 features

2. Then run the learning algorithm on the training set

3. When predicting, use the previously learned $U_{reduce}$ to convert the input feature $x$ into a feature vector $z$, and then make the prediction


Wrong principal component analysis: A common mistake in using principal component analysis is to use it to reduce overfitting (reducing the number of features). This is very bad, it is better to try regularization. The reason is that principal component analysis only discards some features approximately. It does not consider any information related to the result variable, so it may lose very important features. However, when we perform regularization, we will consider the result variables and will not lose important data.

Another common mistake is to use principal component analysis as a part of the learning process by default. Although it often has an effect, it is best to start with all the original features only when necessary (the algorithm runs too slowly or takes up too much Multi-memory) before considering the use of principal component analysis.
