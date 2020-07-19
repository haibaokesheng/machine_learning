9 Week
=====
[TOC]
15、Anomaly Detection
---------------------------------

### 15.1  Problem Motivation


Imagine that you are an aircraft engine manufacturer. When the aircraft engine you produce flows out of the production line, you need to perform **QA** (quality control test). As part of this test, you have measured some characteristics of the aircraft engine. Variables, such as the heat generated when the engine is running, or the vibration of the engine, etc.

![](../images/93d6dfe7e5cb8a46923c178171889747.png)

In this way, you have a data set, from $x^{(1)}$ to $x^{(m)}$, if you produce $m$ engines, you plot these data as The chart looks like this:

![](../images/fe4472adbf6ddd9d9b51d698cc750b68.png)

Every point and every fork here is your unlabeled data. In this way, the anomaly detection problem can be defined as follows: We assume that one day later, you have a new aircraft engine flowing out of the production line, and your new aircraft engine has a feature variable $x_{test}$. The so-called anomaly detection problem is: we want to know whether this new aircraft engine has some kind of anomaly, or that we want to judge whether this engine needs further testing. Because, if it looks like a normal engine, then we can ship it directly to the customer without further testing.

Given a data set $x^{(1)},x^{(2)},..,x^{(m)}$, if we have a normal data set, we want to know the new data $x_{ Is test}$ abnormal, that is, what is the probability that the test data does not belong to the set of data. The model we constructed should be able to tell us the possibility of belonging to a set of data based on the location of the test data $p(x)$.

![](../images/65afdea865d50cba12d4f7674d599de5.png)

In the above figure, the data in the blue circle is more likely to belong to this group of data, and the more remote the data, the lower the probability that it belongs to this group of data.

Fraud detection:

$x^{(i)} = {user's ith activity feature}$

The model $p(x)$ is our possibility of belonging to a set of data, and abnormal users are detected by $p(x) <\varepsilon$.

Anomaly detection is mainly used to identify fraud. For example, the data about users collected online, a feature vector may contain such as: how often users log in, the pages visited, the number of posts published in the forum, and even the typing speed. Try to build a model based on these characteristics, you can use this model to identify users who do not meet the pattern.


### 15.2 Gaussian Distribution


Usually if we think that the variable $x$ conforms to the Gaussian distribution $x \sim N(\mu, \sigma^2)$ then the probability density function is:
$p(x,\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
We can use the existing data to predict the calculation method of $μ$ and $σ^2$ in the population as follows:
$\mu=\frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}$


$\sigma^2=\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}-\mu)^2$

Sample Gaussian distribution:

![](../images/fcb35433507a56631dde2b4e543743ee.png)

Note: For machine learning, we usually only divide $m$ by the variance rather than $(m-1)$ in statistics. Here, by the way, in actual use, the difference between choosing whether to use $1/m$ or $1/(m-1)$ is actually very small. As long as you have a fairly large training set, most people in the field of machine learning I am more used to using the formula of $1/m$. The two versions of the formula are slightly different in theoretical and mathematical characteristics, but in actual use, the difference between them is very small and can be ignored.

### 15.3 Algorithm 

In this video, I will use Gaussian distribution to develop anomaly detection algorithms.

Anomaly detection algorithm:

For a given data set $x^{(1)},x^{(2)},...,x^{(m)}$, we need to calculate $\mu$ and $\ for each feature Estimated value of sigma^2$.

$\mu_j=\frac{1}{m}\sum\limits_{i=1}^{m}x_j^{(i)}$

$\sigma_j^2=\frac{1}{m}\sum\limits_{i=1}^m(x_j^{(i)}-\mu_j)^2$

Once we have obtained estimates of the mean and variance, given a new training instance, calculate $p(x)$ according to the model:

$p(x)=\prod\limits_{j=1}^np(x_j;\mu_j,\sigma_j^2)=\prod\limits_{j=1}^1\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})$

When $p(x) <\varepsilon$, it is abnormal.

The following figure is a training set consisting of two features, and the distribution of features:

![](../images/ba47767a11ba39a23898b9f1a5a57cc5.png)

The following three-dimensional graph represents the density estimation function. The $z$ axis is the estimated value of $p(x)$ based on the values of two features:

![](../images/82b90f56570c05966da116c3afe6fc91.jpg)

We select a $\varepsilon$, and use $p(x) = \varepsilon$ as our decision boundary. When $p(x)> \varepsilon$, the predicted data is normal data, otherwise it is abnormal.

In this video, we introduced how to fit $p(x)$, which is the probability value of $x$, to develop an anomaly detection algorithm. At the same time, in this lesson, we also give the parameters obtained by fitting the data set to estimate the parameters, get the parameters $\mu$ and $\sigma$, and then detect the new sample to determine whether the new sample is abnormal.


### 15.4 Developing and Evaluating an Anomaly Detection System

The anomaly detection algorithm is an unsupervised learning algorithm, which means that we cannot tell us whether the data is really anomalous based on the value of the result variable $ y$. We need another method to help test whether the algorithm is effective. When we develop an anomaly detection system, we start with labeled (abnormal or normal) data, we select a part of normal data from them to build a training set, and then use the remaining normal data and abnormal data mixed data to form a cross Inspection set and test set.

For example: we have data for 10,000 normal engines and data for 20 abnormal engines. We distribute the data like this:

Data from 6000 normal engines is used as training set

Data from 2000 normal engines and 10 abnormal engines are used as cross-check sets

Data from 2000 normal engines and 10 abnormal engines are used as test sets
The specific evaluation methods are as follows:

1. Based on the test set data, we estimate the mean and variance of the features and construct the $p(x)$ function

2. For the cross-check set, we try to use different $\varepsilon$ values as thresholds, and predict whether the data is abnormal, and select $\varepsilon$ according to the value of $F1$ or the ratio of the precision rate to the recall rate.

3. After selecting $\varepsilon$, make a prediction for the test set and calculate the $F1$ value of the anomaly inspection system, or the ratio of the precision rate to the recall rate

### 15.5 Anomaly Detection vs. Supervised Learning



| Anomaly Detection                                | Supervised Learning                                     |
| ----------------------------------- | ---------------------------------------- |
| Very few positive classes (abnormal data $y=1$), a large number of negative classes ($y=0$) | There are also a large number of positive and negative classes                            |
| Many different kinds of anomalies are very difficult. The algorithm is trained based on a very small amount of forward class data.   | There are enough instances of the forward class, enough to train the algorithm. The instances of the forward class encountered in the future may be very similar to the training set. |
| The anomalies encountered in the future may be very different from the anomalies already mastered.            |                                          |
| For example: fraud detection production (such as aircraft engines) detection of computer health in data centers | For example: mail filter weather forecast tumor classification     |



### 15.6 Choosing What Features to Use


Anomaly detection assumes that the features conform to a Gaussian distribution. If the distribution of the data is not Gaussian, the anomaly detection algorithm can work, but it is best to convert the data to a Gaussian distribution, for example, using a logarithmic function: $x= log(x+c)$ , Where $c$ is a non-negative constant; or $x=x^c$, $c$ is a fraction between 0-1, etc. (Editor's note: In **python**, the `np.log1p()` function is usually used. $log1p$ is $log(x+1)$, which can avoid negative results. The reverse function is `np.expm1 ()`)

![](../images/0990d6b7a5ab3c0036f42083fe2718c6.jpg)

Error Analysis：

A common problem is that some abnormal data may also have a higher value of $p(x)$, which is considered normal by the algorithm. In this case, error analysis can help us. We can analyze the data that is incorrectly predicted by the algorithm as normal and observe whether we can find some problems. We may be able to find from the problem that we need to add some new features, and the new algorithm obtained after adding these new features can help us better perform anomaly detection.

Anomaly detection error analysis：

![](../images/f406bc738e5e032be79e52b6facfa48e.png)

We can usually get some new and better features by combining some related features (the feature value of abnormal data is abnormally large or small), for example, in the example of detecting the computer condition of the data center, we can Use the ratio of **CPU** load to network traffic as a new feature. If the value is abnormally large, it may mean that the server is caught in some problems.


16、Recommender Systems
-----------------------------------

### 16.1 Problem Formulation


We start with an example to define the problem of the recommendation system.

If we are a movie supplier, we have 5 movies and 4 users, and we ask users to rate the movies.

![](../images/c2822f2c28b343d7e6ade5bd40f3a1fc.png)

The first three films are love movies, and the last two are action movies. We can see that **Alice** and **Bob** seem to be more inclined to love movies, while **Carol** and **Dave** Seems more inclined to action movies. And no one user has rated all the movies. We hope to build an algorithm to predict how much each of them might rate movies they haven’t watched and use it as a basis for recommendation.

Here are some tags:

$n_u$ represents the number of users

$n_m$ represents the number of movies

$r(i, j)$ If user j rated the movie $i$ then $r(i,j)=1$

$y^{(i, j)}$ on behalf of user $j$ to rate movie $i$

$m_j$ represents the total number of movies that user $j$ has rated

### 16.2  Content Based Recommendations 

In a content-based recommendation system algorithm, we assume that there are some data for what we want to recommend, and these data are the characteristics of these things.

In our example, we can assume that each movie has two characteristics, such as $x_1$ for the romance of the movie, and $x_2$ for the action of the movie.

![](../images/747c1fd6bff694c6034da1911aa3314b.png)

Then each movie has a feature vector, such as $x^{(1)}$ is the feature vector of the first movie is [0.9 0].

Below we will build a recommendation system algorithm based on these features.
Assuming that we use a linear regression model, we can train a linear regression model for each user. For example, ${{\theta }^{(1)}}$ is the parameter of the first user’s model.
So, we have:

$\theta^{(j)}$ Parameter vector of user $j$

Feature vector of $x^{(i)}$movie $i$

For user $j$ and movie $i$, we predict the score: $(\theta^{(j)})^T x^{(i)}$

Cost function

For the user $j$, the cost of this linear regression model is the sum of squared prediction errors, plus the regularization term:
$$
\min_{\theta (j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\left(\theta_{k}^{(j)}\right)^2
$$

Among them, $i:r(i,j)$ means that we only count movies that have been rated by user $j$. In the general linear regression model, both the error term and the regular term should be multiplied by $1/2m$, where we remove $m$. And we do not regularize the variance item $\theta_0$.

The above cost function is only for one user. In order to learn all users, we sum the cost functions of all users:
$$
\min_{\theta^{(1)},...,\theta^{(n_u)}} \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2
$$
If we want to use the gradient descent method to solve the optimal solution, we calculate the partial derivative of the cost function and get the updated formula of gradient descent as:

$$
\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)} \quad (\text{for} \, k = 0)
$$

$$
\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}+\lambda\theta_k^{(j)}\right) \quad (\text{for} \, k\neq 0)
$$



### 16.3 Collaborative Filtering

In the previous content-based recommendation system, for each movie, we have mastered the available features, and used these features to train the parameters of each user. Conversely, if we have user parameters, we can learn the characteristics of the movie.

$$
\mathop{min}\limits_{x^{(1)},...,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j{r(i,j)=1}}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
$$
But if we have neither user parameters nor movie features, neither of these methods is feasible. Collaborative filtering algorithms can learn both at the same time.

Our optimization goal was changed to $x$ and $\theta$ at the same time.
$$
J(x^{(1)},...x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

The result of finding the partial derivative of the cost function is as follows:

$$
x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}+\lambda x_k^{(i)}\right)
$$

$$
\theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}+\lambda \theta_k^{(j)}\right)
$$

Note: In the collaborative filtering slave algorithm, we usually do not use the variance term, if necessary, the algorithm will automatically learn.
The steps of collaborative filtering algorithm are as follows:

1. Initial $x^{(1)},x^{(1)},...x^{(nm)},\ \theta^{(1)},\theta^{(2)}, ...,\theta^{(n_u)}$ is some random small value

2. Use gradient descent algorithm to minimize the cost function

3. After training the algorithm, we predict that $(\theta^{(j)})^Tx^{(i)}$ is the score given by user $j$ to the movie $i$

The feature matrix obtained through this learning process contains important data about movies. These data are not always readable by humans, but we can use these data as a basis for recommending movies to users.

For example, if a user is watching the movie $x^{(i)}$, we can find another movie $x^{(j)}$, based on the distance between the feature vectors of the two movies $\left \| {{x}^{(i)}}-{{x}^{(j)}} \right\|$ size.

### 16.4 Collaborative Filtering Algorithm


Collaborative filtering optimization goals:

Given $x^{(1)},...,x^{(n_m)}$, estimated $\theta^{(1)},...,\theta^{(n_u)}$:
$$
\min_{\theta^{(1)},...,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2
$$


Given $\theta^{(1)},...,\theta^{(n_u)}$，estimated$x^{(1)},...,x^{(n_m)}$：

At the same time minimize $x^{(1)},...,x^{(n_m)}$和$\theta^{(1)},...,\theta^{(n_u)}$：
$$
J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

$$
\min_{x^{(1)},...,x^{(n_m)} \\\ \theta^{(1)},...,\theta^{(n_u)}}J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})
$$