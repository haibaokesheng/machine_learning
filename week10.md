Week Ten
======
[TOC]
17、Large Scale Machine Learning
--------------------------------------------------

### 17.1  Learning With Large Datasets 


If we have a low-variance model, increasing the size of the data set can help you get better results. How should we deal with a training set with 1 million records?

Taking the linear regression model as an example, every time the gradient descent iteration, we need to calculate the square sum of the error of the training set. If our learning algorithm needs 20 iterations, this is already a very large calculation cost.

The first thing to do is to check whether a training set of this scale is really necessary. Maybe we can get better results with only 1000 training sets. We can draw a learning curve to help judge.

![](../images/bdf069136b4b661dd14158496d1d1419.png)

### 17.2 Stochastic Gradient Descent

If we must need a large-scale training set, we can try to use stochastic gradient descent instead of batch gradient descent.

In the stochastic gradient descent method, we define the cost function as the cost of a single training instance:

​                                                           $$cost\left(  \theta, \left( {x}^{(i)} , {y}^{(i)} \right)  \right) = \frac{1}{2}\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{{(i)}} \right)^{2}$$

**Random** The gradient descent algorithm is: first randomly "shuffle" the training set, then:
 Repeat (usually anywhere between1-10){

  **for** $i = 1:m${

 ​       $\theta:={\theta}_{j}-\alpha\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{(i)} \right){{x}_{j}}^{(i)}$      

​        (**for** $j=0:n$)

 ​    }
 }

The stochastic gradient descent algorithm updates the parameter ${{\theta }}$ after each calculation, without first summing all the training sets. When the gradient descent algorithm has not completed an iteration, the random gradient descent algorithm is already Walked a long way. But the problem with such an algorithm is that not every step is taken in the "correct" direction. Therefore, although the algorithm will gradually move toward the position of the global minimum, it may not be able to stand at the point of the minimum value, but hovering around the minimum value point.

![](../images/9710a69ba509a9dcbca351fccc6e7aae.jpg)

### 17.3 Mini-Batch Gradient Descent

The mini-batch gradient descent algorithm is an algorithm between the batch gradient descent algorithm and the stochastic gradient descent algorithm. The parameter ${{\theta }}$ is updated every time the constant $b$ training examples are calculated.

Usually we will make $b$ between 2-100. The advantage of this is that we can use vectorization to loop $b$ training examples. If the linear algebra function library we use is better and can support parallel processing, then the overall performance of the algorithm will not be affected (and random Gradient descent is the same).

### 17.4 Stochastic Gradient Descent Convergence

Now we introduce the debugging of stochastic gradient descent algorithm and the selection of learning rate $α$.

In batch gradient descent, we can make the cost function $J$ a function of the number of iterations, draw a chart, and judge whether the gradient descent is converged according to the chart. However, in the case of large-scale training sets, this is unrealistic because the computational cost is too high.

In stochastic gradient descent, we calculate the cost every time before updating ${{\theta }}$, and then after each iteration of $x$, find the average value of these $x$ times for calculating the training instance, Then plot the function between these averages and the number of iterations of $x$.

![](../images/76fb1df50bdf951f4b880fa66489e367.png)

When we draw such a chart, we may get an image of the function that is bumpy but not significantly reduced (as shown by the blue line in the lower left picture above). We can increase $α$ to make the function smoother, maybe we can see the downward trend (as shown by the red line in the lower left picture above); or maybe the function chart is still bumpy and not falling (as shown by the magenta line) Shown), then our model itself may have some errors.

If the curve we get continues to rise as shown in the lower right above, then we may need to choose a smaller learning rate $α$.

We can also reduce the learning rate as the number of iterations increases, for example:

​ $$\alpha = \frac{const1}{iterationNumber + const2}$$

As we continue to approach the global minimum, by reducing the learning rate, we force the algorithm to converge rather than hover around the minimum.
But usually we don't need to do this to have a very good effect, the calculation of the adjustment of $α$ is usually not worth it

![](../images/f703f371dbb80d22fd5e4aec48aa9fd4.jpg)

### 17.5  Online Learning 

Suppose you have a company that provides transportation services. Users come to ask you about the service for shipping packages from **A** to **B**. At the same time, suppose you have a website that allows users to log in multiple times. , And then they tell you where they want to send the package, and where to send the package, that is, the origin and destination, and then your website provides the service price of the package. For example, I will charge \$50 to ship your package, I will charge \$20 and the like, and then according to the price you open to the user, the user will sometimes accept this shipping service, then this is a positive sample, sometimes they will Go away, and then they refuse to buy your shipping service, so let's assume that we want a learning algorithm to help us and optimize the price we want to offer to users.

An algorithm to model problems when learning from it Online learning algorithms refer to the learning of data streams rather than offline static data sets. Many online websites have a continuous flow of users. For each user, the website hopes to smoothly learn algorithms without storing the data in the database.

If we are operating a logistics company, whenever a user inquires about the courier cost from location A to location B, we give the user a quotation. The user may choose to accept ($y=1$) or not ($y=0 $).

Now, we want to build a model to predict the possibility of users accepting quotations and using our logistics services. So quote
It is one of our characteristics. Other characteristics are distance, starting location, target location, and specific user data. The output of the model is: $p(y=1)$.

The online learning algorithm is somewhat similar to the stochastic gradient descent algorithm. We learn from a single instance instead of looping through a training set defined in advance.

Repeat forever (as long as the website is running) {
  Get $\left(x,y\right)$ corresponding to the current user 
​        $\theta:={\theta}_{j}-\alpha\left( {h}_{\theta}\left({x}\right)-{y} \right){{x}_{j}}$
​       (**for** $j=0:n$) 
    }

Once the learning of a piece of data is completed, we can discard the data without storing it again. The advantage of this method is that our algorithm can be well adapted to the user's tendency, and the algorithm can continuously update the model according to the user's current behavior to suit the user.

Each interaction event does not only generate a data set. For example, we provide users with 3 logistics options at a time, and users choose 2 items. In fact, we can obtain 3 new training examples, so our algorithm can start from 3 instances at a time. To learn and update the model.

Any of these problems can be classified as standard, machine learning problems with a fixed sample set. Perhaps, you can run a website of your own, try to run it for a few days, then save a data set, a fixed data set, and then run a learning algorithm on it. But these are practical problems. In these problems, you will see that large companies will obtain so much data. There is really no need to save a fixed data set. Instead, you can use an online learning algorithm to continuously Learn, learn from the data continuously generated by these users. This is the online learning mechanism, and then as we have seen, the algorithm we use is very similar to the stochastic gradient descent algorithm, the only difference is that we will not use a fixed data set, what we will do is Get a user sample, learn from that sample, then discard that sample and continue, and if you have a continuous stream of data for an application, such an algorithm may be well worth considering. Of course, one of the advantages of online learning is that if you have a changing user base, or if you are trying to predict something, it is slowly changing, just like your user's taste is slowly changing, this online learning algorithm can slowly Debug the hypotheses you have learned and adjust them to the latest user behavior.

### 17.6  Map Reduce and Data Parallelism 

Mapping simplification and data parallelism are very important concepts for large-scale machine learning problems. As mentioned earlier, if we use the batch gradient descent algorithm to solve the optimal solution of a large-scale data set, we need to loop through the entire training set, calculate the partial derivative and the cost, and then sum, the calculation cost is very large. If we can distribute our data set to not many computers and let each computer process a subset of the data set, then we will sum up the calculated results. Such a method is called mapping simplification.

Specifically, if any learning algorithm can be expressed as a summation of the functions of the training set, then this task can be distributed to multiple computers (or different **CPU** cores of the same computer) to achieve acceleration The purpose of processing.

For example, we have 400 training examples, and we can assign the summation task of batch gradient descent to 4 computers for processing:

![](../images/919eabe903ef585ec7d08f2895551a1f.jpg)           

Many advanced linear algebra function libraries have been able to use multiple cores of the multi-core **CPU** to process matrix operations in parallel, which is why the vectorized implementation of the algorithm is so important (faster than calling the loop).

18 Application Example: Photo OCR
------------------------------------------------------------

### 18.1 Problem Description and Pipeline 

What the image text recognition application does is to recognize text from a given picture. This is much more complicated than identifying text from a scanned document.

![](../images/095e4712376c26ff7ffa260125760140.jpg)

In order to complete such work, the following steps need to be taken:

1. **Text detection**-separates the text on the picture from other environmental objects

2. **Character segmentation**-the text is divided into single characters

3. **Character classification**-determine what each character is
   A task flow chart can be used to express this problem. Each task can be solved by a separate team:

![](../images/610fffb413d8d577882d6345c166a9fb.png)

### 18.2 Sliding Windows

Sliding window is a technique used to extract objects from images. If we need to identify pedestrians in a picture, the first thing to do is to use many fixed-size pictures to train a model that can accurately recognize pedestrians. Then we use the size of the picture used to train the pedestrian recognition model to crop on the picture we want to perform pedestrian recognition, and then give the cut slice to the model, let the model judge whether it is a pedestrian, and then slide on the picture to crop The region is re-trimmed, and the newly-cut slices are also handed over to the model for judgment, and so on until all the pictures have been detected.

Once completed, we enlarge the cropped area proportionally, then crop the picture at the new size, and reduce the newly cut slices to the size adopted by the model in proportion to the model for judgment, and so on.

![](../images/1e00d03719e20eeaf1f414f99d7f4109.jpg)

The sliding window technology is also used for text recognition. First, the training model can distinguish characters from non-characters. Then, the sliding window technology is used to recognize characters. Once the character recognition is completed, we will expand the recognized area and then overlap. To merge. Then we use the aspect ratio as a filter condition to filter out areas with a height greater than the width (think that the length of the word is usually greater than the height). The green area in the figure below is the area that is considered to be text after these steps, while the red area is ignored.

![](../images/bc48a4b0c7257591643eb50f2bf46db6.jpg)

The above is the text detection stage.
The next step is to train a model to complete the task of splitting the text into characters. The training set required consists of a picture of a single character and a picture between two connected characters to train the model.

![](../images/0a930f2083bbeb85837f018b74fd0a02.jpg)

![](../images/0bde4f379c8a46c2074336ecce1a955f.jpg)

After the model is trained, we still use sliding window technology for character recognition.

The above is the character segmentation stage.
The final stage is the character classification stage, and a classifier can be trained using neural networks, support vector machines, or logistic regression algorithms.

### 18.3 Getting Lots of Data and Artificial Data 

If our model is of low variance, then obtaining more data for training the model can have a better effect. The problem is, how do we get the data, the data is not always directly available, we may need to create some data manually.

Taking our text recognition application as an example, we can download various fonts from the font website, and then use these different fonts with various random background images to create some examples for training, which allows us to obtain an infinite size Training set. This is an example of creating from scratch.

Another method is to use the existing data and then modify it, such as distorting, rotating, and blurring the existing character pictures. As long as we think that the actual data is likely to be similar to the data after such processing, we can use this method to create a large amount of data.

Several methods for obtaining more data:

   1. Manual data synthesis

   2. Collect and mark data manually

   3. Crowdsourcing