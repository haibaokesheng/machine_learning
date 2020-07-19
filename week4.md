Fourth week
=====
[TOC]

8 Neural Networks: Representation
-----------------------------------------------------

### 8.1 Non-linear Hypotheses

Both linear regression and logistic regression have such a disadvantage that when there are too many features, the calculation load will be very large.

Below is an example:

![](../images/5316b24cd40908fb5cb1db5a055e4de5.png)

When we use the multiple terms of $x_1$, $x_2$ to make predictions, we can apply well.

We have seen before that using non-linear polynomial terms can help us build a better classification model. Suppose we have a lot of features, such as greater than 100 variables, we want to use these 100 features to build a nonlinear polynomial model, the result will be a very surprising number of feature combinations, even if we only use a combination of two features $ (x_1x_2+x_1x_3+x_1x_4+...+x_2x_3+x_2x_4+...+x_{99}x_{100})$, we will also have close to 5000 combined features. This requires too many features for general logistic regression.

Suppose we want to train a model to recognize visual objects (such as whether a car is on a picture), how can we do this? One method is that we use many car pictures and many non-car pictures, and then use the value (saturation or brightness) of each pixel on these pictures as a feature.

![](../images/3ac5e06e852ad3deef4cba782ebe425b.jpg)

If we use small pictures of 50x50 pixels, and we treat all pixels as features, there will be 2500 features. If we want to further combine two or two features to form a polynomial model, there will be about ${{ 2500}^{2}}/2$ features (close to 3 million) features. Ordinary logistic regression models cannot effectively handle so many features. At this time, we need neural networks.

### 8.2 Neurons and the Brain

We can learn mathematics, learn to do calculus, and the brain can handle all kinds of amazing things. It seems that if you want to imitate it, you have to write a lot of different software to simulate all these wonderful things. But can it be assumed that the brain does all these, different things, without the need to use thousands of different programs to achieve it. On the contrary, the brain processing method only needs a single learning algorithm? Although this is only an assumption, let me share with you some evidence in this regard.


![](../images/7912ea75bc7982998870721cb1177226.jpg)

This small red area of this part of the brain is your auditory cortex, and you are now understanding what I am saying, which depends on the ear. The ear receives the sound signal, and transmits the sound signal to your auditory cortex, which is why you can understand my words.
Scientists of the nervous system performed the following interesting experiment to cut off the nerves from the ear to the auditory cortex. In this case, reconnect it to the brain of an animal, so that the signal from the eye to the optic nerve will eventually reach the auditory cortex. If so. Then the results show that the auditory cortex will learn to "see". "Look" here represents every layer of meaning we know. So, if you do this to animals, then animals can complete the visual recognition task, they can look at the images and make appropriate decisions based on the images. They are done through this part of the brain tissue. Here is another example. This red brain tissue is your somatosensory cortex. This is what you use to deal with touch. If you do a reconnection experiment similar to the one just now, the somatosensory cortex can also learn to "see ". This experiment and some other similar experiments are called neural reconnection experiments. In this sense, if the human body has the same brain tissue that can process light, sound or tactile signals, then there may be a learning algorithm that can process at the same time. Vision, hearing and touch, rather than the need to run thousands of different programs or thousands of different algorithms to do the thousands of beautiful things these brains accomplish. Perhaps all we need to do is find some approximate or actual brain learning algorithms, and then implement it to learn how to process these different types of data through self-learning. To a large extent, it can be guessed that if we connect almost any kind of sensor to almost any part of the brain, the brain will learn to process it.


![](../images/2b74c1eeff95db47f5ebd8aef1290f09.jpg)

This picture is an example of learning to "see" with the tongue. Its principle is: this is actually a system called **BrainPort**, which is now **FDA**
(US Food and Drug Administration) In the clinical trial stage, it can help blind people see things. The principle is that if you bring a gray-scale camera on your forehead and face forward, it can acquire a low-resolution gray-scale image of things in front of you. You connect a wire to the electrode array installed on the tongue, then each pixel is mapped to a certain position on your tongue. A point with a high voltage value may correspond to a point with a low dark pixel voltage value. Corresponding to bright pixels, even relying on its current function, using this system will allow you and me to learn to "see" things with our tongue in tens of minutes.

![](../images/95c020b2227ca4b9a9bcbd40099d1766.png)

This is the second example, regarding human body echo localization or human body sonar. You can do it in two ways: you can snap your fingers or suck your tongue. But now there are blind people who do receive such training in schools and learn to interpret the sound wave pattern that bounces back from the environment-this is sonar. If you search for **YouTube**, you will find some videos about an amazing kid who was removed due to cancer eyeballs. Although he lost his eyeballs, he can walk around without hitting his fingers by snapping his fingers Anything, he can skate, he can throw basketball into the basket. Note that this is a child without eyes.

![](../images/697ae58b1370e81749f9feb333bdf842.png)

The third example is a tactile belt. If you wear it on your waist, the buzzer will sound and it will always buzz when facing north. It can make people have a sense of direction, in a way similar to the way birds sense direction.


### 8.3 Model Representation I 

In order to build a neural network model, we first need to think about what is the neural network in the brain? Each neuron can be considered as a (**processing unit**/**Nucleus**), which contains many inputs/dendrites (**input**/**Dendrite**) , And there is an (**output**/**Axon**). A neural network is a network in which a large number of neurons are connected to each other and communicate through electrical impulses.

![](../images/3d93e8c1cd681c2b3599f05739e3f3cc.jpg)

The following is a schematic diagram of a group of neurons that use weak electrical current to communicate. These weak currents are also called action potentials, which are actually weak currents. So if a neuron wants to deliver a message, it will send a weak current to other neurons through its axon. This is the axon.

Here is a nerve connected to the input nerve, or another neuron dendrite, then this neuron receives this message and does some calculations, it may in turn pass its own message on the axon to Other neurons. This is the model of all human thinking: our neurons calculate the messages they receive and send them to other neurons. This is also how our feelings and muscles work. If you want to move a muscle, it will trigger a neuron to send a pulse to your muscle and cause your muscle to contract. If some senses: for example, the eye wants to send a message to the brain, then it sends electrical pulses to the brain like this.

![](../images/7dabd366525c7c3124e844abce8c2dd6.png)

The neural network model is built on many neurons, and each neuron is a learning model. These neurons (also called activation units) take some features as output and provide an output according to their own model. The following figure is an example of a neuron using a logistic regression model as its own learning model. In a neural network, parameters can be used as weights (**weight**).

![](../images/c2233cd74605a9f8fe69fd59547d3853.jpg)

We have designed a neural network similar to neurons, the effect is as follows:

![](../images/fbb4ffb48b64468c384647d45f7b86b5.png)

Among them, $x_1$, $x_2$, $x_3$ are input units (**input units**), and we input the original data to them.
$a_1$, $a_2$, $a_3$ are intermediate units, they are responsible for processing the data, and then submit to the next layer.
Finally, there is the output unit, which is responsible for calculating ${h_\theta}\left( x \right)$.

The neural network model is a network where many logical units are organized according to different levels, and the output variables of each layer are the input variables of the next layer. The following picture is a 3-layer neural network. The first layer becomes the **input layer**, the last layer is called the **output layer**, and the middle layer becomes the **hidden layer** . We add a bias **unit** for each layer:

![](../images/8293711e1d23414d0a03f6878f5a2d91.jpg)
Here are some notations to help describe the model:
$a_{i}^{\left( j \right)}$ represents the $i$ activation unit in the $j$ layer. ${{\theta }^{\left( j \right)}}$ represents the matrix of weights when mapping from layer $j$ to layer $j+1$,For example, ${{\theta }^{\left( 1 \right)}}$ represents a matrix of weights mapped from the first layer to the second layer. Its size is: a matrix with the number of activated cells in the $j+1$ layer as the number of rows, and the number of activated cells in the $j$ layer plus one as the number of columns. For example: The size of ${{\theta }^{\left( 1 \right)}}$ in the neural network shown above is 3*4.

For the model shown above, the activation unit and output are expressed as:

$a_{1}^{(2)}=g(\Theta _{10}^{(1)}{{x}_{0}}+\Theta _{11}^{(1)}{{x}_{1}}+\Theta _{12}^{(1)}{{x}_{2}}+\Theta _{13}^{(1)}{{x}_{3}})$
$a_{2}^{(2)}=g(\Theta _{20}^{(1)}{{x}_{0}}+\Theta _{21}^{(1)}{{x}_{1}}+\Theta _{22}^{(1)}{{x}_{2}}+\Theta _{23}^{(1)}{{x}_{3}})$
$a_{3}^{(2)}=g(\Theta _{30}^{(1)}{{x}_{0}}+\Theta _{31}^{(1)}{{x}_{1}}+\Theta _{32}^{(1)}{{x}_{2}}+\Theta _{33}^{(1)}{{x}_{3}})$
${{h}_{\Theta }}(x)=g(\Theta _{10}^{(2)}a_{0}^{(2)}+\Theta _{11}^{(2)}a_{1}^{(2)}+\Theta _{12}^{(2)}a_{2}^{(2)}+\Theta _{13}^{(2)}a_{3}^{(2)})$

In the discussion above, only one row (a training example) in the feature matrix was fed to the neural network. We need to feed the entire training set to our neural network algorithm to learn the model.
We can know that each $a$ is determined by all the $x$ and each $x$ in the previous layer.

（We call this algorithm from left to right the **forward propagation algorithm**

Let $x$, $\theta$, $a$ be represented by a matrix:

![](../images/20171101224053.png)

We can get $\theta \cdot X=a$.

### 8.4 Model Representation II


 **FORWARD PROPAGATION** 
Relative to the use of loops to encode, the use of vectorization will make the calculation easier. Taking the above neural network as an example, try to calculate the value of the second layer:

![](../images/303ce7ad54d957fca9dbb6a992155111.png)

![](../images/2e17f58ce9a79525089a1c2e0b4c0ccc.png)
We make ${{z}^{\left( 2 \right)}}={{\theta }^{\left( 1 \right)}}x$, then ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$, add $a_{0}^{\left( 2 \right)}=1$ after calculation. The calculated output value is:


![](../images/43f1cb8a2a7e9a18f928720adc1fac22.png)

We make ${{z}^{\left( 3 \right)}}={{\theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}} $, then $h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$.
This is just a calculation for a training example in the training set. If we want to calculate the entire training set, we need to transpose the training set feature matrix so that the features of the same instance are in the same column. which is:
${{z}^{\left( 2 \right)}}={{\Theta }^{\left( 1 \right)}}\times {{X}^{T}} $

 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$

In order to better understand the working principle of **Neuron Networks**, we first cover the left half:

![](../images/6167ad04e696c400cb9e1b7dc1e58d8a.png)

The right part is actually $a_0, a_1, a_2, a_3$, and the $h_\theta(x)$ is output according to **Logistic Regression**:

![](../images/10342b472803c339a9e3bc339188c5b8.png)

In fact, the neural network is like **logistic regression**, but we changed the input vector $\left[ x_1\sim {x_3} \right]$ in **logistic regression** into the middle layer $\left [a_1^{(2)}\sim a_3^{(2)} \right]$, that is: $h_\theta(x)=g\left( \Theta_0^{\left( 2 \right)}a_0^ {\left( 2 \right)}+\Theta_1^{\left( 2 \right)}a_1^{\left( 2 \right)}+\Theta_{2}^{\left( 2 \right)}a_ {2}^{\left( 2 \right)}+\Theta_{3}^{\left( 2 \right)}a_{3}^{\left( 2 \right)} \right)$
We can think of $a_0, a_1, a_2, a_3$ as more advanced eigenvalues, that is, evolutions of $x_0, x_1, x_2, x_3$, and they are determined by $x$ and $\theta$ Because of the gradient descent, $a$ changes and becomes more and more powerful, so these more advanced feature values are far more powerful than just raising the power of $x$, and can better predict new data.
This is the advantage of neural networks over logistic regression and linear regression.


### 8.5 Examples and Intuitions I

Essentially, a neural network can learn its own series of characteristics through learning. In ordinary logistic regression, we are limited to using the original features $x_1,x_2,...,{{x}_{n}}$ in the data, although we can use some binomial terms to combine these features , But we are still limited by these original features. In neural networks, the original features are only the input layer. In the three-layer neural network example above, the third layer, the output layer, uses the features of the second layer instead of the original features in the input layer. , We can think of the features in the second layer as a series of new features used by the neural network to predict the output variables.

In neural networks, the calculation of single-layer neurons (no intermediate layers) can be used to represent logical operations, such as logical **AND**, logical **OR**.

Example: logical **AND**; the left half of the figure below is the design of the neural network and the expression of the **output** layer, the upper right part is the sigmod** function, and the lower half is true Value table.

We can use such a neural network to represent the **AND** function:

![](../images/809187c1815e1ec67184699076de51f2.png)

Where $\theta_0 = -30, \theta_1 = 20, \theta_2 = 20$
Our output function $h_\theta(x)$ is: $h_\Theta(x)=g\left( -30+20x_1+20x_2 \right)$

We know that the image of $g(x)$ is:

![](../images/6d652f125654d077480aadc578ae0164.png)



![](../images/f75115da9090701516aa1ff0295436dd.png)

So we have: $h_\Theta(x) \approx \text{x}_1 \text{AND} \, \text{x}_2$

So our: $h_\Theta(x) $

This is the **AND** function.

Next, introduce an **OR** function:

![](../images/aa27671f7a3a16545a28f356a2fb98c0.png)

**OR** is the same as **AND**, the only difference is the value.

### 8.6 Examples and Intuitions II 
**Binary logical operators** when the input feature is a Boolean value (0 or 1), we can use a single activation layer can be used as a binary logical operator, in order to represent different operators, We just need to choose different weights.

The neuron in the picture below (the three weights are -30, 20, 20) can be regarded as the same as the logical **AND**:

![](../images/57480b04956f1dc54ecfc64d68a6b357.png)

![](../images/7527e61b1612dcf84dadbcf7a26a22fb.png)

The neuron in the picture below (the two weights are 10 and -20 respectively) can be regarded as equivalent to logical **NOT**:

![](../images/1fd3017dfa554642a5e1805d6d2b1fa6.png)

We can use neurons to combine into more complex neural networks to achieve more complex operations. For example, we want to implement the **XNOR** function (the two values entered must be the same, both 1 or 0), that is, $\text{XNOR}=( \text{x}_1\, \text{AND} \, \text{x}_2 )\, \text{OR} \left( \left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT }\, \text{x}_2 \right) \right)​$
First construct a expression that can express $\left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right)​​ $Part neuron:

![](../images/4c44e69a12b48efdff2fe92a0a698768.png)

Then the neuron representing **AND** and representing $\left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text {x}_2 \right)$ neurons are combined with neurons representing OR:

![](../images/432c906875baca78031bd337fe0c8682.png)

We have obtained a neural network that can realize the function of the $\text{XNOR}$ operator.

In this way we can gradually construct more and more complex functions, and we can also get more powerful eigenvalues.

This is the power of neural networks.

### 8.7 Multiclass Classification


When we have more than two classifications (that is, $y=1,2,3...$), such as the following situation, what should we do? If we were to train a neural network algorithm to recognize passers-by, cars, motorcycles and trucks, we should have 4 values in the output layer. For example, the first value is 1 or 0 to predict whether it is a pedestrian, and the second value is used to determine whether it is a car.
The input vector $x$ has three dimensions, two intermediate layers, and 4 neurons in the output layer are used to represent 4 categories, that is, each data will appear in the output layer ${{\left[ a\text{ }b\text{ }c\text{ }d \right]}^{T}}$, and only one of $a,b,c,d$ is 1, indicating the current class. The following is an example of a possible structure of the neural network:

![](../images/f3236b14640fa053e62c73177b3474ed.jpg)

![](../images/685180bf1774f7edd2b0856a8aae3498.png)

The output of the neural network algorithm is one of four possible scenarios:

![](../images/5e1a39d165f272b7f145c68ef78a3e13.png)
