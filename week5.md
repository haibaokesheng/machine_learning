5th week
=====
[TOC]

9.Neural Networks: Learning
---------------------------------------------

### 9.1 代价函数


First introduce some new marking methods that will be discussed later:
Suppose there are $m$ training samples for the neural network, each containing a set of input $x$ and a set of output signals $y$, $L$ represents the number of neural network layers, and $S_I$ represents the **neuron* of each layer *Number ($S_l$ represents the number of neurons in the output layer), and $S_L$ represents the number of processing units in the last layer.

The classification of the neural network is defined as two cases: two-class classification and multi-class classification,

$K$ category classification: $S_L=k, y_i = 1$ means to be assigned to the $i$ category; $(k>2)$

![](../images/8f7c28297fc9ed297f42942018441850.jpg)


In logistic regression, we have only one output variable, also called **scalar**,and only one dependent variable $y$, but in neural networks, we can have many output variables, our $h_\theta (x)$ is a vector of dimension $K$, and the dependent variable in our training set is also a vector of the same dimension, so our cost function will be more complicated than logistic regression, as: $\newcommand{\subk} [1]{ #1_k }$
$$h_\theta\left(x\right)\in \mathbb{R}^{K}$$ $${\left({h_\theta}\left(x\right)\right)}_{i }={i}^{th} \text{output}$$

### 9.2 Backpropagation Algorithm

Previously, we used a forward propagation method when calculating the prediction results of the neural network. We started from the first layer and proceeded to the calculation layer by layer until the last layer of $h_{\theta}\left(x \right)$.
Now, in order to calculate the partial derivative of the cost function $\frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right)$, we need to use an inverse Propagation algorithm, that is, first calculate the error of the last layer, and then find the error of each layer in the reverse direction until the penultimate layer.

Suppose our training set has only one sample $\left({x}^{(1)},{y}^{(1)}\right)$, our neural network is a four-layer neural network, where $ K=4, S_{L}=4, L=4$:

Forward propagation algorithm:

![](../images/2ea8f5ce4c3df931ee49cf8d987ef25d.png)
We start from the error of the last layer. The error is the error between the prediction of the active unit (${a^{(4)}}$) and the actual value ($y^k$), ($k=1: k$).
We use $\delta$ to express the error, then: $\delta^{(4)}=a^{(4)}-y$
We use this error value to calculate the error of the previous layer: $\delta^{(3)}=\left({\Theta^{(3)}}\right)^{T}\delta^{(4) }\ast g'\left(z^{(3)}\right)$
Where $g'(z^{(3)})$ is the derivative of the $S$ shape function, $g'(z^{(3)})=a^{(3)}\ast(1-a^ {(3)})$. And $(θ^{(3)})^{T}\delta^{(4)}$ is the sum of the errors caused by the weights. The next step is to continue to calculate the error of the second layer:
$ \delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}\ast g'(z^{(2)})$
Because the first layer is the input variable, there is no error. After we have all the error expressions, we can calculate the partial derivative of the cost function, assuming $λ=0$, that is, when we don't do any regularization processing:
$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)} \delta_{i}^{l+1}$

It is important to know the meaning of the subscripts in the above formula clearly:

$l$ represents the currently calculated layer.

$j$ represents the subscript of the active unit in the current calculation layer, and will also be the subscript of the $j$ input variable of the next layer.

$i$ represents the subscript of the error cell in the next layer and is the subscript of the error cell in the next layer affected by the $i$ row in the weight matrix.

If we consider regularization, and our training set is a feature matrix rather than a vector. In the above special case, we need to calculate the error unit of each layer to calculate the partial derivative of the cost function. In a more general case, we also need to calculate the error unit of each layer, but we need to calculate the error unit for the entire training set. At this time, the error unit is also a matrix, we use $\Delta^{(l)} _{ij}$ to represent this error matrix. The error caused by the $i$ activation unit in the $l$ layer is affected by the $j$ parameter.

Our algorithm is expressed as:

![](../images/5514df14ebd508fd597e552fbadcf053.jpg)

That is, first use the forward propagation method to calculate the activation unit of each layer, use the results of the training set and the results predicted by the neural network to find the error of the last layer, and then use the error to calculate back to the second layer using the error All errors.

After finding the $\Delta_{ij}^{(l)}$, we can calculate the partial derivative of the cost function, the calculation method is as follows:

$ D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}$              ${if}\; j \neq  0$

$ D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}$                             ${if}\; j = 0$

### 9.3 Backpropagation Intuition


Forward propagation algorithm:

![](../images/5778e97c411b23487881a87cfca781bb.png)

![](../images/63a0e4aef6d47ba7fa6e07088b61ae68.png)

What the back propagation algorithm does is:

![](../images/57aabbf26290e2082a00c5114ae1c5dc.png)

![](../images/1542307ad9033e39093e7f28d0c7146c.png)

$\delta^{(l)}_{j}="error" \ of cost \ for \ a^{(l)}_{j} \ (unit \ j \ in \ layer \ l) $ understand as follows:

$\delta^{(l)}_{j}$ is equivalent to the "error" of the activation item obtained in cell $j$ of layer $l$, that is, the "correct" $a^{(l)} The difference between _{j}$ and the calculated $a^{(l)}_{j}$.

And $a^{(l)}_{j}=g(z^{(l)})$, (g is the sigmoid function). We can imagine that $\delta^{(l)}_{j}$ is the little bit differential that is taken when the function is differentiated, so it is more accurate to say $\delta^{(l)}_{j}=\frac{\partial}{\partial z^{(l)}_{j}}cost(i)$

而 $a^{(l)}_{j}=g(z^{(l)})$ ，（g为sigmoid函数）。我们可以想象 $\delta^{(l)}_{j}$ 为函数求导时迈出的那一丁点微分，所以更准确的说 $\delta^{(l)}_{j}=\frac{\partial}{\partial z^{(l)}_{j}}cost(i)$

### 9 - 4 - Implementation Note Unrolling Parameters


![](../images/0ad78547859e6f794a7f18389d3d6128.png)

![](../images/f9284204de41bffa4f7bc1dea567044e.png)

![](../images/ebd7e196e272737f497853ba60743c44.png)

### 9.5 Gradient Checking 

When we use the gradient descent algorithm for a more complex model (such as a neural network), there may be some errors that are not easily noticeable, meaning that although the cost seems to be decreasing, the final result may not be optimal solution.

To avoid such problems, we adopt a method called **Numerical Gradient Checking**. The idea of this method is to test whether the derivative value we calculated is really what we requested by estimating the gradient value.

The method used to estimate the gradient is to select two very close points along the tangent direction on the cost function and then calculate the average of the two points to estimate the gradient. That is, for a specific $\theta$, we calculate the generation value at $\theta$-$\varepsilon $ and $\theta$+$\varepsilon $ ($\varepsilon $ is a very small value, Usually choose 0.001), and then average the two costs to estimate the substitute value at $\theta$.



![](../images/5d04c4791eb12a74c843eb5acf601400.png)


When $\theta$ is a vector, we need to test the partial derivative. Because the partial derivative test of the cost function only tests for one parameter change, the following is an example that only tests for $\theta_1$:
$$ \frac{\partial}{\partial\theta_1}=\frac{J\left(\theta_1+\varepsilon_1,\theta_2,\theta_3...\theta_n \right)-J \left( \theta_1-\varepsilon_1,\theta_2,\theta_3...\theta_n \right)}{2\varepsilon} $$

Finally, we also need to check the partial derivative calculated by the back propagation method.

According to the above algorithm, the calculated partial derivatives are stored in the matrix $D_{ij}^{(l)}$. When testing, we want to expand the matrix into a vector, and we also expand the $\theta$ matrix into a vector. We calculate an approximate gradient value for each $\theta$ and store these values in an approximate gradient matrix In the end, the resulting matrix is compared with $D_{ij}^{(l)}$.

![](../images/bf65f3f3098025530a3c442eea562f8c.jpg)

### 9.6 Random Initialization

Any optimization algorithm requires some initial parameters. So far we have initialized all the parameters to 0, such an initial method is feasible for logistic regression, but not feasible for neural networks. If we set all the initial parameters to 0, this would mean that all the activated units in our second layer will have the same value. Similarly, if all of our initial parameters are a non-zero number, the result is the same.

We usually start with random values between positive and negative ε. Suppose we want to randomly initialize a parameter matrix with a size of 10×11. The code is as follows:

`Theta1 = rand(10, 11) * (2*eps) – eps`

### 9.7 Putting It Together 

Summarize the steps when using a neural network:

Network structure: The first thing to do is to choose the network structure, that is, decide how many layers to choose and how many units each layer has.

The number of units in the first layer is the number of features in our training set.

The number of units in the last layer is the number of classes resulting from our training set.

If the number of hidden layers is greater than 1, ensure that the number of units in each hidden layer is the same. Generally, the more hidden layer units, the better.

What we really have to decide is the number of hidden layers and the number of cells in each middle layer.

Train the neural network:

1. Random initialization of parameters

2. Calculate all $h_{\theta}(x)$ using forward propagation method

3. Write code to calculate the cost function $J$

4. Calculate all partial derivatives using the back propagation method

5. Use numerical test methods to test these partial derivatives

6. Use optimization algorithms to minimize the cost function