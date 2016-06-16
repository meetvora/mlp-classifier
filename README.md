MLP Classifier
==========================
A *Handwritten* **Multilayer Perceptron Classifier**

This python implementation is an extension of artifical neural network discussed in [Python Machine Learning](https://github.com/rasbt/python-machine-learning-book) and  [Neural networks and Deep learning](http://neuralnetworksanddeeplearning.com) by extending the ANN to **deep** neural network &  including **softmax layers**, along with **log-likelihood** *loss function* and **L1** and **L2** *regularization techniques*.

### Some Basics
An artificial neuron is mathematical function conceived as a model of biological neurons. Each of the nodes in the diagram is a a neuron, which transfer their information to the next layer through transfer function.

![artificial-neuron](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/600px-ArtificialNeuronModel_english.png)

The transfer function is a linear combination of the input neurons and a fixed value - *bias* (threshold in figure). The coefficients of the input neurons are *weights*.  
In the code, bias is a numpy array of size(layers-1) as input layer do not have a bias. The weights, also a numpy array, form a matrix for every two layers in the network.  
Activation function is the output of the given neuron.

```python
X: vectorize{(j-1)th layer}

    w = weights[j-1]
    bias = threshold[j-1]
    transfer_function = dot_product(w, X)
    o = activation(transfer_function + bias)
````

### Details
The implementation includes two types of artificial neurons:
  * Sigmoid Neurons  
  * Softmax Neurons

The loss function associated with Softmax function is the *log-likelihood function*, while the loss function for Sigmoid function is the the *cross-entropy function*. The calculus for both loss functions have been discussed within the code. 

Further, the two most common regularization techiques - *L1* and *L2* have been used to prevent overfitting of training data.

##### Why Softmax?
For **z**<sup>L</sup><sub>j</sub> in some vector **Z**<sup>L</sup>, *softmax(**z**<sup>L</sup><sub>j</sub>)* is defined as  
![](https://s31.postimg.org/f44eizfm3/Screenshot_from_2016_06_16_04_54_45.png)  
The output from the softmax layer can be thought of as a probability distribution.  
![](https://s31.postimg.org/4399dynd7/Screenshot_from_2016_06_16_04_50_36.png) 
  
In many problems it is convenient to be able to interpret the output activation ***O(j)*** as the network's estimate of the probability that the correct output is ***j***.

Refer these [notes](https://www.ics.uci.edu/~pjsadows/notes.pdf) for calculus of softmax function.  
[Source](http://yann.lecun.com/exdb/mnist/) of MNIST training data-set.