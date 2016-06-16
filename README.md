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
![](https://lh3.googleusercontent.com/kxeqwfuEajCwQ3WDgMDQZ2VKWqokFovGIleeIWMT3_dZoTcF4fddmgOY5NSjsh81KdZoc4iGWKDPmhe81IqQ8RlD2qn1ZV8wpbCXQpYLBSEEiIZnN_ARbL55Y93Hg3a6BCKD2DaTXcyn4lR_VoDmSjZA8qhMr26A9GWXA7jlhnrlte0cLF-3ATHGAHTwqf08yEKO3gt7GEFmSqst5MRAsqA832mmHALMGgH_ChmdoYTcHKQHL7WZ25ynAtfQChR2C2KnF8HraLZqhMUiZi3a62wkPqu3lAtge4YcBUrWVe-pC-FmqVyZMjqv9n442S_5z7MO5qQxK4t974vHZ81KNRXaYhhEvtv1pV9y0FFFJiGINTisGKifqIHweN11cANol1PyXjZua6VBjbmnKtZT5lszoUAP3HdMTEh_pztrcOJc_Sg4ttIgyX6lChjm0KgAcytSXq0EyBgpvUIHaL3VONJADF9pqCuUKkHJewLEj1SN5i05hzunQQBbiXslCHcKvqdhKYXqPocQe1buLtZtjeR09JuEFMiRXewzhzH-ThDdJRraem-dKj47Q04B16xHewVZ2ET3pU64oj_PxmZPNJJqmg=w112-h61-no)  
The output from the softmax layer can be thought of as a probability distribution.  
![](https://lh3.googleusercontent.com/eOpMveMdbZQV3GWw-Q_S3cBvnMMyAd3s9OBLZcy-i_gH83xto4uvrE290SFDMpc67qpRHW9s9ubrllZO4B0n2URImMbRJIcTvms9mPDbLIazT8Sv-Qv6sXTLuidsEFjG2i3x18eaWcUjeLN3D0bm3xlSjbARuL7fKXpC2vve-f5YB0ItiS6sGm-Ae1Ys4Xvj-dyGZvthdpB-cW1Mo6dpz0Ry6Eu8sUYwv_mXsj7c-S0aBR0gbwhux75PmlBYiJk5veFr1Wknn4ZOyuQghhnHTao7ESmeQ6H2NvkyyVwbXmGc75ln7sMDpGxO6raa6cDKmqjCp0-3NnU3IFcSmRV32EgAW1hBZ5PLcV2owEovhoYdfSWB8Fx405T8V95YSlMTHIisvBXGBNDkvYBLeo5CLL4CI0WHXs-i7f-Wc-MPF8VmCUF6e0QZXRpS4gJSB5v6h-ZCfv0xFqDXzLqDAhd9dHiEEfKWjOGVGuPlt8WsZm5mZqegFzHCDVRCjbHAZS-sVa-7HjdiwE1zVFGzaPK0CG1nJOiAO4-j58Dhs5rSfLuywDK8ZmhLN6oXOCKHgA0XPNTQ0lKnTbrHCFWykpwj9iim8A=w206-h106-no)  
In many problems it is convenient to be able to interpret the output activation ***O(j)*** as the network's estimate of the probability that the correct output is ***j***.

Refer these [notes](https://www.ics.uci.edu/~pjsadows/notes.pdf) for calculus of softmax function.  
[Source](http://yann.lecun.com/exdb/mnist/) of MNIST training data-set.