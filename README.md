MLP Classifier
==========================
A *Handwritten* **Multilayer Perceptron Classifier**

This python implementation is an extension of artifical neural network discussed in [Python Machine Learning](https://github.com/rasbt/python-machine-learning-book) and  [Neural networks and Deep learning](neuralnetworksanddeeplearning.com) by extending the ANN to **deep** neural network &  including **softmax layers**, along with **log-likelihood** *loss function* and **L1** and **L2** *regularization techniques*.

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
![](https://lh3.googleusercontent.com/ph9owMw5gW6nSi6ajo-_p0Ocy_8eFFVSbkCS4jqU3Ng8gRQDUFPzRd1juP57R7l7YyDwFgjDhLd9SHHz8xZy2vlWN5qpYGnWuQyzcVwPc6HLldNzeY7IUCFmFbhM40WwGCKGvId3GM95CQ20ql0AvWAIjKZ9V62Kvx9Qe5AdL37BG5pEIoGRcDefpi3EWfQ90knzODvPQ7FKfaOLjEcLJ3qRnQjToxXJOZrVuwtVHV5bLkrbZ8VqN-bdFwRe8_hnHEkKUS-oTadpLBbgdXmSjj_RkIDnd0-RjJ8i5nkVdzY8FQT0DozkGzor7o2T_f1yd6TxWjb17-eLLsp8Il9akspOGKkGVe7o9Qh2flT3YU3_3P2gxQATtMZltbXgORGixjjXrQD6s78Bvosx6QN5goTevAFkDJ8mRkb_xMtc9vbPrLjXSLNK-qYJBQ3q7ncRHZDPPilLlJ-a859VsHr4QH2VATZ8aHxnHtBVttx1SQBIG2pxZGNk1_U-HvSH4FaJZNqjDEkwt8fd1FvTmU6yZ6x1q9zL_PmrDrKnJsumqNssrHWjMRyQzpaLjZfB_tMTzixyNxLrkYZ9M12o-iij_YY9PQ=w112-h61-no)  
The output from the softmax layer can be thought of as a probability distribution.  
![](https://lh3.googleusercontent.com/JgWz6nNbk0o1WE18hov7btuwE8-0LT47Bk8Gri6Fg8J1JTFyLGQtyOaABDNJXF_Xfjbss3RwBmOi_OZ68NUUjXszg-6ItyMuay0p7tVRMie9Q-hCudkHbD0nwQ5oxJ7n5eQaTTX55B-X43PTa6tdhMNML6ta6QUy33ItKBbfcTHT476mjg4BBtxtx7GJ-j0uhkEXnH6_pE2j4LppIKHzfW-XTh3qLbZHYDHLvc9-1uF416TH6o-rdzotQslamELbl-l4rr2flodETrQmvb2Nq_dMJV7K8upsPNBZ1AL1q7uk0Bq7A43Sj46RbIsTt0wgiucWU0XdoG5i2YOYVmM-Q0xOMBiYOcA7rqgnH0E07yjzR_Nc71p3R8kieGr42v7GmsrsorxFtmBKeqfixEfvu4ZHfSgRfROZk-KFm1I0i_1pvibHsCOZFjbUdYrCMkW0KDWA_AsYnXop7jJ1AfHbTNff9EhtS4BB0Cu4Khj-bnyM_ztnWnbj10z860Lqj28VBTVcZXLtbKZVdM-dxfM7FfpuU55CSyHsY16oUGrWkSB1JjBmV6MS-nfI-tVtAWkrS2OVkWkfJuRKJT7S3leK-L8sWg=w206-h106-no)  
In many problems it is convenient to be able to interpret the output activation ***O(j)*** as the network's estimate of the probability that the correct output is ***j***.

Refer [these notes](https://www.ics.uci.edu/~pjsadows/notes.pdf) for calculus of softmax function.
[Source]( http://yann.lecun.com/exdb/mnist/) of MNIST training data-set.