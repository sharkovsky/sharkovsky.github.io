# Neural Networks!

#### Deep Learning
#### Big Data
#### Buzz Buzz
#### Hype Hype

HPC Tech Talk
June 2017

*Francesco Cremonesi*

---

![Image](./assets/md/assets/venn.jpg)

---

## Everything you always wanted to know about neural networks but were afraid to ask

---

### inference
Requires:
- topology
- activation function

### training
Requires:
- cost function
- optimization algorithm

---

## The MLP

#### multi layer perceptron

<img src="./assets/md/assets/example_network.png" width="500" style="background-color:white;"/>
<img src="./assets/md/assets/activ.jpg" width="200"/>

---

## inference

#### Given an input, compute the NN output

1. initialize with input $\mathbf{z}^0 = \mathbf{x}$
2. for each layer:
  1. compute weighted inputs
     $$ \mathbf{z}^l = W \mathbf{a}^{l-1} $$
  2. compute activations
    $$ \mathbf{a}^l = \sigma ( \mathbf{z}^l ) $$

---

## matrix-vector operations!

Clearly, NN involve **a lot** of matrix vector operations. That's what makes them good for GPU.

#### exceptions

- convolutional layers
- pooling layers
- batch normalization layers

---

## Training

#### Teach the NN to recognize patterns in data

---

#### cost functions

- cross-entropy
- least squares

#### optimization methods

- Gradient Descent
- Momentum
- AdaM

---

#### Stochastic Gradient Descent

$$ W_i \leftarrow W_i + \alpha \frac{dC}{dW_i} $$

<img src="./assets/md/assets/backprop.png" width="600" style="background-color:white;"/>

---
## Backprop

or, rebranding the **chain rule**!

<img src="./assets/md/assets/prof.jpg" width="350"/>

My <a target="_blank" href="https://sharkovsky.github.io/2017/06/07/naive-backprop.html">fantastic</a> blog post.

---

### Backprop

Recursive algorithm:

- compute the *errors* for a layer;
- use them to compute the *errors* for the **previous** layer.

#### More of the same

Again, the only operations used:
- $\mathbf{z} = W\mathbf{x}$ : matrix-vector multiplication
- $\mathbf{o} = \mathbf{f}(\mathbf{z})$ : applying an elementwise nonlinear function to a vector

---


## Playground!

<a href="http://playground.tensorflow.org/#activation=sigmoid&regularization=L2&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.3&regularizationRate=0&noise=0&networkShape=6,3&seed=0.78731&showTestData=false&discretize=false&percTrainData=50&x=false&y=false&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false"> Tensorflow Playground </a>


---

### Partial Summary

<img src="./assets/md/assets/heraclitus.jpg" width=100 />

> Neural Networks are nothing more than matrix vector multiplications.

> Neural Networks are much more than matrix vector multiplications.
>
> *Heraclitus, 500BC*

---

# Interesting Architectures

---

## Recurrent Neural Networks

shameless <a target="_blank" href="https://sharkovsky.github.io/2017/06/12/naive-recurrent.html">self-publicity</a>.

---

## Convolutional Neural Networks

<img src="./assets/md/assets/conv.png" width="700" />

---

![Image](./assets/md/assets/conv2.png)

---

![Image](./assets/md/assets/conv-viz.png)

---

## Reservoir Computing

![Image](./assets/md/assets/reservoir.png)

See H. Jeger's MIND Institute <a target="_blank" href="http://minds.jacobs-university.de/sites/default/files/uploads/teaching/MLSpring16/ReservoirComputing.pdf">slides</a> and <a target="_blank" href="http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf">write-up</a>

---

# NN Libraries

![frameworks](./assets/md/assets/algorithmia.png)


---

# NN Libraries

Reviews on:
- the [algorithmia slides](http://blog.algorithmia.com/deploying-deep-learning-cloud-services/);
- this [github](https://github.com/zer0n/deepframeworks/blob/master/README.md) repo;
- this [paper](https://arxiv.org/pdf/1608.07249.pdf).

---

# semi final note

Deep Learning $\neq$ Inference + Backprop

- optimization methods
- regularization methods
- cost function
- hyperparameter search, cross validation
- feature engineering

---

# final note

Deep Learning $\neq$ Machine Learning

- Decision trees, random forests
- Support vector machines
- Unsupervised, reinforcement learning
- Principal Component Analysis
- Indipendent Component Analysis
- Markov decision process
- Expectation maximization
- Gaussian mixture models
- Bayesian methods




---

##  Thank you for your attention

---

# more details

---

## ML 101: Regression

> Regression: predict y = f(x) when you don't know f, but you have a lot of (x,y) measurements.

![Image](./assets/md/assets/regression.jpg)

---

### ML 101: Linear Regression

#### Probabilistic model

$$ y = Wx + N(0, \sigma^2) $$

We define a **likelihood** function (*fonction de vraisemblance*, *funzione di verosimiglianza*, *fun&#231;&#227;o de verossimilhan&#231;a*)

$$ L(W) = p(y | X; W ) $$

L(W) = * the **probability** of observing an outcome $y$, **given** that until now I have observed $X$, as a function of the model $W$ *?

---

### ML 101: Linear Regression

Given $x$ and $y$, how do we find $W$?

#### Maximum likelihood

We want to **maximize** the likelihood.

> $$ W = \text{argmax } L(W) $$

---

### ML 101: Linear Regression

Math-magic proves:

> maximizing $L(W)$ is equivalent to minimizing:
> $$ J(W) = \frac{1}{2} \left ( Y - WX \right )^T\left ( Y - WX \right ). $$

Look familiar? LEAST SQUARES!

> $W = (X^TX)^{-1}X^TY$

---

## Summary

1. From a set of observations $(x,y)$ we want to get a model $y = f(x)$;
2. we imposed a probabilistic and a linearity assumption
$$ y = Wx + N(0, \sigma^2); $$
3. by maximing the likelihood $L(W)$ we naturally obtain the classical least-squares (line through a cloud)
$$ W = (X^TX)^{-1}X^TY .$$

But what if you want a different probabilistic model? (in particular, *categorical*).

---

## Generalized Linear Models (GLM)

Three assumptions:
1. a natural parameter $\eta$, **linearly** tied to the inputs $\eta = Wx$;
2. a very general model for the **likelihood**
$$ p(y; \eta) = b(y)e^{ \eta^Ty - a(\eta) } ;$$
3. **goal** to predict the expected value $ h(x) = E[y | x] $.

---

### GLM

Surprise: the least-squares regression from before falls into this category!

But here's new candy for you:

#### Logistic Regression

Valid if your output is categorical (1 or 0) as, for example, in a classification problem.

---

## Logistic regression

Can be seen as GLM, with the underlying distribution of a *Bernoulli*

$$ p(y; \eta) = e^{ \eta^Ty - log( 1 + e^{\eta} ) } $$

which we can rewrite as:

$$ p(y; \eta) = \left ( \frac{1}{1 + e^{-\eta} } \right )^y \left( 1- \left ( \frac{1}{1 + e^{-\eta} } \right ) \right )^{1-y} $$

Familiar? **Logistic function!**

---

## Summary

- two probabilistic models: Gaussian and Bernoulli;
- two types of regression: least-squares and logistic;


---

#### Cross entropy

for **binary classification**

$$
C = \frac{1}{N}\sum\limits_n \Bigg [ y_n \log \left ( \frac{1}{1 + e^{-z_n}} \right )$$
$$ + \left ( 1 - y_n \right ) \log \left ( 1 - \frac{1}{1 + e^{-z_n}} \right ) \Bigg ] $$

#### Gradient Descent

$$ W_i \leftarrow W_i + \alpha \frac{dC}{dW_i} $$

---

## Demystification

**Rebranding** old techniques!

Sigmoid Activation and Cross-Entropy cost come from **Generalized Linear Models (GLM)**.

<img src="./assets/md/assets/prof.jpg" width="350"/>

Simple neural networks are **nothing more** than plain-old regression techniques.

---

## So why all the hype?

![Image](./assets/md/assets/yoda.jpg)
