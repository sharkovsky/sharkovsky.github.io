---
layout: post
title:  "Boltzmann Machines"
date:   2017-06-23 13:53:10 +0200
---

In this post belonging to a series dedicated to machine learning methods that are in some way _closer_ to biological networks than your run-of-the-mill fully connected network I will present Boltzmann Machines (BMs).
These neural networks are characterized by some properties that make them seem _biologically plausible_:
- arbitrary recurrent connectivity;
- inference is equivalent to energy minimization;
- binary (and possibly stochastic) neuron dynamics;
- unsupervised learning.

For more details on why BMs can be considered biologically plausible, see my [other post]({% post_url 2017-06-20-bio-backprop %}).
Obviously, we are from from understanding how the brain works, and these are all mathematical abstractions only loosely related to the physical/biological phenomenon.

# The neuron dynamics

BM neurons can take only two values: 1 and 0.
The state of the $$i^{th}$$ neuron, denoted $$s_i$$, is periodically updated according to the following rules:

$$
\begin{align*}
z_i &= \sum\limits_j w_{ij} s_j + b_i, \\
p( s_i = 1 ) =& \frac{1}{1 + e^{-z_i}},
\end{align*}
$$

where $$w_{ij}$$ are connection weights in the network and $$b_i$$ are biases.

#### The network dynamics algorithm

In order to let the network evolve over time, the following algorithm is implemented.
This algorithm is based on [Gibbs sampling](http://www.mit.edu/~ilkery/papers/GibbsSampling.pdf), a sampling algorithm based on the idea that, to draw samples from a N-dimensional multivariate distribution, one can draw N samples from the uni-variate distributions obtained by fixing all the other dimensions.
