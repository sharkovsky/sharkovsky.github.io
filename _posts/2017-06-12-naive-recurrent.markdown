---
layout: post
title:  "A naive explanation of backprop through time for recurrent neural networks"
date:   2017-06-12 13:53:10 +0200
---

In the second blog post of this series, I would like to repeat the same detailed computations for backpropagating the error in the case of *recurrent neural networks*.

As I mentioned before there are *many* other explanations of backprop out there, much better than mine.
In addition to [M. Nielsen's book](http://neuralnetworksanddeeplearning.com/chap2.html) and the [deep learning book](http://www.deeplearningbook.org/), I'd like to mention a few other blogs that helped me understand things specifically in the case of recurrent networks.
A. Karpathy's [unreasonable effectiveness of recurrent neural networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is one of the most exciting resources on RNNs out there.
Chen's [arXiv paper](https://arxiv.org/pdf/1610.02583) is a nice, academic style, introduction to the topic, Lipton's [review paper](https://arxiv.org/pdf/1506.00019.pdf) gives a nice overview of the challenges related to training recurrent networks, while H. Jaeger's [tutorial](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf) is a *mathematically intense but complete* presentation which then moves on to talk about reservoir computing.
Finally, I took the idea for the training set from P. Roelants' [blog post](http://peterroelants.github.io/posts/rnn_implementation_part01/).

## What is a recurrent neural network?

Recurrent Neural Networks (RNN) are special networks in which the concept of time is explicitly modeled.
Originally, they were developed to deal with input data in the form of sequences, where each *timestep* corresponds to the processing of one element of the input sequence.
The main difference with feedforward networks is that:

> in RNNs, each neuron is characterized by a state (a variable) whose value can change according to some rules, but whose presence is persistent through time.

In practice this means that we need an uglier notation to represent neurons, as shown in this table:

| Network Type | Notation | Description |
| ------------ | :------: | ----------- |
| feedforward  | $$ h^l_i $$ | hidden state of the $$ i^{th} $$ neuron in the $$ l^{th} $$ layer |
| feedforward  | $$ \mathbf{h}^l $$ | hidden state of the vector of neurons in the $$ l^{th} $$ layer |
| recurrent    | $$ h^l_{n,i} $$ | hidden state of the $$ i^{th} $$ neuron in the $$ l^{th} $$ layer *at timestep* $$n $$ |
| recurrent    | $$ \mathbf{h}^l_n $$ | hidden state of the vector of neurons in the $$ l^{th} $$ layer *at timestep* $$n $$ |


Typically we represent RNNs like this:

![rnn-net]({{ site.url }}/assets/recurrent-net.svg)

If you compare this to the representation of a [feedforward network](../07/naive-backprop.html) you'll notice the *recurrent* arrows connecting a neuron to itself.
These arrows can be misleading because one may be tempted to think that a neuron's state is immediately affected by the recurrent connections.
Instead, what happens is that

> recurrent connections affect the state of a neuron through some delay (typically, a single timestep).

#### The layer-wise point of view

As we saw in the post on [feedforward networks](../07/naive-backprop.html), it is often useful to take the layer-wise point of view by considering the full vector of neurons in each layer.
First, consider the visualization of a feedforward network which I have stretched out using my mad graphic skillz in order to give the 3D impression.
Here, I am going to use a nomenclature that no-one else uses, but I promise it's only for a minor thing: I am going to refer to the different layers of a network as the *space dimension* of the network.
The reason behind this is simply to contrast it to the *time dimension* of recurrent neural networks (a nomenclature that everybody likes).

![feedforward-3D]({{ site.url }}/assets/feedforwards-layers.svg)

In the image, each ball is a neuron and each rectangle is a layer.
I have drawn all the connections, but only higlighted those originating from a specific neuron.
Keep an eye on that neuron for later.

Now compare this to the image of a recurrent neural network, which has both a space and a time dimension.

![recurrent-3D]({{ site.url }}/assets/recurrent-layers.svg)

In the image, the same neural network from before is now *repeated* through time, in the sense that the state of each neuron is is computed ad each timestep.
The highlighted neuron from the previous image now not only sends connections to the next layer, but also *sends connections within the same layer, to the next timestep*.
This image would have been a mess had I drawn every single recurrent connection, so as an example I only drew all the recurrent connections of layer $$L-1$$ from timestep $$N$$ to $$N+1$$.
With your third eye, try to imagine that every recurrent layer sends connections both to its downstream layer in space, and to itself at the next timestep.

In terms of notation, we need to distinguish between the weight matrix of recurrent connections, and the weight matrix of feedforward connections.
I am going to use a notation that is nonstandard, so let me apologize now.
Sorry.
I am going to denote the weight matrix of recurrent connections with an $$T$$ left-subscript.
The $$T$$ stands for the time dimension.
Similary, the feedforward weight matrix will be denoted with an $$S$$ left-subscript, where $$S$$ stands for (you guessed it) space.
See how I niftly combined all the nonstandard notations in one mess?
Here is a summary for you:

| Weights | Notation | Alternative Notations |
| ------------ | :------: | ----------- |
| feedforward from layer $$ l-1 $$ to layer $$ l $$| $$ {}_SW^l $$ | $$ W^l, W^l_{xh} $$ |
| recurrent layer $$ l $$| $$ {}_TW^l $$ | $$ W^l_hh $$ |


Obviously, it is possible to mix non-recurrent and recurrent layers, and in this case only the recurrent layers would behave as I have been describing so far.
Also, it is possible to prescribe recurrent layers that do not connect to the future selves, but connect to future other layers.

#### Formulas are not just for babies

If you think a formula can be more expressive than a thousand words, here is the algorithm to perform **inference** on a recurrent layer $$l$$

$$
    \begin{align*}
    &\text{for every timestep } n\\
    &\mathbf{z}^l_n = {}_SW^l\mathbf{h}^{l-1}_n + {}_TW^l\mathbf{h}^l_{n-1} \\
    &\mathbf{h}^l_n = \sigma ( \mathbf{z}^l_n )
    \end{align*}
$$

## BPTT: Backpropagation Through Time

How do we apply the principles of backpropagation in a recurrent neural network?
The principle is the same as for a feedforward network, and indeed many sources claim that *training recurrent networks is just like training feedforward networks with unrolling.*
I found this statement quite obscure, and preferred writing out the formulas explicitly.

#### Simple example

As we did for feedforward nets, let's start by considering the very simple example of a scalar network.

![rnn-net]({{ site.url }}/assets/recurrent-net.svg)

In this uber-simple network, the output $$ y $$ is simply the hidden state of the neuron in the last layer.

Everyone has to start somewhere, so suppose we initialize the hidden states with some arbitrary initial values $$ h^1_0, h^2_0 $$.
At the first timestep, the networks receives the first element of the sequence $$ x_1 $$, and processes it by:

$$
    \begin{align*}
    h^1_1 &= \sigma \left ( {}_Sw^1x_1 + {}_Tw^1h^1_0 \right ) \\
    h^2_1 &= \sigma \left ( {}_Sw^2h^1_1 + {}_Tw^2h^2_0 \right ) \\
    y_1 &= h^2_1.
    \end{align*}
$$

At the next timestep, we have:

$$
    \begin{align*}
    h^1_2 &= \sigma \left ( {}_Sw^1x_2 + {}_Tw^1h^1_1 \right ) \\
    h^2_2 &= \sigma \left ( {}_Sw^2h^1_2 + {}_Tw^2h^2_1 \right ) \\
    y_2 &= h^2_2,
    \end{align*}
$$

which by the magic of substitution is nothing other than

$$
    \begin{align*}
    h^1_2 &= \sigma \left ( {}_Sw^1x_2 + {}_Tw^1  \sigma \left ( {}_Sw^1x_1 + {}_Tw^1h^1_0  \right ) \right ) \\
    h^2_2 &= \sigma \left ( {}_Sw^2\sigma h^1_2 + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_1 + {}_Tw^2h^2_0 \right ) \right ) \\
    y_2 &= h^2_2,
    \end{align*}
$$

where I have substituted the expressions of $$ h^1_1, h^2_1 $$.
Note that I could have also substituted $$ h^1_2 $$, but I didn't because the process for backpropagating the error through the layers is not in the scope of this post.

If we kept going for $$ N $$ timesteps, the formulas would look like this

$$
    \begin{align*}
    h^1_N &= \sigma \left ( {}_Sw^1x_N + {}_Tw^1  \sigma \left ( {}_Sw^1x_{N-1} + {}_Tw^1 \sigma \left ( {}_Sw^1x_{N-2} + \dots {}_Tw^1 h^1_0  \right ) \right ) \right )\\
    h^2_N &= \sigma \left ( {}_Sw^2h^1_N + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right ) \right ) \right )\\
    y_N &= h^2_N,
    \end{align*}
$$

If you feel like this is too familiar for comfort, don't worry you don't suffer from [Fregoli delusion](https://en.wikipedia.org/wiki/Fregoli_delusion).
The same exact recursive substitutions happen if we move thorugh layers and not through time.
As a matter of fact, we almost already know how to backprop through time thanks to our knowledge of backprop through layers!

#### Gradients galore

Let's compute some derivatives.
Suppose we have a cost function $$ C = C (y) $$ that we want to minimize.
Question: _how does the recurrent weight of the last layer influence the cost function at time $$ N $$?_

$$
    \begin{align*}
    \frac{dC(y_N)}{d{}_Tw^2} &= \frac{dC}{dy} \Big |_{y_N} \frac{ dy_N}{d{}_Tw^2} \\
    &= \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \frac{d z^2_N }{d{}_Tw^2},
    \end{align*}
$$

to compute the last term $$ \frac{d z^2_N }{d{}_Tw^2} $$ we have to calculate the derivative of the long expression

$$  \frac{d \left [ {}_Sw^2h^1_N + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} +     \dots {}_Tw^2 h^2_0  \right )\right )\right ]} {d{}_Tw^2}. $$

Fortunately, $$ h^1_N $$ does *not* depend on $$ {}_Tw^2 $$, nor does $$ h^1_{N-1} $$ and so on.
Therefore we can already simplify

$$
    \begin{align*}
    &\frac{d \left [ {}_Sw^2h^1_N + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} +     \dots {}_Tw^2 h^2_0  \right )\right )\right ]} {d{}_Tw^2} =\\
    &\frac{d \left [ {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right )\right )\right ]} {d{}_Tw^2}.
    \end{align*}
 $$

The annoying bit here is that we are facing the derivative of a product: both $$  {}_Tw^2 $$ and $$ \sigma ( \dots ) $$ depend on $$ {}_Tw^2 $$!
So we must write

$$
    \begin{align*}
    &\frac{d \left [ {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right )\right )\right ]} {d{}_Tw^2} = \\
    & \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0 \right )\right ) \\
    & + {}_Tw^2 \frac{d \left [  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0 \right )\right )\right ]}{d{}_Tw^2}.
    \end{align*}
 $$

The expression that remains under the derivative sign here is the same as before, only one timestep *in the past*.
That's good news.
We can recursively simplify the expression for the derivative into:

$$
    \begin{align*}
    \frac{dC(y_N)}{d{}_Tw^2} &= \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \\
            & \Bigg ( \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0 \right )\right ) \\
            & + {}_Tw^2 \frac{d\sigma}{dz} \Big |_{z^2_{N-1}} \Big ( \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0 \right ) + \dots + \frac{d\sigma}{dz} \Big |_{z^2_{1}} h^2_0 \Big ) \Bigg ).
    \end{align*}
$$

Writing the full expression for what's inside the calls to the $$ \sigma $$ function was useful to understand exactly who depends on what in order to compute the derivatives.
To make the notation lighter, but also to gain insight on backprop, let's substitute back the directly the values of the hidden states.
We obtain the **recursive formulation for the derivative**:

$$
    \frac{dC(y_N)}{d{}_Tw^2} = \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \left ( h^2_N + {}_Tw^2 \frac{d\sigma}{dz} \Big |_{z^2_{N-1}} \left ( h^2_{N-1} + \dots +  \frac{d\sigma}{dz} \Big |_{z^2_{1}} h^2_0 \right ) \right ).
$$

The question now is, how does this extend to the upstream layers?
We start off pretty easily:

$$
    \begin{align*}
    \frac{dC(y_N)}{d{}_Tw^1} &= \frac{dC}{dy} \Big |_{y_N} \frac{ dy_N}{d{}_Tw^1} \\
    &= \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \frac{d z^2_N }{d{}_Tw^1},
    \end{align*}
$$

and we have to now compute the annoying derivative

$$ \frac{d z^2_N }{d{}_Tw^1} = \frac{ d \left [ {}_Sw^2h^1_N + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right ) \right ) \right ]}{d{}_Tw^1}. $$

The caveat is that, in contrast to what happened above, many terms in this expression depend on $$ {}_Tw^1 $$, namely $$ h^1_N, h^1_{N-1}, h^1_{N-2}, \dots $$. We continue with our computations, trying not to make transcription mistakes:


$$
    \begin{align*}
    &\frac{ d \left [ {}_Sw^2h^1_N + {}_Tw^2  \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right ) \right ) \right ]}{d{}_Tw^1} \\
    &= {}_Sw^2 \frac{d h^1_N }{d{}_Tw^1} + {}_Tw^2 \frac{ d \left [ \sigma \left ( {}_Sw^2h^1_{N-1} + {}_Tw^2 \sigma \left ( {}_Sw^2h^1_{N-2} + \dots {}_Tw^2 h^2_0  \right ) \right ) \right ]}{d{}_Tw^1}
    \end{align*}
$$

As before, we have a nice recursive property: the expression that remains under the second derivative sign is simply the same as the one we wanted to compute, but _one timestep in the past_.
Before we move on to the fully recursive formulation, we need an expression for $$ \frac{d h^1_N }{d{}_Tw^1} $$.
As before, $$ {}_Sw^1x_N $$ doesn't depend on $$ {}_Tw^1 $$, and for the second term we have to compute the derivative of a product:

$$
    \begin{align*}
    \frac{d h^1_N }{d{}_Tw^1} &= \frac{d \left [ \sigma \left ( {}_Sw^1x_N + {}_Tw^1  \sigma \left ( {}_Sw^1x_{N-1} + {}_Tw^1 \sigma \left ( {}_Sw^1x_{N-2} + \dots {}_Tw^1 h^1_0  \right ) \right ) \right ) \right ]}{d{}_Tw^1} \\
    &= \frac{d\sigma}{dz} \Big |_{z^1_N} \left ( \sigma \left ( {}_Sw^1x_{N-1} + {}_Tw^1 \sigma \left ( {}_Sw^1x_{N-2} + \dots {}_Tw^1 h^1_0  \right ) \right ) + {}_Tw^1 \frac{d \left [ \sigma \left ( {}_Sw^1x_{N-1} + {}_Tw^1 \sigma \left ( {}_Sw^1x_{N-2} + \dots {}_Tw^1 h^1_0  \right ) \right ) \right ]}{d{}_Tw^1} \right ) .
    \end{align*}
$$

And getting the **recursive formulation of the derivative**:

$$
    \begin{align*}
    \frac{dC(y_N)}{d{}_Tw^1} &= \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \left \{ {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} \left ( h^1_{N-1} + {}_Tw^1 \frac{d h^1_{N-1} }{d{}_Tw^1} \right ) + {}_Tw^2 \frac{ d h^2_{N-1} }{d{}_Tw^1} \right \} \\
    &=  \frac{dC}{dy} \Big |_{y_N} \frac{d\sigma}{dz} \Big |_{z^2_N} \Bigg \{ {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} \left ( h^1_{N-1} + {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} \left ( h^1_{N-2} + {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-2}} \left ( h^1_{N-3} + \dots \right ) \right ) \right ) \\
    & + {}_Tw^2  \frac{d\sigma}{dz} \Big |_{z^2_{N-1}} \Bigg [ {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} \left ( h^1_{N-2} + {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-2}} \left ( h^1_{N-3} + {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-3}} \left ( h^1_{N-4} + \dots \right ) \right ) \right ) \\
    & + {}_Tw^2  \frac{d\sigma}{dz} \Big |_{z^2_{N-2}} \left (  \dots + {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{1}} h^1_0 \right ) \dots \Bigg] \Bigg \}.
    \end{align*}
$$

That's a pretty ugly-looking expression.
Lets's try to group together terms by factors of $$ h^1_{*} $$:

$$
    \begin{align*}
    \frac{dC(y_N)}{d{}_Tw^1} &= \frac{dC}{dy} \Big |_{y_N} \Bigg \{ \\
    & \frac{d\sigma}{dz} \Big |_{z^2_N} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} h^1_{N-1}\\
    & + \left (\frac{d\sigma}{dz} \Big |_{z^2_N} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} + \frac{d\sigma}{dz} \Big |_{z^2_N} {}_Tw^2  \frac{d\sigma}{dz} \Big |_{z^2_{N-1}} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} \right ) h^1_{N-2} \\
    & + \left (\frac{d\sigma}{dz} \Big |_{z^2_N} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} {}_Tw^1 \frac{d\sigma}{dz} \Big |_{z^1_{N-2}} + \frac{d\sigma}{dz} \Big |_{z^2_N} {}_Tw^2  \frac{d\sigma }{dz} \Big |_{z^2_{N-1}} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_{N-1}} {}_Tw^1 \frac{d\sigma}{d z} \Big |_{z^1_{N-2}} +\frac{d\sigma}{dz} \Big |_{z^2_N} {}_Tw^2  \frac{d\sigma}{dz} \Big |_{z^2_{N-2}} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_{N-2}} \right ) h^1_{N-3} \\
    & + \dots  \Bigg \}.
    \end{align*}
$$

The nice thing about rearranging the terms in this way is that you get a visual idea of the error terms *trickling back* through layers and through time.
For example, look at the line that ends with $$ h^1_{N-1} $$: it already contains the seed for the following line's term $$ \frac{d\sigma}{dz} \Big |_{z^2_N} {}_Sw^2 \frac{d\sigma}{dz} \Big |_{z^1_N} {}_Tw^1 $$.
This is the essence of backpropagation, where we can compute the errors at time $$ N-1$$ and backpropagate an error signal to compute the error at time $$ N-2 $$.
In the end, the total error will be the sum of the errors at each instant.

A similar derivation can and should be done for $$ \frac{dC(y_N)}{d{}_Sw^2}, \frac{dC(y_N)}{d{}_Sw^1} $$ but I don't have the time or space to do it here.
Instead, let me go straight to the vector notation and a wonderful visualization!

## Vector notation

Imagine now that instead of a single neuron per layer, we had multiple neurons.

![recurrent-3D]({{ site.url }}/assets/recurrent-layers.svg)

Doing the [same tricks](../07/naive-backprop.html) as for a feedforward network, we can write the backprop algorithm in vector notation:

$$
\begin{align*}
&\text{ backwards for each layer } l = L, \dots, 1 \\
&\text{ backwards in time } n = N, \dots, 1 \\
&\delta = {}_T\delta^l_{n+1} + {}_S\delta^{l+1}_n \\
&\Delta {}_SW^l = \Delta {}_SW^l + \left [ \delta \odot \frac{d\sigma}{dz} \Bigg |_{\mathbf{z}^l_n} \right ] \otimes \mathbf{h}^{l-1}_n \\
&\Delta {}_TW^l = \Delta {}_TW^l + \left [ \delta \odot \frac{d\sigma}{dz} \Bigg |_{\mathbf{z}^l_n} \right ] \otimes \mathbf{h}^{l}_{n-1} \\
&{}_S\delta^l_n = ({}_SW^l)^T \left [ \delta \odot \frac{d\sigma}{dz} \Bigg |_{\mathbf{z}^l_n} \right ] \\
&{}_T\delta^l_n = ({}_TW^l)^T \left [ \delta \odot \frac{d\sigma}{dz} \Bigg |_{\mathbf{z}^l_n} \right ] .
\end{align*}
$$

It can be easy to visualize it all in sort of grid, with layers (space) on one axis and time on the other.

![recurrent-backprop]({{ site.url }}/assets/recurrent.svg)

To formulas in the image may not reflect with 100% accuracy the notation of the text.
Please refer to the text as the more up-to-date (and thus, hopefully, correct) version.

### TBTTP: Truncated Backpropagation Through Time

it may not be desirable to go back until the dawn of time for every backpropagation step.
So, a simple but effective technique is used which consists in *truncating* the backward step in time after a given amount of steps $$ \tau_{max} $$.

Another technique often coupled with this one is to avoid performing backprop through time at every time step, but only do it every $$ \tau_b $$ steps.

## Show me the code!

A simple implementation that pretty much follows the notation from this blog post is [here](https://github.com/sharkovsky/sharkovsky.github.io/blob/master/code/FullyConnected-py3.ipynb).
In particular, the code implements a many-to-one architecture where the network ingests a full sequence and produces a single number as output.
As such, it only performs BPTT at *the end* of the sequence, instead of at every time step or every $$ \tau_b $$ steps as descrbied above.
