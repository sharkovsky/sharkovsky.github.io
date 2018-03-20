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
Obviously, we are far from understanding how the brain works, and these are all mathematical abstractions only loosely related to the physical/biological phenomenon.

The most prominent author of Boltzmann machine related papers is G. [Hinton](http://www.scholarpedia.org/article/Boltzmann_machine) (see his [practical guide](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) ), but Y. [Bengio](http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf), I. [Goodfellow](https://papers.nips.cc/paper/5024-multi-prediction-deep-boltzmann-machines.pdf), A. [Krizhevsky](http://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) and others have made significant contributions.
Moreover, although I had to wander all the way to the second page of Google's results to find it, [Gorayni's blog post](http://gorayni.blogspot.ch/2014/06/boltzmann-machines.html) cleared many of my doubts about the actual implementation, and the [MacKay book](http://www.inference.org.uk/itprnn/book.pdf) he mentions turned out to be a gem of clarity for the theoretical aspects.

#### Why Boltzmann machines?

Boltzmann machines are a _generative model_.
This means that they can learn, in an unsupervised way, the probability distribution of your data $$p(data)$$, or in a supervised way $$p(data | label)$$.
As always, Andrew Ng's [explanation](http://cs229.stanford.edu/notes/cs229-notes2.pdf) is worth a thousand of mine.

Moreover, there are three ways (see [chapter 16](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) ) in which one could use Boltzmann machines as a _discriminative model_ instead: train a classifier on top of a subset of the neurons, train a different BM for each class, or train a single BM with two sets of input neurons.

### Network topology

Boltzmann machines are characterized by arbitrary recurrent, **symmetric** connections.
Usually, self connections are also disallowed.

The neurons (or units) are split in two non-overlapping groups: _visible_ and _hidden_ units.

![boltzmann-topology]({{ site.url }}/assets/boltzmann-net-topology.svg)

This distinction is important because it allows the boltzmann machine to learn a _latent_ representation of the data, so the BM can learn not only $$p(data)$$ but also $$p(data|hidden~units)$$.
Moreover, hidden units allow the Boltzmann machine to model distributions over visible state vectors that cannot be modelled by direct pairwise interactions between the visible units (see section [learning with hidden units](http://www.scholarpedia.org/article/Boltzmann_machine)).
This distinction is also at the basis of [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine).

### Inference as energy minimization

This is the idea that the inference process consists of fixing some part of the network and letting the rest _evolve_ naturally to a state of lower energy.
This contrasts with a more static view of neural networks, where the activation of each neuron is computed only once, statically, as a function of the input values.
This point of view is not specific to BMs, actually is used to define a broad range of models under the umbrella term [energy based models](http://www.iro.umontreal.ca/~bengioy/ift6266/H14/ftml-sec5.pdf) and in particular was used before in [Hopfield networks](https://en.wikipedia.org/wiki/Hopfield_network).
Y. Le Cun's [tutorial](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) is a good starting point for understanding energy based models.

I still find it difficult to understand the relationship between inference and energy in an intuitive way.
I will now try to explain the explanation that I've been giving to myself:
- in a stochastic neural network, performing inference means letting the network evolve, possibly under some constraints, until a state that is in some sense stable is reached;
- in this sense, inference is then equivalent to sampling from the probability distribution of possibile network states;
- the relationship between the probability distribution of the network states and the energy has a [simple formulation](https://www.quora.com/How-are-energy-based-models-in-deep-learning-related-to-probability)
$$ p(\mathbf{x}) = \frac{e^{ -E(\mathbf{x}) }}{Z} $$;
- given the formula above, states with lower energy have a higher probability.

Note that in this setting energies can have negative values.

Finally, the example on [slide 35](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec11.pdf) is what really made the idea click for me.
This exmaple is very useful to understand how **connection weights define a probability distribution**.
Consider the following, very simple, Boltzmann machine with an 2-dimensional input vector $$\mathbf{v} = (v1,v2)$$ and two hidden units.

![weights define probability]({{ site.url }}/assets/weights-def-proba.svg)

Then we can compute the probability distribution $$ p(\mathbf{v}) $$ induced by the weights by considering all possible combinations of inputs and hidden units, following this process:
1. compute the energy associated with every network configuration (formula in section below);
2. compute the _partition function_, i.e. the normalizing factor $$Z$$, as the sum over all possible combinations $$ Z = \sum\limits_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})} $$;
3. compute the probability associated with a particular network configuration $$( \mathbf{v_0}, \mathbf{h_0})$$as $$ \frac{ e^{-E(\mathbf{v_0}, \mathbf{h_0})} }{Z}  $$;
4. the probability distribution of the input data is given by the total probability $$p(\mathbf{v}) = \sum\limits_{\mathbf{h}} p(\mathbf{v}, \mathbf{h}) $$.

And we can build a table exactly as in [slide 35](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec11.pdf)

v1  | v2  | h1  | h2  | -Energy  | exp( -Energy) | joint probability $$p(\mathbf{v}, \mathbf{h}) $$ 
--- | --- | --- | --- | :------: | :-----------: | :-----------------------------------------: 
1   |  1  | 1   | 1   | 2        | 7.39          | .186 
1   |  1  | 1   | 0   | 2        | 7.39          | .186 
1   |  1  | 0   | 1   | 1        | 2.72          | .069 
1   |  1  | 0   | 0   | 0        | 1             | .025 
1   |  0  | 1   | 1   | 1        | 2.72          | .069 
1   |  0  | 1   | 0   | 2        | 7.39          | .186 
1   |  0  | 0   | 1   | 0        | 1             | .025 
1   |  0  | 0   | 0   | 0        | 1             | .025 
0   |  1  | 1   | 1   | 0        | 1             | .025 
0   |  1  | 1   | 0   | 0        | 1             | .025 
0   |  1  | 0   | 1   | 1        | 2.72          | .069 
0   |  1  | 0   | 0   | 0        | 1             | .025 
0   |  0  | 1   | 1   | -1       | 0.37          |  .009 
0   |  0  | 1   | 0   | 0        | 1             | .025 
0   |  0  | 0   | 1   | 0        | 1             | .025 
0   |  0  | 0   | 0   | 0        | 1             | .025 

which allows us to compute the marginalized probabilities

$$\mathbf{v}$$ | marginal probability $$p(\mathbf{v})$$  | formula
-------------- | :-------------------------------------: | -------------------------
(1,1)          | .466                                    | .186 + .186 + .069 + .025
(1,0)          | .305                                    | .069 + .186 + .025 + .025
(0,1)          | .144                                    | .025 + .025 + .069 + .025
(0,0)          | .084                                    | .009 + .025 + .025 + .025

### Definition of energy

How did we compute the enrgy values in the table above?
The formula is actually pretty simple: consider the symmetric connection weight matrix $$W$$, biases $$\mathbf{b}$$ and the state vector $$\mathbf{s} = (\mathbf{v}, \mathbf{h})$$, then the (scalar) energy is given by:

$$ E = -\frac{1}{2}\mathbf{s}^TW\mathbf{s} - \mathbf{b}^T\mathbf{s}. $$

N.B. vectors are column vectors here.
To see this formula in action, consider the BM above.
Then

$$ W = \begin{matrix}
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 1 \\
2 & 0 & 0 & -1 \\
0 & 1 & -1 & 0 \end{matrix}, $$

and the biases are all zero.
The energy of configuration $$ \mathbf{s} = ( 1, 0, 1, 1) $$ would be $$ E = -\frac{1}{2} ( 2w_{13} + 2w_{14} + 2w_{34}) = -2 - 0 - (-1) = -1$$.

### The neuron dynamics during inference

BM neurons can take only two values: 1 and 0.
The crucial idea behind letting BMs states' evolve is that it must be done in a sequential way, _one neuron at a time_.
In theory, one can select a candidate transition for the $$i^{th}$$ neuron, compute the difference in energy between the original and the transitioned state, and decide whether to reject or keep the transition based on the probability

$$ p ( keep~transition) = \frac{1}{1 + e^{E(transition) - E(original)} } $$.

Incidentally, this is what I do in my code.

Otherwise, the [wikipedia page](https://en.wikipedia.org/wiki/Boltzmann_machine) gives a nice derivation of the formula for how energy relates to the probability of the $$i^{th}$$ neuron being on or off and which also explains where the formula for the above probability comes from.
The gist is the following: we rely on the property of [Boltzmann distributions](https://en.wikipedia.org/wiki/Boltzmann_distribution) that the energy of a state is proportional to the negative log probability of that state:

$$ E( \mathbf{s}_i = 0 ) - E( \mathbf{s}_i = 1 ) = ln( p( \mathbf{s}_i = 1 ) ) - ln( p( \mathbf{s}_i = 0 ) ) $$,

and by doing some algebra we get to

$$ p( \mathbf{s}_i = 1 ) = \frac{1}{1 + e^{ E( \mathbf{s}_i = 1 ) - E( \mathbf{s}_i = 0 ) } } .$$

So far we have stated the formulas that allow to decide whether the state transition of a single neuron should happen, so how does this fit in the global picture?

# Enter MCMC

No, MCMC is not a rapper suffering from [Palilalia](https://en.wikipedia.org/wiki/Palilalia).

[Markov Chain Monte Carlo (MCMC)](https://theclevermachine.wordpress.com/2012/11/19/a-gentle-introduction-to-markov-chain-monte-carlo-mcmc/) methods, and in particular [Gibbs sampling](https://theclevermachine.wordpress.com/2012/11/05/mcmc-the-gibbs-sampler/), are the answer to the above problem.
The idea is the following: the overall state of the network's dynamics follow a Markov Chain, where each state transition's probability is defined as above; to compute the dynamics of the network, we need to run this Markov Chain to convergence.
The concept of convergence here is defined mystically as the moment when the probability distribution of the states does not depend anymore on the initial value, but is only a function of the energy of each state.
In practice, I have not found a single resource explaining how to understand whether your network has _converged_ or not, and in  a later section I'll explain how I did it for a very very very simple case.

To run the Markov Chain to convergence, we use the MCMC method which consists in repeatedly sampling from the distribution of possible states.
In particular, we repeatedly use Gibbs sampling:
- until _convergence_, do:
  - for each neuron in the network, do:
    - compute transition probability for this neuron, **keeping the rest fixed**
    - randomly decide to keep or reject transition according to probability

As we said before, given the mystical definition of _convergence_, we often run the above loop for a fixed number of iterations.
As a cheaper alternative, we can resort to simulated annealing instead.

# Simulated annealing

Running the MCMC to convergence everytime can be very resource consuming.
To mitigate this problem, we adopt an optimization technique called [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing).

Simulated annealing introduces the concept of _temperature_ $$T$$, a value which modifies the transition probability to:

$$ p ( keep~transition) = \frac{1}{1 + e^{ \frac{ E(transition) - E(original)}{T}} }. $$

This is how the the Markov Chain loop is modified in the presence of simulated annealing:
- given an _annealing schedule_
- for each (temperature value $$T$$, number of iterations $$N$$) tuple in the _annealing schedule_, do:
  - repeat the full Gibbs sampling procedure $$N$$ times, at a temperature value of $$T$$.

### Training... Finally!

If inference means energy minimization, then we want to find a training procedure that will arrange the connection weights such that the observed data have very low energy.
An equivalent way to put it is to say that we want to find weights and biases that define a Boltzmann distribution in which the training vectors have high probability.
Following the procedure described on the [scholarpedia page](http://www.scholarpedia.org/article/Boltzmann_machine), assuming the training data were sampled i.i.d. we can write a gradient ascent formula for the log likelihood as

$$ \Delta W = \sum\limits_{\mathbf{v}} ln \left ( \frac{ d p(\mathbf{v},\mathbf{h}) }{dW} \right ). $$

To compute the expression above we need to define two quantities: 
- $$ \mathbf{s}^{\mathbf{v}} $$ is a sample of the network state vector that was obtained by running the MCMC procedure to _convergence_, **while keeping the input vector $$ \mathbf{v} $$ fixed**;
- $$ \mathbf{s}^m $$ is a sample of the network state vector that was obtained by running the MCMC procedure to _convergence_ having left all units unclamped.

Given that we know the relationship between the probability $$ p(\mathbf{v},\mathbf{h}) $$ and the energy, i.e. $$ p(\mathbf{v},\mathbf{h}) = \frac{ e^{-E(\mathbf{v})}}{\sum\limits_{\mathbf{u},\mathbf{g}} e^{-E(\mathbf{u},\mathbf{g})} } $$ we can [write](https://theclevermachine.wordpress.com/tag/gibbs-sampling/):

$$ \Delta W = \sum\limits_{\mathbf{h}} p(\mathbf{h} | \mathbf{v} )\mathbf{s}^{\mathbf{v}}(\mathbf{s}^{\mathbf{v}})^T - \sum\limits_{\mathbf{h},\mathbf{v}} p(\mathbf{v}, \mathbf{h} )\mathbf{s}^m(\mathbf{s}^m)^T .$$

Computing the exact quantities above requires summing over many many possible configurations.
In practice this can rapidly become unfeasible, so we substitute the above quantity with:

$$ \Delta W = \frac{1}{N_{\mathbf{v}}}\sum\limits_{\mathbf{v}} \mathbf{s}^{\mathbf{v}}(\mathbf{s}^{\mathbf{v}})^T -\frac{1}{N_m} \sum \mathbf{s}^m(\mathbf{s}^m)^T, $$

which corresponds to the following procedure (with a learning rate $$\eta$$):
- initialize updates $$\Delta W = 0, \Delta \mathbf{b} = 0 $$;
- for every training data point, do:
  - keeping the visible units clamped to the values of the training data point, run MCMC to convergence;
  - update  $$\Delta W = \Delta W + \mathbf{s}\mathbf{s}^T, \Delta \mathbf{b} = \Delta \mathbf{b} + \mathbf{s} $$;
- update $$ W = W+ \frac{\eta}{N_{\mathbf{v}}} \Delta W, \mathbf{b} = \mathbf{b} + \frac{\eta}{N_{\mathbf{v}}} \Delta \mathbf{b} $$;
- reset updates $$\Delta W = 0, \Delta \mathbf{b} = 0 $$;
- for a certain number iterations $$N_m$$, do:
  - allow the MCMCM to run to convergence, without clamping anything;
  - update  $$\Delta W = \Delta W + \mathbf{s}\mathbf{s}^T, \Delta \mathbf{b} = \Delta \mathbf{b} + \mathbf{s} $$;
- update $$ W = W - \frac{\eta}{N_m} \Delta W, \mathbf{b} = \mathbf{b} - \frac{\eta}{N_m} \Delta \mathbf{b} $$;

