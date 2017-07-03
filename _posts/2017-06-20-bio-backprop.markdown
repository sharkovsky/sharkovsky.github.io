---
layout: post
title:  "A Summary of Biologically Plausible Deep Learning and Equilibrium Propagation"
date:   2017-06-20 13:53:10 +0200
---

In the past couple of years, Y. Bengio has been interested in the question of whether and how could backpropagation be implemented in a biologically-realistic framework.
I find his work extremely interesting, and have decided to summarize _my own personal understanding_ of it in this blog post.

In particular, I will try to combine information from these three papers:
- [1] [Towards Biologically Plausible Deep Learning](https://arxiv.org/abs/1502.04156) (v3 on ArXiv) ;
- [2] [Early Inference in Energy-Based Models Approximates Back-Propagation](https://arxiv.org/abs/1510.02777) (v2 on ArXiv);
- [3] [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) (v5 on ArXiv).

## The problem of credit assignment

One of the main motivations for Bengio's work (see [1]) is finding a

> credible machine learning interpretation of the learning rules that exist in biological neurons [...] accounting for credit assignment through a long chain of neural connections.

In other words, answering the question of _how does the brain propagate an error?_

There are many aspects to this question, all of which represent interesting unsolved problems:
- how does the brain propagate the signal that a certain prediction was wrong, through its many hidden layers and recurrent connections?
- how does the brain propagate the signal of the magnitude of a given error?
- how does the brain interpret that signal and produce a valid update of its synaptic weights?

Now is a good time to make an important distinction: obviously there are several physiological implementations of all the mechanisms cited above (e.g. [NDMA recruitment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3367554/) in Long Term synaptic Potentiation), and studying these biological mechanisms is an extremely important aspect of neuroscience.
However, in this work we are going to be taking a more abstract view, and asking ourselves the question of _what are the relevant computational (mathematical) aspects related to the questions above?_

## Biologically plausible, whatever do you mean?

Well, this is where the boundaries of objectivity start to become blurred like the mists of Avalon.
There are many many properties observed experimentally in _biological_ neural networks, and ideally a biologically-plausible artificial neural network would implement them all.
Since this is proving to be very hard, researchers must make decisions on which experimentally observed properties they'd like to preserve in their models.
_Enter subjectivity._

#### Reasons why backprop is not biologically plausible

The paper [1] cites 6 reasons why backprop is not biologically plausible:
1. backpropagation is _purely linear_ (I'm not sure in what sense);
2. if backpropagation were a thing (biologically), the feedback paths would require exact knowledge of the derivatives of the activation functions of neurons, and to this time this has never been observed experimentally;
3. similary, feedback paths would be required to have the same exact weights (but transposed) of the feedforward paths, a.k.a the [weight transport problem](https://arxiv.org/pdf/1411.0247.pdf);
4. backpropagation is based on the idea that communication between neurons happens via (smooth) continuous values, but experimental evidence shows that neurons communicate via binary discontinuous events (spikes) although there is still a lot of [discussion](http://romainbrette.fr/why-do-neurons-spike/) on how the brain interprets these events;
5. backpropagation happens in two phases (feedforward and backprop), which would require the brain to have a perfectly synchronized clock that alternates between the two;
6. there is no widely-accepted theory of what the output targets (i.e. cost function) of the brain should be, nor what its biological substrate would look like.

#### Boltzmann machines as a biologically plausible model

In Bengio's point of view, [Boltzmann machines](http://www.scholarpedia.org/article/Boltzmann_machine) represent the most biologically plausible learning algorithm for deep architectures.
I will try to summarize what I think can be some explanations for this:
- Boltzmann machines are based on (stochastic) binary units, so they address concern number 4 above.
These units are often divided in _visible_ and _hidden_ units, and in the case of [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) this division gives rise to two separate layers without inter-layer connections.
- Boltzmann machines do not implement backprop but an [unsupervised learning](http://ac.els-cdn.com/S0364021385800124/1-s2.0-S0364021385800124-main.pdf?_tid=4f334284-55d2-11e7-8e6b-00000aab0f27&acdnat=1497974921_7ca4e6ecdfae3729c94e4c45295808c0) algorithm, which addresses concern 6 above.
Indeed, Boltzmann machines are a _generative model_ in the sense that they learn a probability distribution of the input dataset conditioned on some internal representation.
In order to use BMs as a classifier, one needs to additionally train a classifier on top of the hidden units, an approach that is reminiscent of [reservoir computing](https://en.wikipedia.org/wiki/Reservoir_computing).
In this case, however, the concerns raised in point 6 are still valid, as it is not clear what the output targets of the braing would be.
- Training Boltzmann machines does not require explicitly computing any derivative, which addresses point 2.
- I'm not sure how point 1 above should be interpreted, but Bengio's work implicitly assumes that it is also addressed by Boltzmann machines.

Unfortunately, two issues remain with the biological plausibility of BMs: the _weight transport problem_ (point 3) and the two-phase training (point 5).


# Equilibrium propagation: an even more biologically plausible model

Bengio proposes in [2] a new learning framework that addresses the issue of two-phase learning.
Bengio's framework doesn't completely remove the second phase, but still addresses the issue in the sense that _only one kind of neural computation is required for both phases_, something that is not true for standard Boltzmann machines.

The main steps of the Equilibrium Propagation algorithms are:
- inference is energy minimization, just like in Boltzmann machines, although the actual formual used for the energy is different;
- we distinguish three subsets of the units in the network, namely input, hidden and output units;
- a total energy is defined as the sum of an internal energy the models the interactions within the network and an external energy (modulated by a constant) that models how the targets influence the output units;
- learning happens in two phases: the _free phase_ where all the neurons are performing inference, and the _weakly clamped phase_ where the output units are nudged towards an (observed) target;
- the input units are always clamped, the hidden units are always free, the output units are free during the first phase and _weakly clamped_ during the second phase;
- weights are updated according to a rule with a formulation similar to classical Boltzmann machines, but which is shown [2,3] to approximate gradient descent on the least squares error.

There are some additional compelling arguments for the biological plausibility of Equilibrium Propagation:
- although the two-phases are not completely removed, the fact that they require the same kind of neural computation means that they are more open to biologically-plausible interpretations, for example
  - when you move your arm you make a prediction of where it is going to be, and your eyes allow you to see if the prediction is correct;
  - you see an image and you are trying to understand what it contains, and after a while a "teacher" tells you its contents;
- Bengio et al. show in [1,3] that their learning rule can be interpreted as the classic [STDP learning rule](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity);











