---
layout: post
title:  "A naive explanation of backprop"
date:   2017-06-07 13:53:10 +0200
---

I got lost trying to implement some simple neural networks from scratch in python.
In particular, I figured out that I didn't really understant how backpropagation works and what is the magic behind it.
I decide to inaugurate a blog with my thoughts on backpropagation.

### The handwaving backprop explanation

Everyone is talking about it, and everyone has their own, two-sentence explanation that they casually drop in conversation when they want to look cool at a conference.
I like to think of myself as a funny guy, so I usually tell the joke:

> Backpropagation is merely a rebranding of the chain rule. Yes, I find it quite..... derivative.

if you haven't fallen off your chair, let me restate what pretty much everybody says about backprop: *"in neural networks, the error is associated with the gradient of the cost function. Thanks to the backpropagation algorithm, we have a fast and efficient way of computing this gradient w.r.t every parameter of the model."*

#### Why gradients?

This might seem like a pretty silly question, but it still required me some thinking: *why exactly do we use the gradient as an error signal?*

There are obviously some intuitive explanations for this: as a first approximation, a small change in a parameter corresponds to a change in the output that is proportional to the gradient w.r.t. that parameter.
However, the moment of truth that finally put my mind at ease was the realization that using gradients as an error signal is only a consequence of the **arbitrary** decision of using Gradient Descent as an optimization algorithm.
If we were to use different methods (e.g. Newton's) we would be computing more than just gradients (see e.g. section 8.6 of the [deep learning book](http://www.deeplearningbook.org/)).
