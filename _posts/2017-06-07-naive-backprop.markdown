---
layout: post
title:  "A naive explanation of backprop"
date:   2017-06-07 13:53:10 +0200
---

I got lost trying to implement some simple neural networks from scratch in python.
In particular, I figured out that I didn't really understant how backpropagation works and what is the magic behind it.
I decide to inaugurate a blog with my thoughts on backpropagation.

#### Why gradients?

This might seem like a pretty silly question, but it still required me some thinking: *why exactly do we use gradients as an error signal?*

There are obviously some intuitive explanations for this: as a first approximation, a small change in a parameter corresponds to a change in the output that is proportional to the gradient w.r.t. that parameter.
However, I felt more satisfied when I realized that using gradients as an error signal is a consequence of the **arbitrary** decision of using Gradient Descent as an optimization algorithm.

