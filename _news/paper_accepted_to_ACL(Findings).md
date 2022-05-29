---
layout: post
date: 2022-02-26
inline: false
title: New paper on exposure bias accepted to ACL (Findings).
---

## Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation.

My recent work on analyzing exposure bias and quantifying its impact was accepted in ACL (Findings) 2022. 

In this work, we analyzed the language generation from an imitation learning perspective. We posed language generation as a sequential decision-making problem, and language modeling as imitation learning problem. We then showed the equivalence of maximum likelihood training for language modeling and behavior cloning-- an imitation learning algorithm. 

Behavior cloning is known to suffer from accumulation of errors. This error accumulation is analyzed using inference-time regret of the model policy. This regret has been shown to grown near-quadratically w.r.t. to sequence length. 

We borrowed this analysis to language generation to show

* *Error accumulation does happen during language generation.*
* *Perplexity failes to capture this accumulation.*
* *Error accumulation correlates with poor generation quality.*

Please see the paper and the repo below for more details.

Paper: [https://arxiv.org/abs/2204.01171](https://arxiv.org/abs/2204.01171)

Code: [https://github.com/kushalarora/quantifying_exposure_bias](https://github.com/kushalarora/quantifying_exposure_bias)

ACL 2022 Slides: [slides](/assets/acl_2022_qeb_slides.pdf)