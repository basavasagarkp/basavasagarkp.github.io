---
title: "Convergence Analysis of Distributed Stochastic Gradient Descent Algorithms"
excerpt: "Distributed SGD<br/><img src='/images/stochastic_image.jpeg'>"
collection: portfolio
---

With ever growing size of datasets, it is important for algorithms to perform with same or better accuracy in less time. Stochastic Gradient Descent(SGD) is often the most used algorithm in these type of settings. In order to run SGD on multiple machines or on multiple core processor, Distributed SGD was adapted. There are two approaches for distributed SGD, Synchronous and Asynchronous, both of which are widely used. Synchronous SGD accumulates all the gradients before an update, while the asynchronous SGD updates after receiving gradient from each worker. The problem with the latter is the stale gradient. We are going to analyze the convergence rates of vanilla SGD with fixed and varying step sizes, convergence rates of ASGD with fixed and varying step sizes.

Dowload the Report from [here](https://drive.google.com/file/d/1uEx20Tt-tSjxjFDjL_jM16Ubwog2QtxS/view?usp=sharing)

PPT from [here](https://drive.google.com/file/d/1FDizojSglez8gJWJzVTmBAHnrRbn2PUS/view?usp=sharing)
