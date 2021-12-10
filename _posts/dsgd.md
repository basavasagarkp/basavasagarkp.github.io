---
title: 'Stochastic Gradient Descent, Why so popular?'
date: 2021-11-23
permalink: /posts/dsgd
tags:
  - optimization
  - stochastic gradient descent
  - machine learning
---

# Stochastic Gradient Descent, Why so popular?

Stochastic Gradient Descent is probably one of the most widely used algorithms maybe beside FFT(my favorite though, truly ingenious).  Its widely used in optimization theory (of-course!), control systems, deep learning and any domain in which it is required to optimize parameters of the function to take a minimum value. One of the main reasons for its popularity is it's very easy to understand and one can get a quite good intuition by just seeing the equation. However, behind its simple formulation hides a monster which has to be tackled in-order to successfully apply the algorithm to ensure convergence (by convergence I mean to reach to a close optimal value within a specified error range). 

Let's go back to basics, differentiation of a function at a point on a function gives us the slope of tangent at that point. Why not call this slope (steepness) with a fancy name? Let's call it **Gradient**. Gradient can also be understood as change in the output when there is small change in the input. Hmm, how is it helpful to find out the minimum value of a function. Well, it so happens that the Gradient of a function at a point also gives the direction of steepest ascent, but wait we want minimum value not the max, so let's put a negative sign in front of it (:blush:).  Now its called Gradient Descent. 

Let us a take this example where we want to find the minimum value of the function given a random point. Now gradient descent is an iterative algorithm which calculates the Gradient(derivative) of the function at this random point.  This gives us the steepness of the curve and the direction of ascent. If we subtract this quantity from our random point we will move slightly (the amount we move is decided by a very important parameter called **Learning Rate**) towards the minimum value. ![sgd_image](/home/sagar/md file/sgd_image.webp)

Okay all these  look nice for simple functions like above, but is it useful for high dimensional function with many minima's and how fast will it converge to the minima if at all it converges. To answer these question we have to dig deep into early formulation of the algorithm proposed by Robbins & Munro. The subsequent analysis is also called as Stochastic Approximation by the Optimization Field folks. 

## Stochastic Approximation

What, after all, is stochastic approximation? Historically, stochastic approximation started as a scheme for solving a nonlinear equation _h(x)=0_ given 'noisy measurements' of the function _h_. That is, we are given a black box which on input _x_, gives as its output $h(x)+\xi$, where $\xi$ is a zero mean random variable representing noise. The stochastic approximation scheme proposed by Robbins and Monro $(1951)^{\dagger}$ was to run the iteration
$$
x_{n+1}=x_{n}+a(n)\left[h\left(x_{n}\right)+M_{n+1}\right],
$$
where $\left\{M_{n}\right\}$ is the noise sequence and $\{a(n)\}$ are positive scalars. The expression in the square brackets on the right is the noisy measurement. That is, $h\left(x_{n}\right)$ and $M_{n+1}$ are not separately available, only their sum is. We shall assume $\left\{M_{n}\right\}$ to be a martingale difference sequence. Well what is martingale difference sequence? It requires its own blog which I will link if I get time to write one, for now I will put in the definition from wikipedia(:wink:). Martingale differnce sequence is related to the concept of Martingale in probablity theory, where Martingale is defined as a sequence of random variables for which, at a particular time, the conditional expectation of the next value in the sequnce is equal to the present value, regardless of all prior values. 

