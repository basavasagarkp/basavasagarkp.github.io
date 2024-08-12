---
layout: post
title: "Modes in Diffusion Models"
author: "Sagar"
date: 2024-07-14
---
{%- include mathjax.html -%}


## 				What are Modes in Diffusion Models?

Diffusion models are great, they take the idea of iterative noise into a data point, which is like gradient descent at inference. Slowly, these models are peneterating robotics as well, because of one main feature that I want to explain here. It is their ability to learn different modes in the data and conditionally generate a sample from one of these modes at inference time. 

## But what are modes?

 Assume that your data has multiple distinct patterns or clusters. For example, if you're working with images of different types of objects, each type might represent a different mode. In robotics, at a high level this could translate to various modes of operation, such as walking, running, or climbing for a legged robot or at a low level this could translate to different ways of picking object, different turns you could take to reach the same goal position.

In statistical terms, a mode is a peak in the probability distribution of your data. In simpler terms, it's a pattern or behavior that is commonly found in your dataset.
<div class="image-container">
<figure>
  <img src="../../../../assets/images/modes.png" alt="Description of Image" width="525" height="500"/>
  <figcaption> Three modes in a distribution </figcaption>
</figure>
</div>

## Gaussian Mixture Models (GMMs)

Well, the concept of modes and the need to learn them is nothing new, [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html) have been known since 1950s. However, there are multiple problems with them, but their one main problem is mode collapse. Mode collapes occurs in GMMs when multiple true modes in the data are merged into a single Gaussian component. This happens because the model tries to fit the data into a predefined number of Gaussians, which may not match the true number of modes.

<div class="image-container">
<figure>
  <img src="../../../../assets/images/gmm_mode_collapse.png" alt="Description of Image" width="525" height="500"/>
  <figcaption> An Illustration of Mode Collapse in GMMs when
  we do not know the true number of modes </figcaption>
</figure>
</div>


## Modes in Diffusion Models

I already assume that you know a little bit theory about Diffusion Models, but you can learn about them [here](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Diffusion models model the distribution by learning a sequence of noise-adding steps and then reversing this process to generate samples.

But why doesn't mode collapse occur in Diffusion Models (compared to other methods)? The iterative noise-addition and reversing process (also called iterative denoising step) in diffusion act as an implicit regularization term which prevent the model from overfitting to specific training examples. This iterative denoising procedure also helps in learning multiple scales in data, where early steps focus on large broad structures or mean of the distribution and later steps focus on refining details. 

<div class="image-container">
<figure>
  <img src="../../../../assets/images/diffusion_mean_blog_comp.gif" alt="Description of Image" width="525" height="500"/>
    <figcaption>An Illustration of how diffusion models learning at different scales of data during iterative timesteps. Adapted from <a href="https://x.com/alec_helbling/status/1783378117249089625?t=yvEXY9j9DVOhcVzPsSDc-g&s=08">Alec Helbling</a></figcaption>

</figure>
</div>


For example in image dataset multiple scales could be different levels of abstractions in the image, such as overall composition, significant textures, fine-grained details. And in robotics this could mean high-level plan to move from one point to another or low-level manipulation skills requiring precise movements. 

Well probability modeling has been around for a long time, and the theory behind diffusion has also been known for a long time. But what has really changed is we have gotten really good at training large neural networks on dataset, that is the main difference between what was known vs what has recently been done. Neural networks with all the ingridients to remove obstacles in their learning have paved way to efficiently learn these modes in distributions.   

Another main reason why diffusion models are able to capture and generate multiple modes in the distribution is stochastic initialization and stochastic sampling procedure. During the sampling procedure, the sample is initialized as a random sample from Gaussian Distribution this ensures that all the possible modes are covered, and the sample is stochastically optimized (which means that a small random noise is added at each timestep, this is a bit exaggerated in the below videos but it shows the point) over large number of iterations, which helps it hop over and converge to different modes.


<div class="image-row">
  <figure>
    <img src="../../../../assets/images/mode_sampling_animation_15_0.gif" alt="Description of Image 1" />
    <figcaption>Conditioned on 0</figcaption>
  </figure>
  <figure>
    <img src="../../../../assets/images/mode_sampling_animation_15_1_mode_1.gif" alt="Description of Image 2" />
    <figcaption>Conditioned on 1</figcaption>
  </figure>
  <figure>
    <img src="../../../../assets/images/mode_sampling_animation_15_0.gif" alt="Description of Image 3" />
    <figcaption>Conditioned on 2</figcaption>
  </figure>
</div>
Another great advantage of diffusion models is that they render themselves to conditional generation, i.e., we can condition the generation process on some input that directs the mode of convergence. That is really cool, not only can we get sample but we can get the sample we want based on high-level conditional data. And the icing on top of the cake is that the conditioning of diffusion models is also flexible, which means we can condition the generation of sample either using a number or a language prompt or an image. This is especially useful in robotics where you want to convert control your robots action or policy using high-level natural language commands. Here in the below example, I sample a data point from the distribution based on numerical specification of the mode.

#### No Free Lunch, here too!

Well, okay diffusion models are able to capture modes better than GMMs while avoiding mode collapse, but it comes at a cost. Look at the below figure, if we have means a bit closer than they are, diffusion model produces samples that are in-between the two modes (mode interpolation). This might be the reason for hallucinations that you see in most of the diffusion models outputs in internet. 

<div class="image-container">
<figure>
  <img src="../../../../assets/images/mode_close.png" alt="Description of Image" width="300" height="300"/>
  <figcaption> When the modes of the data are close, diffusion models interpolate between the modes during sampling causing hallucinations.</figcaption>
</figure>
</div>


Now why this happens?  As mentioned above the diffusion models learn the mean of the distribution first and then specific offsets. In frequency domain this could be termed as learning low-frequency first and then high-frequency. And neural networks particularly CNNs (which are most widely used neural network architecture for diffusion) have high affinity towards <a href="https://arxiv.org/abs/2006.10739"> low-frequency</a>. So they are able to model low-frequency or mean of the distribution perfectly but when it comes to high-frequency components they interpolate. This interpolation in high-frequency domain leads to mode interpolation when they are nearby. This is shown by Aithal et al. [<a href="https://arxiv.org/pdf/2406.09358"> 1 </a>] where they hypothesize that the mode interpolation might be due to inability of the neural network to mode high-frequency change in score function. 

I will shamelessly plug my own paper on this topic, where we tried to solve similar issue (we had in mind the high-frequency change in value function of a RL agent) albeit for lower-dimensional input space. 

Well one can solve this by training longer, but that could come at the risk of overfitting the model and reducing the diversity and also this can become increasingly infeasible as the data increases. *I also hypothesize that training for longer might create a more discontinuous function which when coupled with stochastic denoising can make reliability a huge problem and might also make the models more susceptible to adversarial attacks.*




<div class="image-container">
<figure>
  <img src="../../../../assets/images/diffusion_policy_multimodality.png" alt="Description of Image" width="300" height="300"/>
  <figcaption> Multi-modal paths for achieving the same goal in robotics. Diffusion Model (policy) is able to preserve these modes, which is especially useful in robotics. Adapted from <a href="https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf">Diffusion Policy</a></figcaption>
</figure>
</div>


