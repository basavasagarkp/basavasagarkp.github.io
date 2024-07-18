### 				What are Modes in Diffusion Models?

Diffusion models are great, they take the idea of iterative noise into a data point, which is like gradient descent at inference. Slowly, these models are peneterating robotics as well, because of one main feature that I want to explain here. It is their ability to learn different modes in the data and conditionally generate a sample from one of these modes at inference time. 

### But what are modes?

 Assume that your data has multiple distinct patterns or clusters. For example, if you're working with images of different types of objects, each type might represent a different mode. In robotics, at a high level this could translate to various modes of operation, such as walking, running, or climbing for a legged robot or at a low level this could translate to different ways of picking object, different turns you could take to reach the same goal position.

In statistical terms, a mode is a peak in the probability distribution of your data. In simpler terms, it's a pattern or behavior that is commonly found in your dataset. Traditional generative models might struggle to capture these distinct modes effectively, often resulting in blurry or averaged samples that don't clearly belong to any specific mode.

### Gaussian Mixture Models (GMMs)

Well, the concept of modes and the need to learn them is nothing new, [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html) have been known since 1950s. However, there are multiple problems with them, but their one main problem is mode collapse. Mode collapes occurs in GMMs when multiple true modes in the data are merged into a single Gaussian component. This happens because the model tries to fit the data into a predefined number of Gaussians, which may not match the true number of modes.

### Diffusion Models

I already assume that you know a little bit theory about Diffusion Models, but you can learn about them [here](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). One of the main advatanges of diffusion models is their ability to model different modes without collapsing and also their flexible conditional generation. 

Why do Diffusion Models generalize? 

Well probability modeling has been around for a long time, and the theory behind diffusion has also been known for a long time, but the main problem with MCMC has been it takes exponential time for the models to converge to move from one high distribution mode to another. The way modern diffusion models overcome this is by using smart heuristics learned by neural networks.  

