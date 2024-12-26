---
layout: post
title: "Robot Learning using two blobs"
author: "Sagar"
date: 2024-12-07
---
{%- include mathjax.html -%}

## Understanding Robot Learning using two blobs

Okay, I am going to argue or convince myself some of the interesting things about robot learning using two blobs. The blobs in our case are manifold for two skills in robot learning. But what is a manifold? It is a mathematical abstraction that specifies a complex geometric object embedded in higher-dimensional space in a lower-dimensional space. Basically, it states that most of the high-dimensional datasets can really be understood as lying on somekind of low-dimensional manifold (the lower dimension is also somtimes called intrinsic dimension).

Now, coming to our blobs, I will borrow some of the visualizations and idea from Task and Motion Planning (TAMP) community for conveying my ideas. 
Assume that we live in a universe where we have a library of skills, each of which are independent and composable and we can use this collection of skills to solve any task. That's a reasonable assumption!

Let us say that we have a task at hand, say "Pick up this mug and place it on the table". We can access the skills needed for this based on this natural language description of the task along with vision (scene description can be necessary sometimes). For this task, we need "Pick" skill and "Place" skill. Let a single-point on these manifolds describe the configuration of our robot, and a line along these manifolds represent a trajectory.

### Composition of skills
We can think of composition of the skills as intersection of the corresponding manifolds in the task space. And the boundary of this intersection can be seen as configurations at which we can switch between these skills (mode in TAMP).
Now, when we train our model on datasets for behavior cloning, we are essentially teaching them how to move along this manifold. So, each demonstrations is a line or trajectory along this manifold and our training set would be a bunch of lines along these planes starting and ending at different positions.

## Model Size and Resolution
I really like this analogy of connecting the scaling laws to the resolution of the task-space manifolds. When you have a small model or a small dataset, the manifold is kinda blurry and you have a rough (noisy) esimate of where you are in the current task and how you move from their to a configuration where you can switch between skills. But as you increase your model-size as well as the dataset size, the resolution of the manifold increases as well. Now you will have a better (less noisy) estimate of your configuration and how to move from that configuration to the boundary of a different skill.

Connection to Retreival-based methods: Retrieval-based methods can be thought of as retreiving certain trajectories from a larger dataset that can increase the local resolution of the manifold. By getting similar trajectories and finetuning on these demonstrations can increase the resolution of the manifold which inturn can allow for more accurate estimate and direction to move along the manifold for completing the tasks. However, one of the mistakes that most of the retrieval-based methods get wrong is that when we retreive trajectories based on similarity metrics, a lot of these trajectories might be redundant. This is mainly because similarity based metrics does not take into account the information gain from individual samples. This hurts generalization capability of the model. One way to overcome this have a notation of information gain when retreiving the trajectories from a target distribution.

Connection to Action-Horizon
Increasing action-horizon seems to increase the success rate in general. I believe that this is the property of certain boundaries being a bit more forgiving to overstepping the boundary or there might be a strong feedback that attracts the nearby the configurations to the boundary (think of it as the dimension near the boundary dropping even further than the intrinsic dimension of the manifold). 

Interesting Problems:

Identifying the intrisic dimension of the manifolds.

Identifying the local complexity of the manifolds.

Application of transducive learning for sampling trajectories for better estimation of the manifold.

Why does increasing the composition in the dataset can help in more compositions?

