# Music and Robotics are more related than you think!

I am huge proponent of fast feedback loops, the faster feedbacks you get the faster you will be able to iterate and converge to a good solution. This is true almost universally, but stands out most astoundingly in the research pace of computer science vs other disciplines of science or engineering. Mostly, because when you are working in the world of bits you not limited by the fixed time for information flow as in the world of atoms. You could parallelize your processes and run experiments at scale, only limited by the hardware and computational efficiency of the algorithm. 

Anyways, I have a similar opinion with robotics, although there are now ways to parallelize the robotics research in simulation and that would surely be the way in the future, we are (were) not there yet. Thus, it made sense to abstract the action space and focus on computer vision and reasoning as their own problem that can be fit to the broader robotics settings. But computer vision and reasoning are not the bottleneck now, we have converged to good representations that are transferable across a wide range of scenarios and intelligence is almost free. Now the last piece of the puzzle lies in effectively bridging the gap between these representions and actions. 

I argue that we can benefit a lot from incorporating the advances from music generation in robotics, partly because the feedback loop is still much faster in the former compared to the latter. So let me lay out some of the similarities and differences between these two fields and hopefully convince you to read more papers in these areas to bridge the gap between them.

(Disclaimer: When I say robotics, I mostly have the perspective of robotic manipulation. There are other sub-areas of research in robotics that are equally important, but my current knowledge limits me from talking about them.)

### The Problems Statement

It makes sense to first layout the problems statement that both of these fields are optimizing for, before trying to connect them.

**Music Generation**:  The problem of music generation is similar to the problem statements of image generation and language modeling, where given a large dataset of samples, you learn a lower dimensional manifold that these samples lie on, from which you can sample efficiently to generate novel samples. To be precise, let the song you have on your mobile be a data point (or sample) on a very high-dimensional space. The dimensions of this space are not straightforward as pixels in images or tokens in language. So you have to think in terms of frequencies, amplitudes, and time-varying patterns that collectively represent the complex audio waveform of a musical piece.

**Action Modelling in Robotics**