# Neural tangent kernel

In each category, papers are sorted chronologically.

[https://www.youtube.com/watch?v=NGon2JyjO6Y]: #
+ [Recent Developments in Over-parametrized Neural Networks, Part II](https://www.youtube.com/watch?v=NGon2JyjO6Y)
    +  YouTube, Simons institute workshop.
    + General overview of a few things (mostly optimization).

## Optimization

[https://arxiv.org/abs/1810.12065]: #
+ [On the Convergence Rate of Training Recurrent Neural Networks](./papers/1810.12065.pdf)
    + 29/10/2018
    + See below

[https://arxiv.org/pdf/1811.03962.pdf]: #
+ [A Convergence Theory for Deep Learning via Over-Parameterization](./papers/1811.03962.pdf)
    + 9/11/2018
    + Simplification of [On the Convergence Rate of Training Recurrent Neural Networks](./papers/1810.12065.pdf).
    + Convergence to global optima whp for GD and SGD.
    + Works for \ell_2, cross entropy and other losses. 
    + Works for fully connected, ResNets, ConvNets, (and RNNs, in the paper above)


[https://arxiv.org/pdf/1811.03804.pdf]: #
+ [Gradient Descent Finds Global Minima of Deep Neural Network.](./papers/1811.03804.pdf)
    + 9/11/2018
    + Du et al 
    + Convergence to global optima whp for GD for \ell_2.
    + Exponential width wrt depth needed in fully connected. Polynomial for resnets.

[https://arxiv.org/pdf/1901.08572.pdf]: #
+ [Width Provably Matters in Optimization for Deep Linear Neural Networks.](./papers/1901.08572.pdf)
    + 01/2019
    + Du et al. 
    + Deep linear neural network
    + Convergence to global minima if low polynomial width is assumed.


### Infinite limit

[https://arxiv.org/pdf/1806.07572.pdf ]: #
+ [Neural Tangent Kernel: Convergence and Generalization in Neural Networks ](./papers/1806.07572.pdf)
    + 06/2018
    + Original paper.

[https://arxiv.org/pdf/1902.06720.pdf]: #
+ [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](./papers/1902.06720.pdf)
    + 02/2019
    + They apparently prove that a finite learning rate is enough for the model to follow NTK dynamics in infinite width limit.
    + Experiments


[https://arxiv.org/pdf/1904.11955.pdf]: #
+ [On Exact Computation with an Infinitely Wide Neural Net](./papers/1904.11955.pdf)
    + 04/2019
    + Shows that NTK work somewhat worse than NNs, but not as much worse as previous work suggested.
    + Claims to show a proof that sounds similar to those of Allen-Zhu, Du etc. but not sure what the difference is.

### Lazy training

[https://arxiv.org/pdf/1812.07956.pdf ]: #
+ [On Lazy Training in Differentiable Programming](./papers/1812.07956.pdf)
    + 12/2018
    + They show that NTK regime can be controlled by rescaling the model, and show (experimentally) that neural nets in practice perform better than those in lazy regime.
    + Also this seems to be independent of width. So scaling the model is a much easier way to get to lazy training, versus the infinite width + infinitesimal learning rate route??

[https://arxiv.org/pdf/1906.08034.pdf]: #
+ [Disentangling feature and lazy learning in deep neural networks: an empirical study](./papers/1906.08034.pdf)
    + 06/2019
    + Similar to above (Chizat et al.), but more experimental.

## Generalization

[https://arxiv.org/pdf/1811.04918.pdf]: #
+ [Learning and Generalization in Overparameterized NeuralNetworks, Going Beyond Two Layers](./papers/1811.04918.pdf)
    + 11/2018
    + Theorems are not based on NTKs, but it has experiments showing how generalization for 3-layer NNs is better than for its corresponding NTK.

[https://arxiv.org/pdf/1902.01384.pdf]: #
+ [Generalization Error Bounds of Gradient Descent for Learning Over-parameterized Deep ReLU Networks](./papers/1902.01384.pdf)
    + 02/2019
    + See below
    
[https://arxiv.org/pdf/1905.13210.pdf]: #
+ [Generalization Bounds of Stochastic Gradient Descent for Wide and Deep Neural Networks](./papers/1905.13210.pdf)
    + 05/2019
    + Seems very similar to the one above. What are the differences? Just that this is SGD vs GD in the above paper?
    + Improves on the Arora2019 paper showing generalization bounds for NTK.
    + I’d be interested in understanding the connection of their bound to classical margin and pac bayes bounds for kernel regression.
    + They don’t show any plots demonstrating how good their bounds are, which probably means they are vacuous though...


[https://arxiv.org/pdf/1905.10337.pdf]: #
+ [What Can ResNet Learn Efficiently, Going Beyond Kernels?](./papers/1905.10337.pdf)
    + 05/2019
    + Shows in the PAC setting that there are ("simple") functions that ResNets learn efficiently and such that any kernel gets test error much greater for the same sample complexity. in particular NTKs too.
    
[https://arxiv.org/pdf/1905.10843.pdf]: #
+ [Asymptotic learning curves of kernel methods: empirical data v.s. Teacher-Student paradigm](./papers/1905.10843.pdf)
    + 05/2019
    + I think that getting learning curves for neural nets is a very interesting challenge.
    + Here they do it for kernels, but if the NN behaves like a kernel, it would be relevant..

[https://arxiv.org/pdf/1906.05392.pdf]: #
+ [Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian](./papers/1906.05392.pdf)
    + 06/2019

## Others

[https://arxiv.org/pdf/1902.04760.pdf]: #
+ [Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation](./papers/1902.04760.pdf)
    + 02/2019
    + Although this paper is really cool in that it shows that most kinds of neural networks become GPs when infinitely wide, w.r.t. NTK, it just shows a proof where the layer widths can go to infinity at the same time, and generalizes it to more architectures, so doesn’t feel like necessarily much new insight?

[https://arxiv.org/pdf/1905.12173.pdf]: #
+ [On the Inductive Bias of Neural Tangent Kernels](./papers/1905.12173.pdf)
    + 05/2019
    + This is just about properties of NTK (so not studying NNs directly).
    + They find that the NTK model has different type of stability to deformations of the input than other NNGPs, and better approximation properties (whatever that means)

[https://arxiv.org/pdf/1906.01930.pdf]: #
+ [Approximate Inference Turns Deep Networks into Gaussian Processes](./papers/1906.01930.pdf)
    + 06/2019
    + Shows Bayesian NNs (of any width) are equivalent to GPs, surprisingly with kernel given by NTK

# ToClassify

[https://arxiv.org/pdf/1905.05095.pdf]: #
+ [Spectral Analysis of Kernel and Neural Embeddings: Optimization and Generalization](./papers/1905.05095.pdf)
    + 05/2019
    + They just study what happens when you use a neural network or a kernel representation for data (fed as input to a NN I guess).

[https://arxiv.org/pdf/1805.00915.pdf]: #
+ [Neural Networks as Interacting Particle Systems: Asymptotic Convexity of the Loss Landscape and Universal Scaling of the Approximation Error ](./papers/1805.00915.pdf)
    + 05/2018
    + Hard to fully judge from abstract, but my experience tells me that a lot of these “analogy-based” works, don’t tend to have much new in them. But maybe I'm wrong.
    
[https://arxiv.org/pdf/1810.09665.pdf]: #
+ [A jamming transition from under- to over-parametrization affects loss landscape and generalization](./papers/1810.09665.pdf)
    + 10/2018
    + Similar to the above. Analogy-based stuff, with nothing that sounds very novel. Sounds like results showing that the Jacobian is non-degenerate (but I bet it’s not looking at SGD in particular, or otherwise making some unrealistic assumption hmm), and then classic (by now lol) peaked overfitting curve.
    + I’d find these ones more intresting (though not about NTK): https://arxiv.org/abs/1605.08361 https://arxiv.org/abs/1710.10928 https://arxiv.org/abs/1803.02968

[https://arxiv.org/pdf/1808.09372.pdf]: #
+ [Mean Field Analysis of Neural Networks: A Central Limit Theorem](./papers/1808.09372.pdf)
    + 08/2018
    + they only look at one hidden layer and squared error loss, so I’m not convinced of the novelty of results?

[https://arxiv.org/pdf/1906.06321.pdf]: #
+ [Provably Efficient $Q$-learning with Function Approximation via Distribution Shift Error Checking Oracle](./papers/1906.06321.pdf)
    + 06/2019
    + Not about NTK, but authors suggest it could be extended to use NTK to analyze NN-based function approximation.

# Some notes

+ NTK depends on initialization.
