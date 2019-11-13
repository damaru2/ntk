This is a list of papers that use the Neural Tangent Kernel (NTK). In each category, papers are sorted chronologically. Some of these papers were presented in the NTK reading group during the summer 2019 at the University of Oxford.

We used [hypothes.is](https://web.hypothes.is/) to some extent, see [this](https://via.hypothes.is/https://arxiv.org/pdf/1806.07572.pdf) for instance. There are notes for a few of the papers, which you can find linked below the relevant papers.

## Schedule
+ 2/08/2019 [[notes](./notes/Neural_Tangent_kernels___Jacot_et_al.pdf)] Neural Tangent Kernel: Convergence and Generalization in Neural Networks.
+ 9/08/2019 [[notes](./notes/du_et_al.pdf)] Gradient Descent Finds Global Minima of Deep Neural Network.
+ 16/08/2019 Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks + insights from Gradient Descent Provably Optimizes Over-parameterized Neural Networks.
+ 23/08/2019 On Lazy Training in Differentiable Programming
+ 13/09/2019 Generalization bounds of stochastic gradient descent for wide and deep networks
+ 18/10/2019 [[notes](./notes/low_rank_jac_thm.pdf)] Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian

# Neural tangent kernel

[https://www.youtube.com/watch?v=NGon2JyjO6Y]: #
+ [Recent Developments in Over-parametrized Neural Networks, Part II](https://www.youtube.com/watch?v=NGon2JyjO6Y)
    + Interesting, nice overview of a few things, mostly related to optimization and NTK
    + YouTube, Simons institute workshop.
    + Part I is interesting, but take into account that it is about other optimization things for NNs, but not about NTK.

## Optimization

### Infinite limit

[https://arxiv.org/pdf/1806.07572.pdf ]: #
+ [Neural Tangent Kernel: Convergence and Generalization in Neural Networks ](./papers/1806.07572.pdf)  -- [link](https://arxiv.org/pdf/1806.07572.pdf)
    + [Notes](./notes/Neural_Tangent_kernels___Jacot_et_al.pdf)
    + 06/2018
    + Original NTK paper.
    + Exposes the idea of the NTK for the first time, although the proof that the Kernel in the limit is deterministic is done tending the number of neurons of each layer to infinity, layer by layer sequentially.
    + It proves positive definiteness of the kernel for certain regimes, thus proving you can optimize to reach a global minimum at a linear rate.

[https://arxiv.org/pdf/1902.06720.pdf]: #
+ [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](./papers/1902.06720.pdf)  -- [link](https://arxiv.org/pdf/1902.06720.pdf)
    + 02/2019
    + They apparently prove that a finite learning rate is enough for the model to follow NTK dynamics in infinite width limit.
    + Experiments


[https://arxiv.org/pdf/1904.11955.pdf]: #
+ [On Exact Computation with an Infinitely Wide Neural Net](./papers/1904.11955.pdf)  -- [link](https://arxiv.org/pdf/1904.11955.pdf)
    + 04/2019
    + Shows that NTK work somewhat worse than NNs, but not as much worse as previous work suggested.
    + Claims to show a proof that sounds similar to those of Allen-Zhu, Du etc. but not sure what the difference is.


### Finite results

[https://arxiv.org/abs/1810.02054]: #
+ [Gradient Descent Provably Optimizes Over-parameterized Neural Networks](./papers/1810.02054.pdf)  -- [link](https://arxiv.org/abs/1810.02054)
    + 04/10/2018
    + A preliminar result of Gradient Descent Finds Global Minima of Deep Neural Network (below) but only for two layer neural networks.

[https://arxiv.org/abs/1810.12065]: #
+ [On the Convergence Rate of Training Recurrent Neural Networks](./papers/1810.12065.pdf)  -- [link](https://arxiv.org/abs/1810.1206)
    + 29/10/2018
    + See below

[https://arxiv.org/pdf/1811.03962.pdf]: #
+ [A Convergence Theory for Deep Learning via Over-Parameterization](./papers/1811.03962.pdf)  -- [link](https://arxiv.org/pdf/1811.03962.pdf)
    + 9/11/2018
    + Simplification of [On the Convergence Rate of Training Recurrent Neural Networks](./papers/1810.12065.pdf).
    + Convergence to global optima whp for GD and SGD.
    + Works for \ell_2, cross entropy and other losses. 
    + Works for fully connected, ResNets, ConvNets, (and RNNs, in the paper above)


[https://arxiv.org/pdf/1811.03804.pdf]: #
+ [Gradient Descent Finds Global Minima of Deep Neural Network.](./papers/1811.03804.pdf)  -- [link](https://arxiv.org/pdf/1811.03804.pdf)
    + [Notes](./notes/du_et_al.pdf)
    + 9/11/2018
    + Du et al 
    + Convergence to global optima whp for GD for \ell_2.
    + Exponential width wrt depth needed in fully connected. Polynomial for resnets.

[https://arxiv.org/pdf/1901.08572.pdf]: #
+ [Width Provably Matters in Optimization for Deep Linear Neural Networks](./papers/1901.08572.pdf)  -- [link](https://arxiv.org/pdf/1901.08572.pdf)
    + 12/2019
    + Du et al. 
    + Deep linear neural network
    + Convergence to global minima if low polynomial width is assumed.

[https://arxiv.org/pdf/1811.08888.pdf]: #
+ [Stochastic Gradient Descent Optimizes Over-parameterized Deep ReLU Networks](./papers/1811.08888.pdf)  -- [link](https://arxiv.org/pdf/1811.08888.pdf)
    + 21/11/2018

[https://arxiv.org/pdf/1812.10004.pdf]: #
+ [Overparameterized Nonlinear Learning: Gradient Descent Takes the Shortest Path?](./papers/1812.10004.pdf)  -- [link](https://arxiv.org/pdf/1812.10004.pdf)
    + 25/11/2018
    + Results for one hidden layer NNs, generalized linear models and low-rank matrix regression.

[https://arxiv.org/abs/1905.13654.pdf]: #
+ [Training Dynamics of Deep Networks using Stochastic Gradient Descent via Neural Tangent Kernel](./papers/1905.13654.pdf)  -- [link](https://arxiv.org/abs/1905.13654.pdf)
    + 06/2019
    + SGD analyzed from the point of view of Stochastic Differential Equations


### Lazy training

[https://arxiv.org/pdf/1812.07956.pdf ]: #
+ [On Lazy Training in Differentiable Programming](./papers/1812.07956.pdf)  -- [link](https://arxiv.org/pdf/1812.07956.pdf)
    + 12/2018
    + They show that NTK regime can be controlled by rescaling the model, and show (experimentally) that neural nets in practice perform better than those in lazy regime.
    + Also this seems to be independent of width. So scaling the model is a much easier way to get to lazy training, versus the infinite width + infinitesimal learning rate route??

[https://arxiv.org/pdf/1906.08034.pdf]: #
+ [Disentangling feature and lazy learning in deep neural networks: an empirical study](./papers/1906.08034.pdf)  -- [link](https://arxiv.org/pdf/1906.08034.pdf)
    + 06/2019
    + Similar to above (Chizat et al.), but more experimental.
    
[https://arxiv.org/pdf/1906.05827.pdf]: #
+ [Kernel and deep regimes in overparametrized models](./papers/1906.05827.pdf)  -- [link](https://arxiv.org/pdf/1906.05827.pdf)
    + 06/2019
    + Large initialization leads to kernel/lazy regime
    + Small initialization leads to deep/active/adaptive regime, which can sometimes lead to better generalization. They claim this is the regime that allows one to "exploit the power of depth", and thus is key to understanding deep learning.
    + The systems they analyze in detail are rather simple (like matrix completion) or artificial (like a very ad-hoc type of neural network)

## Generalization

[https://arxiv.org/pdf/1811.04918.pdf]: #
+ [Learning and Generalization in Overparameterized NeuralNetworks, Going Beyond Two Layers](./papers/1811.04918.pdf)  -- [link](https://arxiv.org/pdf/1811.04918.pdf)
    + 11/2018
    + Theorems are not based on NTKs, but it has experiments showing how generalization for 3-layer NNs is better than for its corresponding NTK.

[https://arxiv.org/pdf/1901.08584.pdf]: #
+ [Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks](./papers/1901.08584.pdf)  -- [link](https://arxiv.org/pdf/1901.08584.pdf)
    + 01/2019
    + Arora et al
    + "Our work is related to kernel methods, especially recent discoveries of the connection between deep
learning and kernels (Jacot et al., 2018; Chizat & Bach, 2018b;...) Our analysis utilized several properties of a related kernel from the ReLU activation."

[https://arxiv.org/pdf/1902.01384.pdf]: #
+ [Generalization Error Bounds of Gradient Descent for Learning Over-parameterized Deep ReLU Networks](./papers/1902.01384.pdf)  -- [link](https://arxiv.org/pdf/1902.01384.pdf)
    + 02/2019
    + See below
    
[https://arxiv.org/pdf/1905.13210.pdf]: #
+ [Generalization Bounds of Stochastic Gradient Descent for Wide and Deep Neural Networks](./papers/1905.13210.pdf)  -- [link](https://arxiv.org/pdf/1905.13210.pdf)
    + 05/2019
    + Seems very similar to the one above. What are the differences? Just that this is SGD vs GD in the above paper?
    + Improves on the Arora2019 paper showing generalization bounds for NTK.
    + I’d be interested in understanding the connection of their bound to classical margin and pac bayes bounds for kernel regression.
    + They don’t show any plots demonstrating how good their bounds are, which probably means they are vacuous though...


[https://arxiv.org/pdf/1905.10337.pdf]: #
+ [What Can ResNet Learn Efficiently, Going Beyond Kernels?](./papers/1905.10337.pdf)  -- [link](https://arxiv.org/pdf/1905.10337.pdf)
    + 05/2019
    + Shows in the PAC setting that there are ("simple") functions that ResNets learn efficiently and such that any kernel gets test error much greater for the same sample complexity. in particular NTKs too.
    
[https://arxiv.org/pdf/1905.10843.pdf]: #
+ [Asymptotic learning curves of kernel methods: empirical data v.s. Teacher-Student paradigm](./papers/1905.10843.pdf)  -- [link](https://arxiv.org/pdf/1905.10843.pdf)
    + 05/2019
    + I think that getting learning curves for neural nets is a very interesting challenge.
    + Here they do it for kernels, but if the NN behaves like a kernel, it would be relevant..

[https://arxiv.org/pdf/1906.05392.pdf]: #
+ [Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian](./papers/1906.05392.pdf)  -- [link](https://arxiv.org/pdf/1906.05392.pdf)
    + 06/2019
    + [Notes](./notes/low_rank_jac_thm.pdf)
    + Uses NTK mainly and splits the eigenspace into two (based on a cutoff value of the eigenvalues). Projection of residuals onto the top eigenspace trains very fast and the rest could not train at all and loss could increase. Trade off based on cutoff value.
    + Two layers.
    + \ell\_2 loss.

## Others

[https://arxiv.org/pdf/1902.04760.pdf]: #
+ [Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation](./papers/1902.04760.pdf)  -- [link](https://arxiv.org/pdf/1902.04760.pdf)
    + 02/2019
    + Although this paper is really cool in that it shows that most kinds of neural networks become GPs when infinitely wide, w.r.t. NTK, it just shows a proof where the layer widths can go to infinity at the same time, and generalizes it to more architectures, so doesn’t feel like necessarily much new insight?

[https://arxiv.org/pdf/1905.12173.pdf]: #
+ [On the Inductive Bias of Neural Tangent Kernels](./papers/1905.12173.pdf)  -- [link](https://arxiv.org/pdf/1905.12173.pdf)
    + 05/2019
    + This is just about properties of NTK (so not studying NNs directly).
    + They find that the NTK model has different type of stability to deformations of the input than other NNGPs, and better approximation properties (whatever that means)

[https://arxiv.org/pdf/1906.01930.pdf]: #
+ [Approximate Inference Turns Deep Networks into Gaussian Processes](./papers/1906.01930.pdf)  -- [link](https://arxiv.org/pdf/1906.01930.pd)
    + 06/2019
    + Shows Bayesian NNs (of any width) are equivalent to GPs, surprisingly with kernel given by NTK

# ToClassify

[https://arxiv.org/pdf/1905.05095.pdf]: #
+ [Spectral Analysis of Kernel and Neural Embeddings: Optimization and Generalization](./papers/1905.05095.pdf)  -- [link](https://arxiv.org/pdf/1905.05095.pdf)
    + 05/2019
    + They just study what happens when you use a neural network or a kernel representation for data (fed as input to a NN I guess).

[https://arxiv.org/pdf/1808.09372.pdf]: #
+ [Mean Field Analysis of Neural Networks: A Central Limit Theorem](./papers/1808.09372.pdf)  -- [link](https://arxiv.org/pdf/1808.09372.pdf)
    + 08/2018
    + they only look at one hidden layer and squared error loss, so I’m not convinced of the novelty of results?

[https://arxiv.org/pdf/1906.06321.pdf]: #
+ [Provably Efficient $Q$-learning with Function Approximation via Distribution Shift Error Checking Oracle](./papers/1906.06321.pdf)  -- [link](https://arxiv.org/pdf/1906.06321.pdf)
    + 06/2019
    + Not about NTK, but authors suggest it could be extended to use NTK to analyze NN-based function approximation.

[https://arxiv.org/pdf/1911.00809.pdf] #
+ [Enhanced Convolutional Neural Tangent Kernels](./papers/1911.00809.pdf)  -- [link](https://arxiv.org/pdf/1911.00809.pdf)
    + 11/2019
    + Enhances the NTK for convolutional networks of "On Exact Computation..." by adding some implicit data augmentation to the kernel that encodes some kind of local translation invariance and horizontal flipping.
    + They have experiments that show good empirical performance, in particular they get 89% accuracy for CIFAR-10, matching AlexNet. This is the first time a kernel gets this results.


# Some notes

+ NTK depends on initialization.

