# Generative Modeling via Drifting

**Mingyang Deng**¹, **He Li**¹, **Tianhong Li**¹, **Yilun Du**², **Kaiming He**¹

¹MIT ²Harvard University

Project page: [lambertae.github.io/projects/drifting](https://lambertae.github.io/projects/drifting)

---

**Figure 1: Drifting Model.**
A network $f$ performs a pushforward operation: $q = f_\# p_{\text{prior}}$, mapping a prior distribution $p_{\text{prior}}$ (e.g., Gaussian, not shown here) to a pushforward distribution $q$ (orange). The goal of training is to approximate the data distribution $p_{\text{data}}$ (blue). As training iterates, we obtain a sequence of models $\{f_i\}$, which corresponds to a sequence of pushforward distributions $\{q_i\}$. Our Drifting Model focuses on the evolution of this pushforward distribution at _training-time_. We introduce a drifting field (detailed in main text) that approaches zero when $q$ matches $p_{\text{data}}$. This drifting field provides a loss function (y-axis, in log-scale) for training.

---

## Abstract

Generative modeling can be formulated as learning a mapping $f$ such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, e.g., in diffusion/flow-based models. In this paper, we propose a new paradigm called _Drifting Models_, which evolve the pushforward distribution during training and naturally admit one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. In experiments, our one-step generator achieves state-of-the-art results on ImageNet 256×256, with FID 1.54 in latent space and 1.61 in pixel space. We hope that our work opens up new opportunities for high-quality one-step generation.

---

## 1. Introduction

Generative models are commonly regarded as more challenging than discriminative models. While discriminative modeling typically focuses on mapping individual samples to their corresponding labels, generative modeling concerns mapping from one distribution to another. This can be expressed as learning a mapping $f$ such that the _pushforward_ of a prior distribution $p_\text{prior}$ matches the data distribution, namely, $f_\# p_\text{prior} \approx p_{\text{data}}$. Conceptually, generative modeling learns a _functional_ (here, $f_\#$) that maps from one function (here, a distribution) to another.

The "pushforward" behavior can be realized _iteratively_ at _inference_ time, e.g., in prevailing paradigms such as Diffusion and Flow Matching. When generating, these models map noisier samples to slightly cleaner ones, progressively evolving the sample distribution toward the data distribution. This modeling philosophy can be viewed as decomposing a complex pushforward map (i.e., $f_\#$) into a chain of more feasible transformations, applied at inference time.

In this paper, we propose _Drifting Models_, a new paradigm for generative modeling. Drifting Models are characterized by learning a pushforward map that evolves during _training_ time, thereby removing the need for an iterative inference procedure. The mapping $f$ is represented by a single-pass, non-iterative network. As the training process is inherently iterative in deep learning optimization, it can be naturally viewed as evolving the pushforward distribution, $f_\# p_\text{prior}$, through the update of $f$. See Figure 1.

To drive the evolution of the training-time pushforward, we introduce a _drifting field_ that governs the sample movement. This field depends on the generated distribution and the data distribution. By definition, this field becomes zero when the two distributions match, thereby reaching an equilibrium in which the samples no longer drift.

Building on this formulation, we propose a simple training objective that minimizes the _drift_ of the generated samples. This objective induces sample movements and thereby evolves the underlying pushforward distribution through iterative optimization (e.g., SGD). We further introduce the designs of the drifting field, the neural network model, and the training algorithm.

Drifting Models naturally perform _single-step_ ("1-NFE") generation and achieve strong empirical performance. On ImageNet 256×256, we obtain a 1-NFE FID of **1.54** under the standard latent-space generation protocol, achieving a new state-of-the-art among single-step methods. This result remains competitive even when compared with _multi-step_ diffusion-/flow-based models. Further, under the more challenging _pixel_-space generation protocol (i.e., without latents), we reach a 1-NFE FID of **1.61**, substantially outperforming previous pixel-space methods. These results suggest that Drifting Models offer a promising new paradigm for high-quality, efficient generative modeling.

---

## 2. Related Work

**Diffusion-/Flow-based Models.**
Diffusion models (e.g., Sohl-Dickstein et al., Ho et al., Song et al.) and their flow-based counterparts (e.g., Lipman et al., Liu et al., Albergo et al.) formulate noise-to-data mappings through _differential equations_ (SDEs or ODEs). At the core of their inference-time computation is an _iterative_ update, e.g., of the form $\mathbf{x}_{i+1} = \mathbf{x}_i + \Delta\mathbf{x}_i$, such as with an Euler solver. The update $\Delta\mathbf{x}_i$ depends on the neural network $f$, and as a result, generation involves multiple steps of network evaluations.

A growing body of work has focused on reducing the steps of diffusion-/flow-based models. Distillation-based methods (e.g., Salimans et al., Luo et al., Yin et al., Zhou et al.) distill a pretrained multi-step model into a single-step one. Another line of research aims to train one-step diffusion/flow models from scratch (e.g., Song et al., Frans et al., Boffi et al., Geng et al.). To achieve this goal, these methods incorporate the SDE/ODE dynamics into training by approximating the induced trajectories. In contrast, our work presents a conceptually different paradigm and does not rely on SDE/ODE formulations as in diffusion/flow models.

**Generative Adversarial Networks (GANs).**
GANs (Goodfellow et al., 2014) are a classical family of models that train a generator by discriminating generated samples from real data. Like GANs, our method involves a single-pass network $f$ that maps noise to data, whose "goodness" is evaluated by a loss function; however, unlike GANs, our method does not rely on adversarial optimization.

**Variational Autoencoders (VAEs).**
VAEs (Kingma & Welling, 2013) optimize the evidence lower bound (ELBO), which consists of a reconstruction loss and a KL divergence term. Classical VAEs are one-step generators when using a Gaussian prior. Today's prevailing VAE applications often resort to priors learned from other methods, e.g., diffusion or autoregressive models, where VAEs effectively act as tokenizers.

**Normalizing Flows (NFs).**
NFs (Rezende & Mohamed, Dinh et al., Zhai et al.) learn mappings from data to noise and optimize the log-likelihood of samples. These methods require invertible architectures and computable Jacobians. Conceptually, NFs operate as one-step generators at inference, with computation performed by the inverse of the network.

**Moment Matching.**
Moment-matching methods (Dziugaite et al., Li et al.) seek to minimize the Maximum Mean Discrepancy (MMD) between the generated and data distributions. Moment Matching has recently been extended to one-/few-step diffusion (Zhou et al., 2025). Related to MMD, our approach also leverages the concepts of kernel functions and positive/negative samples. However, our approach focuses on a drifting field that explicitly governs the sample drifts at training time. Further discussion is in Appendix A.6.

**Contrastive Learning.**
Our drifting field is driven by positive samples from the data distribution and negative samples from the generated distribution. This is conceptually related to the positive and negative samples in _contrastive representation learning_ (Hadsell et al., Oord et al.). The idea of contrastive learning has also been extended to generative models, e.g., to GANs or Flow Matching.

---

## 3. Drifting Models for Generation

We propose Drifting Models, which formulate generative modeling as a _training-time_ evolution of the pushforward distribution via a drifting field. Our model naturally performs one-step generation at inference time.

### 3.1 Pushforward at Training Time

Consider a neural network $f: \mathbb{R}^C \mapsto \mathbb{R}^D$. The input of $f$ is $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$ (e.g., any noise of dimension $C$), and the output is denoted by $\mathbf{x} = f(\boldsymbol{\epsilon}) \in \mathbb{R}^D$. In general, the input and output dimensions need not be equal.

We denote the distribution of the network output by $q$, i.e., $\mathbf{x} = f(\boldsymbol{\epsilon}) \sim q$. In probability theory, $q$ is referred to as the _pushforward_ distribution of $p_{\boldsymbol{\epsilon}}$ under $f$, denoted by:

$$q = f_{\#} p_{\boldsymbol{\epsilon}}.$$

Here, "$f_{\#}$" denotes the pushforward induced by $f$. Intuitively, this notation means that $f$ transforms a distribution $p_{\boldsymbol{\epsilon}}$ into another distribution $q$. The goal of generative modeling is to find $f$ such that $f_{\#} p_{\boldsymbol{\epsilon}} \approx p_\text{data}$.

Since neural network _training_ is inherently iterative (e.g., SGD), the training process produces a sequence of models $\{f_i\}$, where $i$ denotes the training iteration. This corresponds to a sequence of pushforward distributions $\{q_i\}$ during training, where $q_i = [f_i]_{\#} p_{\boldsymbol{\epsilon}}$ for each $i$. The training process progressively evolves $q_i$ to match $p_{\text{data}}$.

When the network $f$ is updated, a sample at training iteration $i$ is implicitly "drifted" as: $\mathbf{x}_{i+1} = \mathbf{x}_i + \Delta\mathbf{x}_i$, where $\Delta\mathbf{x}_i := f_{i+1}(\boldsymbol{\epsilon}) - f_i(\boldsymbol{\epsilon})$ arises from parameter updates to $f$. This implies that the update of $f$ determines the "residual" of $\mathbf{x}$, which we refer to as the "drift".

### 3.2 Drifting Field for Training

Next, we define a **drifting field** to govern the training-time evolution of the samples $\mathbf{x}$ and, consequently, the pushforward distribution $q$. A drifting field is a function that computes $\Delta\mathbf{x}$ given $\mathbf{x}$. Formally, denoting this field by $\mathbf{V}_{p,q}(\cdot) \colon \mathbb{R}^d \to \mathbb{R}^d$, we have:

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \mathbf{V}_{p,q_i}(\mathbf{x}_i),$$

Here, $\mathbf{x}_i = f_i(\boldsymbol{\epsilon}) \sim q_i$ and after drifting we denote $\mathbf{x}_{i+1} \sim q_{i+1}$. The subscripts $p, q$ denote that this field depends on $p$ (e.g., $p = p_\text{data}$) and the current distribution $q$.

Ideally, when $p = q$, we want all $\mathbf{x}$ to stop drifting, i.e., $\mathbf{V} = \mathbf{0}$. In this paper, we consider the following proposition:

**Proposition.** Consider an **anti-symmetric** drifting field:

$$\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x}), \quad \forall \mathbf{x}.$$

Then we have: $q = p \Rightarrow \mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0}, \forall \mathbf{x}$.

The proof is straightforward.[^1] Intuitively, anti-symmetry means that swapping $p$ and $q$ simply flips the sign of the drift. This proposition implies that if the pushforward distribution $q$ matches the data distribution $p$, the drift is zero for any sample and the model achieves an equilibrium.

[^1]: $q = p \Rightarrow \mathbf{V}_{p,q} = \mathbf{V}_{q,p} = -\mathbf{V}_{p,q} \Rightarrow \mathbf{V}_{p,q} = \mathbf{0}$

We note that the converse implication, i.e., $\mathbf{V}_{p,q} = \mathbf{0} \Rightarrow q = p$, is false in general for arbitrary choices of $\mathbf{V}$. For our kernelized formulation (Sec. 3.3), we give sufficient conditions under which $\mathbf{V}_{p,q} \approx \mathbf{0}$ implies $q \approx p$ (Appendix A.5).

**Training Objective.**
The property of equilibrium motivates a definition of a training objective. Let $f_\theta$ be a network parameterized by $\theta$, and $\mathbf{x} = f_\theta(\boldsymbol{\epsilon})$ for $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$. At the equilibrium where $\mathbf{V} = \mathbf{0}$, we set up the following _fixed-point_ relation:

$$f_{\hat{\theta}}(\boldsymbol{\epsilon}) = f_{\hat{\theta}}(\boldsymbol{\epsilon}) + \mathbf{V}_{p, q_{\hat{\theta}}}\!\big(f_{\hat{\theta}}(\boldsymbol{\epsilon})\big).$$

Here, $\hat{\theta}$ denotes the optimal parameters that can achieve the equilibrium, and $q_{\hat{\theta}}$ denotes the pushforward of $f_{\hat{\theta}}$.

This equation motivates a fixed-point iteration during training. At iteration $i$, we seek to satisfy:

$$f_{\theta_{i+1}}(\boldsymbol{\epsilon}) \leftarrow f_{\theta_i}(\boldsymbol{\epsilon}) + \mathbf{V}_{p, q_{\theta_i}}\!\big(f_{\theta_i}(\boldsymbol{\epsilon})\big).$$

We convert this update rule into a loss function:

$$\mathcal{L} = \mathbb{E}_{\boldsymbol{\epsilon}}\Big[\big\|\underbrace{f_{\theta}(\boldsymbol{\epsilon})}_{\text{prediction}} - \underbrace{\texttt{stopgrad}\big(f_{\theta}(\boldsymbol{\epsilon}) + \mathbf{V}_{p,q_\theta}\!\big(f_\theta(\boldsymbol{\epsilon})\big)\big)}_{\text{frozen target}}\big\|^2\Big]. \tag{1}$$

Here, the stop-gradient operation provides a frozen state from the last iteration, following Chen et al. (2021) and Song et al. (2023). Intuitively, we compute a frozen target and move the network prediction toward it.

We note that the _value_ of our loss function $\mathcal{L}$ is equal to $\mathbb{E}_{\boldsymbol{\epsilon}}\big[\|\mathbf{V}(f(\boldsymbol{\epsilon}))\|^2\big]$, that is, the squared norm of the drifting field $\mathbf{V}$. With the stop-gradient formulation, our solver does not directly back-propagate through $\mathbf{V}$, because $\mathbf{V}$ depends on $q_\theta$ and back-propagating through a distribution is nontrivial. Instead, our formulation minimizes this objective _indirectly_: it moves $\mathbf{x} = f_\theta(\boldsymbol{\epsilon})$ towards its drifted version, i.e., towards $\mathbf{x} + \Delta\mathbf{x}$ that is frozen at this iteration.

### 3.3 Designing the Drifting Field

The field $\mathbf{V}_{p,q}$ depends on two distributions $p$ and $q$. To obtain a computable formulation, we consider the form:

$$\mathbf{V}_{p,q}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}^+ \sim p}\, \mathbb{E}_{\mathbf{y}^- \sim q} [\mathcal{K}(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-)],$$

where $\mathcal{K}(\cdot, \cdot, \cdot)$ is a kernel-like function describing interactions among three sample points. $\mathcal{K}$ can optionally depend on $p$ and $q$. Our framework supports a broad class of functions $\mathcal{K}$, as long as $\mathbf{V} = \mathbf{0}$ when $p = q$.

For the instantiation in this work, we introduce a form of $\mathbf{V}$ driven by attraction and repulsion. We define the following fields inspired by the _mean-shift_ method (Cheng, 1995):

$$\mathbf{V}^+_p(\mathbf{x}) := \frac{1}{Z_p} \mathbb{E}_p\!\left[k(\mathbf{x}, \mathbf{y}^+)(\mathbf{y}^+ - \mathbf{x})\right], \tag{2}$$

$$\mathbf{V}^-_q(\mathbf{x}) := \frac{1}{Z_q} \mathbb{E}_q\!\left[k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^- - \mathbf{x})\right], \tag{3}$$

Here, $Z_p$ and $Z_q$ are normalization factors:

$$Z_p(\mathbf{x}) := \mathbb{E}_p[k(\mathbf{x}, \mathbf{y}^+)], \quad Z_q(\mathbf{x}) := \mathbb{E}_q[k(\mathbf{x}, \mathbf{y}^-)].$$

Intuitively, the above equations compute the weighted mean of the vector difference $\mathbf{y} - \mathbf{x}$. The weights are given by a kernel $k(\cdot, \cdot)$ normalized accordingly. We then define $\mathbf{V}$ as:

$$\mathbf{V}_{p,q}(\mathbf{x}) := \mathbf{V}^+_p(\mathbf{x}) - \mathbf{V}^-_q(\mathbf{x}).$$

Intuitively, this field can be viewed as attracting by the data distribution $p$ and repulsing by the sample distribution $q$. This is illustrated in Figure 2.

Substituting, we obtain:

$$\mathbf{V}_{p,q}(\mathbf{x}) = \frac{1}{Z_p Z_q} \mathbb{E}_{p,q}\!\left[k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^+ - \mathbf{y}^-)\right]. \tag{4}$$

Here, the vector difference reduces to $\mathbf{y}^+ - \mathbf{y}^-$; the weight is computed from two kernels and normalized jointly. This form is an instantiation of the general framework. It is easy to see that $\mathbf{V}$ is anti-symmetric: $\mathbf{V}_{p,q} = -\mathbf{V}_{q,p}$.

**Figure 2: Illustration of drifting a sample.**
A generated sample $\mathbf{x}$ (black) drifts according to a vector: $\mathbf{V} = \mathbf{V}^+_p - \mathbf{V}^-_q$. Here, $\mathbf{V}^+_p$ is the mean-shift vector of the positive samples (blue) and $\mathbf{V}^-_q$ is the mean-shift vector of the negative samples (orange). $\mathbf{x}$ is attracted by $\mathbf{V}^+_p$ and repulsed by $\mathbf{V}^-_q$.

**Kernel.**
The kernel $k(\cdot, \cdot)$ can be a function that measures the similarity. In this paper, we adopt:

$$k(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\frac{1}{\tau}\|\mathbf{x} - \mathbf{y}\|\right), \tag{5}$$

where $\tau$ is a temperature and $\|\cdot\|$ is $\ell_2$-distance. We view $\tilde{k}(\mathbf{x}, \mathbf{y}) \triangleq \frac{1}{Z} k(\mathbf{x}, \mathbf{y})$ as a normalized kernel, which absorbs the normalization. In practice, we implement $\tilde{k}$ using a _softmax_ operation, with logits given by $-\frac{1}{\tau}\|\mathbf{x} - \mathbf{y}\|$, where the softmax is taken over $\mathbf{y}$. This softmax operation is similar to that of InfoNCE (Oord et al., 2018) in contrastive learning. In our implementation, we further apply an extra softmax normalization over the set of $\{\mathbf{x}\}$ within a batch, which slightly improves performance in practice. This additional normalization does not alter the antisymmetric property of the resulting $\mathbf{V}$.

**Equilibrium and Matched Distributions.**
Since our training loss encourages minimizing $\|\mathbf{V}\|^2$, we hope that $\mathbf{V} \approx \mathbf{0}$ leads to $q \approx p$. While this implication does not hold for arbitrary choices of $\mathbf{V}$, we empirically observe that decreasing the value of $\|\mathbf{V}\|^2$ correlates with improved generation quality. In Appendix A.5, we provide an identifiability heuristic: for our kernelized construction, the zero-drift condition imposes a large set of bilinear constraints on $(p, q)$, and under mild non-degeneracy assumptions this forces $p$ and $q$ to match (approximately).

**Algorithm 1: Training Loss.**
Note: for brevity, here the negative samples `y_neg` are from the same batch of generated data, though they can include other source of negatives.

```python
# f: generator
# y_pos: [N_pos, D], data samples

e = randn([N, C])  # noise
x = f(e)           # [N, D], generated samples
y_neg = x          # reuse x as negatives

V = compute_V(x, y_pos, y_neg)
x_drifted = stopgrad(x + V)

loss = mse_loss(x - x_drifted)
```

**Stochastic Training.**
In stochastic training (e.g., mini-batch optimization), we estimate $\mathbf{V}$ by approximating the expectations with empirical means. For each training step, we draw $N$ samples of noise $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$ and compute a batch of $\mathbf{x} = f_\theta(\boldsymbol{\epsilon}) \sim q$. The generated samples also serve as the negative samples in the same batch, i.e., $\mathbf{y}^- \sim q$. On the other hand, we sample $N_\text{pos}$ data points $\mathbf{y}^+ \sim p_\text{data}$. The drifting field $\mathbf{V}$ is computed in this batch of positive and negative samples.

### 3.4 Drifting in Feature Space

Thus far, we have defined the objective directly in the raw data space. Our formulation can be extended to any feature space. Let $\phi$ denote a feature extractor (e.g., an image encoder) operating on real or generated samples. We rewrite the loss in the feature space as:

$$\mathbb{E}\!\left[\left\|\phi(\mathbf{x}) - \texttt{stopgrad}\!\Big(\phi(\mathbf{x}) + \mathbf{V}\big(\phi(\mathbf{x})\big)\Big)\right\|^2\right]. \tag{6}$$

Here, $\mathbf{x} = f_\theta(\boldsymbol{\epsilon})$ is the output (e.g., images) of the generator. $\mathbf{V}$ is defined in the feature space: in practice, this means that $\phi(\mathbf{y}^+)$ and $\phi(\mathbf{y}^-)$ serve as the positive/negative samples. It is worth noting that feature encoding is a training-time operation and is not used at inference time.

This can be further extended to multiple features, e.g., at multiple scales and locations:

$$\sum_j \mathbb{E}\!\left[\left\|\phi_j(\mathbf{x}) - \texttt{stopgrad}\!\Big(\phi_j(\mathbf{x}) + \mathbf{V}\big(\phi_j(\mathbf{x})\big)\Big)\right\|^2\right]. \tag{7}$$

Here, $\phi_j$ represents the feature vectors at the $j$-th scale and/or location from an encoder $\phi$. With a ResNet-style image encoder, we compute drifting losses across multiple scales and locations, which provides richer gradient information for training.

The feature extractor plays an important role in the generation of high-dimensional data. As our method is based on the kernel $k(\cdot, \cdot)$ for characterizing sample similarities, it is desired for semantically similar samples to stay close in the feature space. This goal is aligned with self-supervised learning (e.g., He et al., Chen et al.). We use pre-trained self-supervised models as the feature extractor.

**Relation to Perceptual Loss.**
Our feature-space loss is related to perceptual loss (Zhang et al., 2018) but is conceptually different. The perceptual loss minimizes: $\|\phi(\mathbf{x}) - \phi(\mathbf{x}_\text{target})\|_2^2$, that is, the regression target is $\phi(\mathbf{x}_\text{target})$ and requires pairing $\mathbf{x}$ with its target. In contrast, our regression target is $\phi(\mathbf{x}) + \mathbf{V}\big(\phi(\mathbf{x})\big)$, where the drifting is in the feature space and requires no pairing. In principle, our feature-space loss aims to match the pushforward distributions $\phi_\# q$ and $\phi_\# p$.

**Relation to Latent Generation.**
Our feature-space loss is _orthogonal_ to the concept of generators in the latent space (e.g., Latent Diffusion). In our case, when using $\phi$, the generator $f$ can still produce outputs in the pixel space or the latent space of a tokenizer. If the generator $f$ is in the latent space and the feature extractor $\phi$ is in the pixel space, the tokenizer decoder is applied before extracting features from $\phi$.

### 3.5 Classifier-Free Guidance

Classifier-free guidance (CFG) (Ho & Salimans, 2022) improves generation quality by extrapolating between class-conditional and unconditional distributions. Our method naturally supports a related form of guidance.

In our model, given a class label $c$ as the condition, the underlying target distribution $p$ now becomes $p_{\text{data}}(\cdot | c)$, from which we can draw positive samples: $\mathbf{y}^+ \sim p_{\text{data}}(\cdot | c)$. To achieve guidance, we draw negative samples either from generated samples or _real_ samples from different classes. Formally, the negative sample distribution is now:

$$\tilde{q}(\cdot | c) \triangleq (1 - \gamma)\, q_\theta(\cdot | c) + \gamma\, p_{\text{data}}(\cdot | \varnothing). \tag{8}$$

Here, $\gamma \in [0, 1)$ is a mixing rate, and $p_{\text{data}}(\cdot | \varnothing)$ denotes the unconditional data distribution.[^2]

[^2]: This should be the data distribution excluding the class $c$. For simplicity, we use the unconditional data distribution.

The goal of learning is to find $\tilde{q}(\cdot | c) = p_\text{data}(\cdot | c)$. Substituting, we obtain:

$$q_\theta(\cdot | c) = \alpha\, p_{\text{data}}(\cdot | c) - (\alpha - 1)\, p_{\text{data}}(\cdot | \varnothing), \tag{9}$$

where $\alpha = \frac{1}{1-\gamma} \geq 1$. This implies that $q_\theta(\cdot | c)$ is to approximate a linear combination of conditional and unconditional data distributions. This follows the spirit of original CFG.

In practice, Eq. (8) means that we sample extra negative examples from the data in $p_{\text{data}}(\cdot | \varnothing)$, in addition to the generated data. We note that, in our method, CFG is a _training-time_ behavior by design: the one-step (1-NFE) property is preserved at inference time.

---

## 4. Implementation for Image Generation

We describe our implementation for image generation on ImageNet at resolution 256×256. Full implementation details are provided in Appendix A.1.

**Tokenizer.** By default, we perform generation in latent space. We adopt the standard SD-VAE tokenizer, which produces a 32×32×4 latent space in which generation is performed.

**Architecture.** Our generator ($f_\theta$) has a DiT-like architecture (Peebles & Xie, 2023). Its input is 32×32×4-dim Gaussian noise $\boldsymbol{\epsilon}$, and its output is the generated latent $\mathbf{x}$ of the same dimension. We use a patch size of 2, i.e., like DiT/2. Our model uses adaLN-zero for processing class-conditioning or other extra conditioning.

**CFG conditioning.** We follow Geng et al. (2025) and adopt CFG-conditioning. At training time, a CFG scale $\alpha$ is randomly sampled. Negative samples are prepared based on $\alpha$, and the network is conditioned on this value. At inference time, $\alpha$ can be freely specified and varied without retraining.

**Batching.** When class labels are involved, we sample a batch of $N_c$ class labels. For each label, we perform Algorithm 1 _independently_. Accordingly, the _effective_ batch size is $B = N_c \times N$, which consists of $N_c \times N$ negatives and $N_c \times N_\text{pos}$ positives.

We define a "training epoch" based on the number of generated samples $\mathbf{x}$. In particular, each iteration generates $B$ samples, and one epoch corresponds to $N_\text{data}/B$ iterations for a dataset of size $N_\text{data}$.

**Feature Extractor.** Our model is trained with drifting loss in a feature space. The feature extractor $\phi$ is an image encoder. We mainly consider a ResNet-style encoder, pre-trained by self-supervised learning, e.g., MoCo (He et al., 2020) and SimCLR (Chen et al., 2020). When these pre-trained models operate in pixel space, we apply the VAE decoder to map our generator's latent-space output back to pixel space for feature extraction. Gradients are backpropagated through the feature encoder and VAE decoder. We also study an MAE pre-trained in latent space.

For all ResNet-style models, features are extracted from multiple stages (i.e., multi-scale feature maps). The drifting loss is computed at each scale and then combined.

**Pixel-space Generation.**
While our experiments primarily focus on latent-space generation, our models support pixel-space generation. In this case, $\boldsymbol{\epsilon}$ and $\mathbf{x}$ are both 256×256×3. We use a patch size of 16 (i.e., DiT/16). The feature extractor $\phi$ is directly on the pixel space.

---

## 5. Experiments

### 5.1 Toy Experiments

**Evolution of the generated distribution.**
Figure 3 visualizes a 2D toy case, where $q$ evolves toward a bimodal distribution $p$ at training time, under three initializations.

In this toy example, our method approximates the target distribution without exhibiting mode collapse. This holds even when $q$ is initialized in a collapsed single-mode state. This provides intuition into why our method is robust to mode collapse: if $q$ collapses onto one mode, other modes of $p$ will attract the samples, allowing them to continue moving and pushing $q$ to continue evolving.

**Figure 3: Evolution of the generated distribution.**
The distribution $q$ (orange) evolves toward a bimodal target $p$ (blue) during training. We show three initializations of $q$: **(top)**: initialized between the two modes; **(middle)**: initialized far from both modes; **(bottom)**: initialized collapsed onto one mode. Across all initializations, our method approximates the target distribution without mode collapse.

**Evolution of the samples.**
Figure 4 shows the training process on two 2D cases. A small MLP generator is trained. The loss (whose value equals $\|\mathbf{V}\|^2$) decreases as the generated distribution converges to the target. This is in line with our motivation that reducing the drift and pushing towards the equilibrium will approximately yield $p = q$.

**Figure 4: Evolution of samples.**
We show generated points sampled at different training iterations, along with their loss values. The loss (whose value equals $\|\mathbf{V}\|^2$) decreases as the distribution converges to the target. (y-axis is log-scale.)

### 5.2 ImageNet Experiments

We evaluate our models on ImageNet 256×256. Ablation studies use a B/2 model on the SD-VAE latent space, trained for 100 epochs. The drifting loss is in a feature space computed by a latent-MAE encoder. We report FID on 50K generated images.

**Anti-symmetry.**
Our derivation of equilibrium requires the drifting field to be anti-symmetric. In Table 1, we conduct a _destructive_ study that intentionally breaks this anti-symmetry. The anti-symmetric case (our ablation default) works well, while other cases fail catastrophically.

**Table 1: Importance of anti-symmetry.**
Breaking the anti-symmetry leads to failure. (Setting: B/2 model, 100 epochs)

| Case                        | Drifting field $\mathbf{V}$      | FID      |
| --------------------------- | -------------------------------- | -------- |
| **anti-symmetry** (default) | $\mathbf{V}^+ - \mathbf{V}^-$    | **8.46** |
| 1.5× attraction             | $1.5\mathbf{V}^+ - \mathbf{V}^-$ | 41.05    |
| 1.5× repulsion              | $\mathbf{V}^+ - 1.5\mathbf{V}^-$ | 46.28    |
| 2.0× attraction             | $2\mathbf{V}^+ - \mathbf{V}^-$   | 86.16    |
| 2.0× repulsion              | $\mathbf{V}^+ - 2\mathbf{V}^-$   | 112.84   |
| attraction-only             | $\mathbf{V}^+$                   | 177.14   |

**Allocation of Positive and Negative Samples.**
Our method samples positive and negative examples to estimate $\mathbf{V}$. In Table 2, we study the effect of $N_\text{pos}$ and $N_\text{neg}$, under fixed epochs and fixed batch size $B$.

**Table 2: Allocation of positive and negative samples.**
We control the total compute by fixing the epochs (100) and the batch size $B = N_c \times N_\text{pos}$ (4096).

_Increasing positive samples (left):_

| $N_c$ | $N_\text{pos}$ | $N_\text{neg}$ | $B$  | FID      |
| ----- | -------------- | -------------- | ---- | -------- |
| 64    | 1              | 64             | 4096 | 20.43    |
| 64    | 16             | 64             | 4096 | 10.39    |
| 64    | 32             | 64             | 4096 | 8.97     |
| 64    | **64**         | 64             | 4096 | **8.46** |

_Increasing negative samples (right):_

| $N_c$ | $N_\text{pos}$ | $N_\text{neg}$ | $B$  | FID      |
| ----- | -------------- | -------------- | ---- | -------- |
| 512   | 8              | 8              | 4096 | 11.82    |
| 256   | 16             | 16             | 4096 | 10.16    |
| 128   | 32             | 32             | 4096 | 9.32     |
| 64    | 64             | **64**         | 4096 | **8.46** |

**Feature Space for Drifting.**
Our model computes the drifting loss in a feature space. Table 3 compares the feature encoders.

**Table 3: Feature space for drifting.**
We compare self-supervised learning (SSL) encoders. Standard SimCLR and MoCo encoders achieve competitive results, whereas our customized latent-MAE performs best and benefits from increased width and longer training. (Generator setting: B/2 model, 100 epochs)

| SSL method           | arch   | block      | width | SSL ep. | FID      |
| -------------------- | ------ | ---------- | ----- | ------- | -------- |
| SimCLR               | ResNet | bottleneck | 256   | 800     | 11.05    |
| MoCo-v2              | ResNet | bottleneck | 256   | 800     | 8.41     |
| latent-MAE (default) | ResNet | basic      | 256   | 192     | 8.46     |
| latent-MAE           | ResNet | basic      | 384   | 192     | 7.26     |
| latent-MAE           | ResNet | basic      | 512   | 192     | 6.49     |
| latent-MAE           | ResNet | basic      | 640   | 192     | 6.30     |
| latent-MAE           | ResNet | basic      | 640   | 1280    | 4.28     |
| latent-MAE + cls ft  | ResNet | basic      | 640   | 1280    | **3.36** |

**Table 4: From ablation to final setting.**

| Case                      | Arch | Epochs | FID      |
| ------------------------- | ---- | ------ | -------- |
| (a) baseline              | B/2  | 100    | 3.36     |
| (b) longer                | B/2  | 320    | 2.51     |
| (c) longer + hyper-param. | B/2  | 1280   | 1.75     |
| (d) larger model          | L/2  | 1280   | **1.54** |

**System-level Comparisons.**
We compare with previous methods in Table 5 (latent space) and Table 6 (pixel space).

**Table 5: System-level comparison: ImageNet 256×256 generation in latent space.**
FID is on 50K images, all reported with CFG if applicable. The parameter numbers are "generator + decoder". All generators are trained from scratch (i.e., not distilled).

| Method                        | Space  | Params    | NFE   | FID↓     | IS↑   |
| ----------------------------- | ------ | --------- | ----- | -------- | ----- |
| _Multi-step Diffusion/Flows_  |        |           |       |          |       |
| DiT-XL/2                      | SD-VAE | 675M+49M  | 250×2 | 2.27     | 278.2 |
| SiT-XL/2                      | SD-VAE | 675M+49M  | 250×2 | 2.06     | 270.3 |
| SiT-XL/2+REPA                 | SD-VAE | 675M+49M  | 250×2 | 1.42     | 305.7 |
| LightningDiT-XL/2             | VA-VAE | 675M+70M  | 250×2 | 1.35     | 295.3 |
| RAE+DiT-XL/2                  | RAE    | 839M+415M | 50×2  | **1.13** | 262.6 |
| _Single-step Diffusion/Flows_ |        |           |       |          |       |
| iCT-XL/2                      | SD-VAE | 675M+49M  | 1     | 34.24    | --    |
| Shortcut-XL/2                 | SD-VAE | 675M+49M  | 1     | 10.60    | --    |
| MeanFlow-XL/2                 | SD-VAE | 676M+49M  | 1     | 3.43     | --    |
| AdvFlow-XL/2                  | SD-VAE | 673M+49M  | 1     | 2.38     | 284.2 |
| iMeanFlow-XL/2                | SD-VAE | 610M+49M  | 1     | 1.72     | 282.0 |
| _Drifting Models_             |        |           |       |          |       |
| **Drifting Model, B/2**       | SD-VAE | 133M+49M  | 1     | 1.75     | 263.2 |
| **Drifting Model, L/2**       | SD-VAE | 463M+49M  | 1     | **1.54** | 258.9 |

**Table 6: System-level comparison: ImageNet 256×256 generation in pixel space.**
FID is on 50K images, all reported with CFG if applicable. The parameter numbers are of the generator. All generators are trained from scratch (i.e., not distilled).

| Method                        | Space | Params | NFE    | FID↓     | IS↑   |
| ----------------------------- | ----- | ------ | ------ | -------- | ----- |
| _Multi-step Diffusion/Flows_  |       |        |        |          |       |
| ADM-G                         | pix   | 554M   | 250×2  | 4.59     | 186.7 |
| SiD, UViT/2                   | pix   | 2.5B   | 1000×2 | 2.44     | 256.3 |
| VDM++, UViT/2                 | pix   | 2.5B   | 256×2  | 2.12     | 267.7 |
| SiD2, UViT/2                  | pix   | --     | 512×2  | 1.73     | --    |
| SiD2, UViT/1                  | pix   | --     | 512×2  | **1.38** | --    |
| JiT-G/16                      | pix   | 2B     | 100×2  | 1.82     | 292.6 |
| PixelDiT/16                   | pix   | 797M   | 200×2  | 1.61     | 292.7 |
| _Single-step Diffusion/Flows_ |       |        |        |          |       |
| EPG-L/16                      | pix   | 540M   | 1      | 8.82     | --    |
| _GANs_                        |       |        |        |          |       |
| BigGAN                        | pix   | 112M   | 1      | 6.95     | 152.8 |
| GigaGAN                       | pix   | 569M   | 1      | 3.45     | 225.5 |
| StyleGAN-XL                   | pix   | 166M   | 1      | 2.30     | 265.1 |
| _Drifting Models_             |       |        |        |          |       |
| **Drifting Model, B/16**      | pix   | 134M   | 1      | 1.76     | 299.7 |
| **Drifting Model, L/16**      | pix   | 464M   | 1      | **1.61** | 307.5 |

Our method achieves **1.54** FID with _native_ 1-NFE generation. It outperforms all previous 1-NFE methods, which are based on approximating diffusion-/flow-based trajectories. Notably, our Base-size model competes with previous XL-size models. Our best model (FID 1.54) uses a CFG scale of 1.0, which corresponds to "no CFG" in diffusion-based methods.

**Pixel-space Generation.** Our method can naturally work _without_ the latent VAE, i.e., the generator $f$ directly produces 256×256×3 images. Our _one-step_, _pixel-space_ method achieves **1.61** FID, which outperforms or competes with previous multi-step methods. Comparing with other one-step, pixel-space methods (GANs), our method achieves 1.61 FID using only 87G FLOPs; by comparison, StyleGAN-XL produces 2.30 FID using 1574G FLOPs.

### 5.3 Experiments on Robotic Control

Beyond image generation, we further evaluate our method on robotics control. Our experiment designs and protocols follow _Diffusion Policy_ (Chi et al., 2025). At the core of Diffusion Policy is a multi-step, diffusion-based generator; we replace it with our one-step Drifting Model. We directly compute drifting loss on the _raw_ representations for control, using no feature space.

**Table 7: Robotics Control: Comparison with Diffusion Policy.**

| Task                                              | Setting | Diffusion Policy (NFE: 100) | **Drifting Policy** (NFE: 1) |
| ------------------------------------------------- | ------- | --------------------------- | ---------------------------- |
| _Single-Stage Tasks (State & Visual Observation)_ |         |                             |                              |
| Lift                                              | State   | 0.98                        | **1.00**                     |
| Lift                                              | Visual  | **1.00**                    | **1.00**                     |
| Can                                               | State   | 0.96                        | **0.98**                     |
| Can                                               | Visual  | 0.97                        | **0.99**                     |
| ToolHang                                          | State   | 0.30                        | **0.38**                     |
| ToolHang                                          | Visual  | **0.73**                    | 0.67                         |
| PushT                                             | State   | **0.91**                    | 0.86                         |
| PushT                                             | Visual  | 0.84                        | **0.86**                     |
| _Multi-Stage Tasks (State Observation)_           |         |                             |                              |
| BlockPush                                         | Phase 1 | 0.36                        | **0.56**                     |
| BlockPush                                         | Phase 2 | 0.11                        | **0.16**                     |
| Kitchen                                           | Phase 1 | **1.00**                    | **1.00**                     |
| Kitchen                                           | Phase 2 | **1.00**                    | **1.00**                     |
| Kitchen                                           | Phase 3 | **1.00**                    | 0.99                         |
| Kitchen                                           | Phase 4 | **0.99**                    | 0.96                         |

Our 1-NFE model matches or exceeds the state-of-the-art Diffusion Policy that uses 100 NFE. This comparison suggests that Drifting Models can serve as a promising generative model across different domains.

---

## 6. Discussion and Conclusion

We present _Drifting Models_, a new paradigm for generative modeling. At the core of our model is the idea of modeling the evolution of pushforward distributions _during training_. This allows us to focus on the update rule, i.e., $\mathbf{x}_{i+1} = \mathbf{x}_i + \Delta\mathbf{x}_i$, during the iterative training process. This is in contrast with diffusion-/flow-based models, which perform the iterative update at _inference_ time. Our method naturally performs one-step inference.

Given that our methodology is substantially different, many open questions remain. For example, although we show that $q = p \Rightarrow \mathbf{V} = \mathbf{0}$, the converse implication does not generally hold in theory. While our designed $\mathbf{V}$ performs well empirically, it remains unclear under what conditions $\mathbf{V} \to \mathbf{0}$ leads to $q \to p$.

From a practical standpoint, although our paper presents an effective instantiation of drifting modeling, many of our design decisions may remain sub-optimal. For example, the design of the drifting field and its kernels, the feature encoder, and the generator architecture remain open for future exploration.

From a broader perspective, our work reframes iterative neural network training as a mechanism for distribution evolution, in contrast to the differential equations underlying diffusion-/flow-based models. We hope that this perspective will inspire the exploration of other realizations of this mechanism in future work.

---

## Acknowledgements

We greatly thank Google TPU Research Cloud (TRC) for granting us access to TPUs. We thank Michael Albergo, Ziqian Zhong, Yilun Xu, Zhengyang Geng, Hanhong Zhao, Jiangqi Dai, Alex Fan, and Shaurya Agrawal for helpful discussions. Mingyang Deng is partially supported by funding from MIT-IBM Watson AI Lab.

---

## Appendix A: Additional Implementation Details

Table A1 summarizes the configurations and hyper-parameters for ablation studies and system-level comparisons.

**Table A1: Configurations for ImageNet 256×256.**

|                                         | Ablation default                | B/2, latent                     | L/2, latent                                             | B/16, pixel                     | L/16, pixel                     |
| --------------------------------------- | ------------------------------- | ------------------------------- | ------------------------------------------------------- | ------------------------------- | ------------------------------- |
| **Generator Architecture**              |                                 |                                 |                                                         |                                 |                                 |
| arch                                    | DiT-B/2                         | DiT-B/2                         | DiT-L/2                                                 | DiT-B/16                        | DiT-L/16                        |
| input size                              | 32×32×4                         | 32×32×4                         | 32×32×4                                                 | 32×32×4                         | 32×32×4                         |
| patch size                              | 2×2                             | 2×2                             | 2×2                                                     | 16×16                           | 16×16                           |
| hidden dim                              | 768                             | 768                             | 1024                                                    | 768                             | 1024                            |
| depth                                   | 12                              | 12                              | 24                                                      | 12                              | 24                              |
| register tokens                         | 16                              | 16                              | 16                                                      | 16                              | 16                              |
| style embedding tokens                  | 32                              | 32                              | 32                                                      | 32                              | 32                              |
| **Feature Encoder for Drifting Loss**   |                                 |                                 |                                                         |                                 |                                 |
| arch                                    | ResNet                          | ResNet                          | ResNet                                                  | ResNet + ConvNeXt-V2            | ResNet + ConvNeXt-V2            |
| SSL pre-train method                    | latent-MAE                      | latent-MAE                      | latent-MAE                                              | pixel-MAE                       | pixel-MAE                       |
| ResNet: input size                      | 32×32×4                         | 32×32×4                         | 32×32×4                                                 | 256×256×3                       | 256×256×3                       |
| ResNet: conv₁ stride                    | 1                               | 1                               | 1                                                       | 8                               | 8                               |
| ResNet: base width                      | 256                             | 640                             | 640                                                     | 640                             | 640                             |
| ResNet: block type                      | bottleneck                      | bottleneck                      | bottleneck                                              | bottleneck                      | bottleneck                      |
| ResNet: blocks / stage                  | [3, 4, 6, 3]                    | [3, 4, 6, 3]                    | [3, 4, 6, 3]                                            | [3, 4, 6, 3]                    | [3, 4, 6, 3]                    |
| ResNet: size / stage                    | [32², 16², 8², 4²]              | [32², 16², 8², 4²]              | [32², 16², 8², 4²]                                      | [32², 16², 8², 4²]              | [32², 16², 8², 4²]              |
| MAE: masking ratio                      | 50%                             | 50%                             | 50%                                                     | 50%                             | 50%                             |
| MAE: pre-train epochs                   | 192                             | 1280                            | 1280                                                    | 1280                            | 1280                            |
| classification finetune                 | No                              | 3k steps                        | 3k steps                                                | 3k steps                        | 3k steps                        |
| **Generator Optimizer**                 |                                 |                                 |                                                         |                                 |                                 |
| optimizer                               | AdamW (β₁=0.9, β₂=0.95)         | same                            | same                                                    | same                            | same                            |
| learning rate                           | 2e-4                            | 4e-4                            | 4e-4                                                    | 2e-4                            | 4e-4                            |
| weight decay                            | 0.01                            | 0.0                             | 0.01                                                    | 0.01                            | 0.01                            |
| warmup steps                            | 5k                              | 10k                             | 10k                                                     | 10k                             | 10k                             |
| gradient clip                           | 2.0                             | 2.0                             | 2.0                                                     | 2.0                             | 2.0                             |
| training steps                          | 30k                             | 200k                            | 200k                                                    | 100k                            | 100k                            |
| training epochs                         | 100                             | 1280                            | 1280                                                    | 640                             | 640                             |
| EMA decay                               | 0.999                           | {0.999, 0.9995, 0.9998, 0.9999} | same                                                    | same                            | same                            |
| **Drifting Loss Computation**           |                                 |                                 |                                                         |                                 |                                 |
| class labels $N_c$                      | 64                              | 128                             | 128                                                     | 128                             | 128                             |
| positive samples $N_\text{pos}$         | 64                              | 128                             | 64                                                      | 128                             | 128                             |
| generated samples $N_\text{neg}$        | 64                              | 64                              | 64                                                      | 64                              | 64                              |
| effective batch $B$                     | 4096                            | 8192                            | 8192                                                    | 8192                            | 8192                            |
| temperatures $\tau$                     | {0.02, 0.05, 0.2}               | same                            | same                                                    | same                            | same                            |
| **CFG Configuration**                   |                                 |                                 |                                                         |                                 |                                 |
| train: CFG $\alpha$ range               | [1, 4]                          | [1, 4]                          | [1, 4]                                                  | [1, 4]                          | [1, 4]                          |
| train: CFG $\alpha$ sampling            | $p(\alpha) \propto \alpha^{-3}$ | $p(\alpha) \propto \alpha^{-5}$ | 50%: $\alpha{=}1$, 50%: $p(\alpha) \propto \alpha^{-3}$ | $p(\alpha) \propto \alpha^{-5}$ | $p(\alpha) \propto \alpha^{-5}$ |
| train: uncond samples $N_\text{uncond}$ | 16                              | 32                              | 32                                                      | 32                              | 32                              |
| inference: CFG $\alpha$ search          | [1.0, 3.5]                      | same                            | same                                                    | same                            | same                            |

### A.1 Pseudo-code for Computing Drifting Field $\mathbf{V}$

Algorithm A1 provides the pseudo-code for computing $\mathbf{V}$. The computation is based on taking empirical means in Eq. (4) and Eq. (5), which are implemented as softmax over the $\mathbf{y}$-sample axis. In practice, we further normalize over the $\mathbf{x}$-sample axis, also implemented as softmax on the same logit matrix.

It is worth noting that this implementation preserves the desired property of $\mathbf{V}$. In principle, this implementation can be viewed as a Monte Carlo estimation of a drifting field:

$$\mathbf{V}_{p,q}(\mathbf{x}) = \mathbb{E}_{\mathcal{B},p,q}\big[\tilde{K}_\mathcal{B}(\mathbf{x}, \mathbf{y}^+)\tilde{K}_\mathcal{B}(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^+ - \mathbf{y}^-)\big],$$

where $\mathcal{B}$ consists of other samples in the batch and $\tilde{K}_\mathcal{B}$ denotes normalizing the distance based on statistics within $\mathcal{B}$. This $\mathbf{V}$ also satisfies $\mathbf{V}_{p,p}(\mathbf{x}) = \mathbf{0}$.

**Algorithm A1: Computing the drifting field $\mathbf{V}$.**

```python
def compute_V(x, y_pos, y_neg, T):
  # x: [N, D]
  # y_pos: [N_pos, D]
  # y_neg: [N_neg, D]
  # T: temperature

  # compute pairwise distance
  dist_pos = cdist(x, y_pos)  # [N, N_pos]
  dist_neg = cdist(x, y_neg)  # [N, N_neg]

  # ignore self (if y_neg is x)
  dist_neg += eye(N) * 1e6

  # compute logits
  logit_pos = -dist_pos / T
  logit_neg = -dist_neg / T

  # concat for normalization
  logit = cat([logit_pos, logit_neg], dim=1)
  # normalize along both dimensions
  A_row = logit.softmax(dim=-1)
  A_col = logit.softmax(dim=-2)
  A = sqrt(A_row * A_col)

  # back to [N, N_pos] and [N, N_neg]
  A_pos, A_neg = split(A, [N_pos,], dim=1)

  # compute the weights
  W_pos = A_pos  # [N, N_pos]
  W_neg = A_neg  # [N, N_neg]
  W_pos *= A_neg.sum(dim=1, keepdim=True)
  W_neg *= A_pos.sum(dim=1, keepdim=True)

  drift_pos = W_pos @ y_pos  # [N_x, D]
  drift_neg = W_neg @ y_neg  # [N_x, D]

  V = drift_pos - drift_neg
  return V
```

### A.2 Generator Architecture

**Input and output.**
The input to the generator consists of random noise along with conditioning:

$$f_\theta: (\boldsymbol{\epsilon}, c, \alpha) \mapsto \mathbf{x}$$

where $\boldsymbol{\epsilon}$ denotes random variables, $c$ is a class label, and $\alpha$ is the CFG strength. $\boldsymbol{\epsilon}$ may consist of both continuous random variables (e.g., Gaussian noise) and discrete ones (e.g., uniformly distributed integers; see random style embeddings). For latent-space models, the output $\mathbf{x} \in \mathbb{R}^{32\times 32\times 4}$ is in the SD-VAE latent space. For pixel-space models, the output $\mathbf{x} \in \mathbb{R}^{256\times 256\times 3}$ is directly an image.

**Transformer.**
We adopt a DiT-style Transformer. Following Yao et al. (2025), we use SwiGLU, RoPE, RMSNorm, and QK-Norm. The input Gaussian noise is patchified into $256 = 16\times 16$ tokens (patch size 2×2 for latent, 16×16 for pixel). Conditioning $(c, \alpha)$ is processed by adaLN, as well as by in-context conditioning tokens. The output tokens are unpatchified back to the target shape.

**In-context tokens.**
Following Li et al. (2025), we prepend 16 learnable tokens to the sequence for in-context conditioning. These tokens are formed by summing the projected conditioning vector with positional embeddings.

**Random style embeddings.**
Our framework allows arbitrary noise distributions beyond Gaussians. Inspired by StyleGAN, we introduce an additional 32 "style tokens": each of which is a random index into a codebook of 64 learnable embeddings. These are summed and added to the conditioning vector. This does not change the sequence length and introduces negligible overhead in terms of parameters and FLOPs.

Effect of style embeddings on ablation default:

|     | w/o style | w/ style |
| --- | --------- | -------- |
| FID | 8.86      | **8.46** |

In contrast to diffusion-/flow-based methods, our method can naturally handle different types of noise or random variables. With random style embeddings, the input random variables consist of two parts: (1) Gaussian noise, and (2) discrete indices for style embeddings. Our model $f$ produces the pushforward distribution of their joint distribution.

### A.3 Implementation of ResNet-style MAE

In addition to standard self-supervised learning models (MoCo, SimCLR), we develop a customized ResNet-style MAE model as the feature encoder for drifting loss.

**Overview.**
Unlike standard MAE, which is based on ViT, our MAE trains a convolutional ResNet that provides multi-scale features. For latent-space models, the input and output have dimension 32×32×4; for pixel-space models, the input and output have dimension 256×256×3. Our MAE consists of a ResNet-style encoder paired with a deconvolutional decoder in a U-Net-style encoder-decoder architecture. We only use the ResNet-style encoder for feature extraction when computing the drifting loss.

**MAE Encoder.**
The encoder follows a classical ResNet design. It maps an input to multi-scale feature maps (4 scales in ResNet):

$$\text{Encoder}: \mathbf{x} \mapsto \{\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4\}$$

Here, a feature map $\mathbf{f}_i$ has dimension $H_i \times W_i \times C_i$, with $H_i \times W_i \in \{32^2, 16^2, 8^2, 4^2\}$ and $C_i \in \{C, 2C, 4C, 8C\}$ for a base width $C$. The architecture follows standard ResNet design, with GroupNorm (GN) used in place of BatchNorm (BN). All residual blocks are "basic" blocks (i.e., each consisting of two 3×3 convolutions). Following the standard ResNet-34: the encoder has a 3×3 convolution (without downsampling) and 4 stages with [3, 4, 6, 3] blocks; downsampling (stride 2) happens at the first block of stages 2 to 4.

For latent-space (i.e., latent-MAE), the input of this ResNet is 32×32×4; for pixel-space, the 256×256×3 input is first patchified (by a 8×8 patch) into 32×32×192. The ResNet operates on the input with $H \times W = 32 \times 32$.

**MAE Decoder.**
The decoder returns to the input shape via deconvolutions and skip connections:

$$\text{Decoder}: \{\mathbf{f}_4, \mathbf{f}_3, \mathbf{f}_2, \mathbf{f}_1\} \mapsto \hat{\mathbf{x}}.$$

It starts with a 3×3 convolutional block on $\mathbf{f}_4$, followed by 4 upsampling blocks. Each upsampling block performs: bilinear 2×2 upsampling → concatenating with encoder's skip connection → GN → two 3×3 convolutions with GN and ReLU. A final 1×1 convolution produces the output channels.

**Masking.**
The MAE is trained to reconstruct randomly masked inputs. Unlike the ViT-based MAE, which removes the masked tokens from the sequence, we simply zero out masked patches. For the input of a shape $H \times W = 32 \times 32$, we mask 2×2 patches by zeroing. Each patch is independently masked with 50% probability.

**MAE training.**
We minimize the $\ell_2$ reconstruction loss on the masked regions. We use AdamW with learning rate $4 \times 10^{-3}$ and a batch size of 8192. EMA with decay 0.9995 is used. Following He et al. (2022), we apply random resized crop augmentation to the input.

**Classification fine-tuning.**
For our best feature encoder, we fine-tune the MAE model with a linear classifier head. The loss is $\lambda \mathcal{L}_{\text{cls}} + (1-\lambda)\mathcal{L}_{\text{recon}}$. We fine-tune all parameters for 3k iterations, where $\lambda$ follows a linear warmup schedule, increasing from 0 to 0.1 over the first 1k iterations.

### A.4 Other Pretrained Feature Encoders

**MoCo and SimCLR.** We evaluate publicly available self-supervised encoders trained on ImageNet in pixel space: MoCo, SimCLR. We use the ResNet-50 variant. For latent-space generation, we apply the VAE decoder to map generator outputs from latent space (32×32×4) to pixel space (256×256×3) before feature extraction.

**MAE with ConvNeXt-V2.** In our pixel-space generator, we also investigate ConvNeXt-V2 as the feature encoder. We note that ConvNeXt-V2 is a self-supervised pre-trained model using the MAE objective, followed by classification fine-tuning. Like ResNet, ConvNeXt-V2 is a multi-stage architecture.

### A.5 Multi-scale Features for Drifting Loss

Given an image, the feature encoder produces feature maps at multiple scales, with multiple spatial locations per scale. We compute one drifting loss per feature (e.g., per scale and/or per location). Specifically, we compute the kernel, the drift, and the resulting loss independently for each feature. The resulting losses are summed.

For each stage in a ResNet, we extract features from the output of every 2 residual blocks, together with the final output. This yields a set of feature maps, each of shape $H_i \times W_i \times C_i$. For each feature map, we produce:

(a) $H_i \times W_i$ vectors, one per location (each $C_i$-dim);
(b) 1 global mean and 1 global std (each $C_i$-dim);
(c) $\frac{H_i}{2} \times \frac{W_i}{2}$ vectors of means and stds (each $C_i$-dim), computed over 2×2 patches;
(d) $\frac{H_i}{4} \times \frac{W_i}{4}$ vectors of means and stds (each $C_i$-dim), computed over 4×4 patches.

In addition, for the encoder's input ($H_0 \times W_0 \times C_0$), we compute the mean of squared values ($x^2$) per channel and obtain a $C_0$-dim vector.

Effect of these designs on the ablation default:

|     | (a,b) | (a–c) | (a–d)    |
| --- | ----- | ----- | -------- |
| FID | 9.58  | 9.10  | **8.46** |

This shows that our method benefits from richer feature sets.

### A.6 Feature and Drift Normalization

To balance the multiple loss terms from multiple features, we perform normalization for each feature $\phi_j$.

**Feature Normalization.** Consider a feature $\phi_j \in \mathbb{R}^{C_j}$. We define a normalization scale $S_j \in \mathbb{R}$ and the normalized feature is denoted by:

$$\tilde{\phi}_j := \phi_j / S_j.$$

We want the _average_ distance to be $\sqrt{C_j}$:

$$\mathbb{E}_\mathbf{x}\, \mathbb{E}_\mathbf{y}\big[ dist_j(\mathbf{x}, \mathbf{y})\big] \approx \sqrt{C_j}.$$

To achieve this, we set the normalization scale $S_j$ as:

$$S_j = \frac{1}{\sqrt{C_j}} \mathbb{E}_\mathbf{x}\, \mathbb{E}_\mathbf{y}\big[\|\phi_j(\mathbf{x}) - \phi_j(\mathbf{y})\|\big].$$

With the normalized feature, the kernel is set as:

$$k(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\frac{1}{\tilde{\tau}_j}\|\tilde{\phi}_j(\mathbf{x}) - \tilde{\phi}_j(\mathbf{y})\|\right),$$

where $\tilde{\tau}_j := \tau \cdot \sqrt{C_j}$. We set $\tau \in \{0.02, 0.05, 0.2\}$.

**Drift Normalization.** We perform a drift normalization on $\mathbf{V}_j$, for each feature $\phi_j$. Formally, we define a normalization scale $\lambda_j \in \mathbb{R}$ and denote:

$$\tilde{\mathbf{V}}_j := \mathbf{V}_j / \lambda_j,$$

where we set $\lambda_j = \sqrt{\mathbb{E}\left[\frac{1}{C_j}\|\mathbf{V}_j\|^2\right]}$.

With the normalized feature and normalized drift, the drifting loss of the feature $\phi_j$ is:

$$\mathcal{L}_j = \text{MSE}\!\left(\tilde{\phi}_j(\mathbf{x}) - \texttt{sg}\!\left(\tilde{\phi}_j(\mathbf{x}) + \tilde{\mathbf{V}}_j\right)\right),$$

The overall loss is the sum across all features: $\mathcal{L} = \sum_j \mathcal{L}_j$.

**Multiple temperatures.**
Using normalized feature distances, the value of temperature $\tau$ determines what is considered "nearby". To improve robustness across different features and across different pretrained models, we adopt multiple temperatures.

Formally, for each $\tau$ value, we compute the normalized drift, denoted by $\tilde{\mathbf{V}}_{j,\tau}$. Then we compute an aggregated field: $\tilde{\mathbf{V}}_j \leftarrow \sum_\tau \tilde{\mathbf{V}}_{j,\tau}$.

Effect of multiple temperatures on ablation default:

| $\tau$ | 0.02  | 0.05     | 0.2  | {0.02, 0.05, 0.2} |
| ------ | ----- | -------- | ---- | ----------------- |
| FID    | 10.62 | **8.67** | 8.96 | **8.46**          |

Using multiple temperatures can achieve slightly better results than using a single optimal temperature. We fix $\tau \in \{0.02, 0.05, 0.2\}$ and do not require tuning this hyperparameter across different configurations.

**Normalization across spatial locations.**
For a feature map of resolution $H_i \times W_i$, there are $H_i \times W_i$ per-location features. We assume that features at different locations within the same feature map share the same normalization scale.

### A.7 Classifier-Free Guidance (CFG)

To support CFG, at training time, we include $N_\text{unc}$ additional unconditional samples (real images from random classes) as extra negatives. These samples are weighted by a factor $w$ when computing the kernel. For a generated sample $\mathbf{x}$, the effective negative distribution it compares with is:

$$\tilde{q}(\cdot | c) \triangleq \frac{(N_\text{neg}-1) \cdot q_\theta(\cdot|c) + N_\text{unc}\, w \cdot p_{\text{data}}(\cdot|\varnothing)}{(N_\text{neg}-1) + N_\text{unc}\, w}.$$

Given a CFG strength $\alpha$, we compute $w$ accordingly, which is used to weight the kernel. At inference time, we specify a value of $\alpha$. The inference-time computation remains to be one-step (1-NFE).

### A.8 Sample Queue

Our method requires access to randomly sampled _real_ (positive/unconditional) data. Instead of a specialized data loader, we adopt a _sample queue_ of cached data, similar to the queue used in MoCo. For each class label, we keep a queue of size 128; for unconditional samples, we maintain a separate global queue of size 1000. At each training step, we push the latest 64 new real samples into the corresponding queues. We sample without replacement.

### A.9 Training Loop

In the training loop, each step proceeds as:

1. Sample a batch ($N_c$) of class labels.
2. For each label $c$, sample a CFG scale $\alpha$.
3. Sample a batch ($N_\text{neg}$) of noise $\boldsymbol{\epsilon}$. Feed $(\boldsymbol{\epsilon}, c, \alpha)$ to the generator $f$ to produce generated samples.
4. Sample positive samples (same class, $N_\text{pos}$) and unconditional samples (for CFG, $N_\text{unc}$).
5. Extract features on all generated, positive, and unconditional samples.
6. Compute the drifting loss using the features.
7. Run backpropagation and parameter update.

---

## Appendix B: Additional Experimental Results

### B.1 Ablations on Pixel-Space Generation

**Table B1: Ablations on pixel-space generation.**

| Feature encoder $\phi$                | Latent (B/2) FID | Pixel (B/16) FID |
| ------------------------------------- | ---------------- | ---------------- |
| MAE (width 256, epoch 192)            | 8.46             | 32.11            |
| MAE (width 640, epoch 1280) + cls ft. | 3.36             | 9.35             |
| + MAE w/ ConvNeXt-V2                  | --               | **3.70**         |

**Table B2: Pixel-space generation: from ablation to final setting.**

| Case                      | Arch | Epochs | FID      |
| ------------------------- | ---- | ------ | -------- |
| (a) baseline              | B/16 | 100    | 3.70     |
| (b) longer + hyper-param. | B/16 | 320    | 2.19     |
| (c) longer                | B/16 | 640    | 1.76     |
| (d) larger model          | L/16 | 640    | **1.61** |

Table B1 shows that the choice of feature encoder plays a more significant role in pixel-space generation quality. A weaker MAE encoder yields an FID of 32.11, whereas a stronger MAE encoder improves performance to an FID of 9.35. Adding ConvNeXt-V2 further improves the result to an FID of 3.70. We achieve an FID of 1.61 for pixel-space generation.

### B.2 Ablation on Kernel Normalization

**Table B3: Ablation on kernel normalization.**

| Kernel normalization                                 | FID      |
| ---------------------------------------------------- | -------- |
| softmax over $\mathbf{x}$ and $\mathbf{y}$ (default) | **8.46** |
| softmax over $\mathbf{y}$                            | 8.92     |
| no normalization                                     | 10.54    |

Using the $\mathbf{y}$-only softmax performs well (8.92 FID), whereas using both $\mathbf{x}$ and $\mathbf{y}$ softmax improves the result (8.46 FID). On the other hand, even without normalization, performance remains decent, demonstrating the robustness of our method. We note that all three variants satisfy the equilibrium condition $\mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0}$ when $p = q$.

### B.3 Ablation on CFG

Figure B1 shows the CFG scale $\alpha$ used at inference time. It shows that the CFG formulation developed for our models exhibits behavior similar to that observed in diffusion-/flow-based models. Increasing the CFG scale leads to higher IS values, whereas beyond the FID sweet spot, further increases in IS come at the cost of worse FID. Notably, with our best model (L/2), the optimal FID is achieved at $\alpha = 1.0$.

**Figure B1: Effect of CFG scale $\alpha$.**
(a): FID vs. $\alpha$. (b): IS vs. $\alpha$. (c): IS vs. FID. We show the L/2 (solid) and B/2 (dashed) models.

### B.4 Nearest Neighbor Analysis

We show generated images together with their nearest real images retrieved from the ImageNet training set using CLIP features. These visualizations suggest that our method generates novel images that are visually distinct from their nearest neighbors, rather than merely memorizing training samples.

### B.5 Qualitative Results

Uncurated samples from our latent-L/2 model with CFG = 1.0 achieve FID = 1.54, IS = 258.9. Side-by-side comparisons with improved MeanFlow (iMF) are also provided, where both methods generate images with a single neural function evaluation (1-NFE). For fair comparison, we set the CFG scale to match the IS of iMF visualizations (IS 354.4, CFG=1.5), and achieve FID 3.01 with our method (DiT-L/2), compared to FID 3.92 for iMF (DiT-XL/2, CFG=6.0).

---

## Appendix C: Additional Derivations

### C.1 On Identifiability of the Zero-Drift Equilibrium

In Sec. 3.2, we showed that anti-symmetry implies $p = q \Rightarrow \mathbf{V}(\mathbf{x}) \equiv \mathbf{0}$. Here we investigate the converse: under what conditions does $\mathbf{V}(\mathbf{x}) \approx \mathbf{0}$ imply $p \approx q$?

**Setup.** Consider a general interaction kernel $K(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-) \in \mathbb{R}^d$ and the drifting field:

$$\mathbf{V}_{p,q}(\mathbf{x}) := \mathbb{E}_{\mathbf{y}^+ \sim p,\; \mathbf{y}^- \sim q}\big[K(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-)\big].$$

We assume that $p$ and $q$ belong to a finite-dimensional model class spanned by a linearly independent basis $\{\varphi_i\}_{i=1}^m$:

$$p(\mathbf{y}) = \sum_{i=1}^m a_i\, \varphi_i(\mathbf{y}), \qquad q(\mathbf{y}) = \sum_{i=1}^m b_i\, \varphi_i(\mathbf{y}),$$

where $\mathbf{a}, \mathbf{b} \in \mathbb{R}^m$ are coefficient vectors.

**Bilinear expansion over test locations.** Consider a set of test locations (probes) $\mathcal{X} = \{\mathbf{x}_k\}_{k=1}^N$ with sufficiently large $N$ (e.g., $N \gg m^2$). For each pair of basis indices $(i,j)$, we define the _induced interaction vector_ $\mathbf{U}_{ij} \in \mathbb{R}^{d \times N}$:

$$\mathbf{U}_{ij}[:, \mathbf{x}] \triangleq \iint K(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-)\, \varphi_i(\mathbf{y}^+)\, \varphi_j(\mathbf{y}^-)\, d\mathbf{y}^+ d\mathbf{y}^-.$$

The drifting field evaluated on $\mathcal{X}$ is a bilinear combination:

$$\mathbf{V}_\mathcal{X} \triangleq \sum_{i=1}^m \sum_{j=1}^m a_i b_j \mathbf{U}_{ij}.$$

**Linear independence assumption.** Our anti-symmetry condition implies $\mathbf{U}_{ij} = -\mathbf{U}_{ji}$ (and consequently $\mathbf{U}_{ii} = \mathbf{0}$). We make the _generic non-degeneracy assumption_: The set of vectors $\{\mathbf{U}_{ij}\}_{1 \le i < j \le m}$ is linearly independent in $\mathbb{R}^{dN}$.

**Uniqueness of the equilibrium.** The zero-drift condition $\mathbf{V}(\mathbf{x}) \equiv \mathbf{0}$ implies $\mathbf{V}_\mathcal{X} = \mathbf{0}$. Grouping terms by the independent basis vectors $\{\mathbf{U}_{ij}\}_{i<j}$:

$$\sum_{1 \le i < j \le m}(a_i b_j - a_j b_i)\mathbf{U}_{ij} = \mathbf{0}.$$

By the linear independence assumption, $a_i b_j - a_j b_i = 0$ for all $i, j$. This implies that the vector $\mathbf{a}$ is parallel to $\mathbf{b}$. Since $p$ and $q$ are probability densities (implying $\int p = \int q = 1$), we must have $\mathbf{a} = \mathbf{b}$, and thus $p = q$.

**Connection to the mean shift field.** The mean-shift field fits this framework. The update vector (before normalization) is $\mathbb{E}_{p,q}[k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^+ - \mathbf{y}^-)]$. This corresponds to an interaction kernel of the form:

$$K(\mathbf{x}, \mathbf{y}^+, \mathbf{y}^-) = k(\mathbf{x}, \mathbf{y}^+)\, k(\mathbf{x}, \mathbf{y}^-)\, (\mathbf{y}^+ - \mathbf{y}^-).$$

Since we can choose $N$ such that $dN \gg m^2$, the linear independence of $\{\mathbf{U}_{ij}\}$ is expected to hold for generic configurations. For general distributions $p$ and $q$, we can approximate them using a sufficiently large basis expansion, and when the basis approximation is sufficiently accurate, $p \approx q$.

### C.2 The Drifting Field of MMD

In principle, if a method minimizes a discrepancy between two distributions $p$ and $q$ and reaches minimum at $p = q$, then a drifting field $\mathbf{V}$ exists that governs sample movement: we can let $\mathbf{V} \propto -\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, which is zero when $p = q$.

**Gradients of Drifting Loss.** With $\mathbf{x} = f_\theta(\boldsymbol{\epsilon})$, our drifting loss can be written as:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim q}\Big[\big\|\mathbf{x} - \texttt{sg}\big(\mathbf{x} + \mathbf{V}(\mathbf{x})\big)\big\|^2\Big].$$

The gradient w.r.t. $\mathbf{x}$ gives $\frac{\partial \mathcal{L}(\mathbf{x})}{\partial \mathbf{x}} = -2\mathbf{V}(\mathbf{x})$, leading to:

$$\mathbf{V}(\mathbf{x}) = -\frac{1}{2} \frac{\partial \mathcal{L}(\mathbf{x})}{\partial \mathbf{x}}. \tag{10}$$

**Gradients of MMD Loss.** In MMD-based methods, the difference between two distributions $p$ and $q$ is measured by squared MMD:

$$\mathcal{L}_{\text{MMD}^2}(p, q) = \mathbb{E}_{\mathbf{x},\mathbf{x}' \sim q}[\xi(\mathbf{x}, \mathbf{x}')] - 2\, \mathbb{E}_{\mathbf{y} \sim p,\; \mathbf{x} \sim q}[\xi(\mathbf{y}, \mathbf{x})] + \text{const}.$$

The gradient w.r.t. $\mathbf{x}$ is:

$$\frac{\partial \mathcal{L}_{\text{MMD}^2}(\mathbf{x})}{\partial \mathbf{x}} = 2\mathbb{E}_{\mathbf{y}^- \sim q}\Big[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^-)}{\partial \mathbf{x}}\Big] - 2\mathbb{E}_{\mathbf{y}^+ \sim p}\Big[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^+)}{\partial \mathbf{x}}\Big].$$

Comparing with Eq. (10), we obtain the underlying drifting field corresponding to the MMD loss:

$$\mathbf{V}_\text{MMD}(\mathbf{x}) \triangleq \mathbb{E}_{\mathbf{y}^+ \sim p}\Big[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^+)}{\partial \mathbf{x}}\Big] - \mathbb{E}_{\mathbf{y}^- \sim q}\Big[\frac{\partial \xi(\mathbf{x}, \mathbf{y}^-)}{\partial \mathbf{x}}\Big].$$

For a radial kernel $\xi(\mathbf{x}, \mathbf{y}) = \xi(R)$ where $R = \|\mathbf{x} - \mathbf{y}\|^2$:

$$\mathbf{V}_\text{MMD}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}^+ \sim p}\Big[2\xi'(\|\mathbf{x}-\mathbf{y}^+\|^2)(\mathbf{x} - \mathbf{y}^+)\Big] - \mathbb{E}_{\mathbf{y}^- \sim q}\Big[2\xi'(\|\mathbf{x}-\mathbf{y}^-\|^2)(\mathbf{x} - \mathbf{y}^-)\Big].$$

When $\xi$ is a Gaussian function, we have $\tilde{k}(\mathbf{x}, \mathbf{y}) = \frac{1}{\sigma^2}\exp(-\frac{1}{2\sigma^2}\|\mathbf{x} - \mathbf{y}\|^2)$.

**Relations and Differences.** We summarize the key differences between our model and the MMD-based methods:

(i) Our method is formulated around the drifting field $\mathbf{V}$, which is more flexible and general.
(ii) Our method supports and leverages _normalized_ kernels $\frac{1}{Z}k(\mathbf{x}, \mathbf{y})$ that cannot be naturally derived from the MMD perspective.
(iii) Our $\mathbf{V}$-centric formulation enables a flexible step size for drifting (i.e., $\mathbf{x} \leftarrow \mathbf{x} + \eta\mathbf{V}$) and therefore naturally supports $\mathbf{V}$-normalization.
(iv) Our $\mathbf{V}$-centric formulation allows the equilibrium concept to be naturally extended to support CFG, whereas a CFG variant for MMD remains unexplored.

In summary, although a special case of our method reduces to MMD, our $\mathbf{V}$-centric framework is more general and enables unique possibilities that are important in practice.

---

_[Figures and image visualizations (uncurated samples, side-by-side comparisons with improved MeanFlow) are not included in this Markdown conversion as they reference external image files.]_
