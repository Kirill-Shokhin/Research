# Mamba or first Selective SSM
![image (7)](https://github.com/Kirill-Shokhin/Research/assets/46619252/b6209055-8de7-43ae-902e-f116735ef9b2)

*In the era of ubiquitous Transformer dominance, where they consumed more and more silicon chips, and it seemed that things couldn't get any better, with the cost of each new token skyrocketing exponentially from the previous ones, she emerged - **Mamba**.*

Transformers have made a profound impact on the field of Deep Learning, showcasing remarkable efficiency. However, they face a significant limitation in terms of input sequence length (context) due to their quadratic computational complexity. Most models operate with a context length of less than 10 000, rendering them impractical for tasks with large volumes of input data. Despite various rumors swirling around, it would indeed be strange to witness a strong AI that could be chatted into oblivion within a matter of minutes.

|   | Complexity | Context length | Throughput |
| :---:  | :---:  | :---:  | :---:  |
| Transformer  | $L^2$   | $\sim 10^3$  |  $x$ |
| Mamba  | $L$  | $\sim 10^6$  | $5x$ |

Mamba is built on a fundamentally different approach - **state space model (SSM)**, which, although much older than Transformers, had not shown sufficient effectiveness in deep learning, particularly as a language model. Mamba boasts a linear computational dependency and five times higher throughput than Transformers. The authors tested their innovation on a series of models with parameters **up to 2.8 billion**, which may not yet rival **ChatGPT**, but already outperforms current top language models in its weight category. The context length was chosen to match that of the corresponding Transformer. So a context size of a million was only tested on simple synthetic tests. However, this is also significant, as neither Transformers nor convolutions were able to handle these tests. In this article, we will delve into the mathematics behind the new architecture, exploring its advantages and limitations in detail.

## Linear state space model
### Continuous case

The state space model, upon which the entire concept is built, appears in continuous form as follows:

$$ \boldsymbol{\dot h}(t) = \boldsymbol{Ah}(t)+\boldsymbol{Bx}(t) $$

$$\boldsymbol{y}(t) = \boldsymbol{Ch}(t)+\boldsymbol{Dx}(t) $$

Using such a model, a differential equation of the $`N`$-th order can be expressed as $`N`$ first-order equations in matrix form, where $`\boldsymbol{h}(t)`$ is the state vector containing derivatives from order $`0`$ to $`N-1`$, $`x(t)`$ - input signal, and $`y(t)`$  - output signal. Thus, the complexity of the described system nonlinearly increases with $`N`$.

Alternatively, one can view the model as follows: a one-dimensional signal $`x(t)`$ is mapped to an $`N`$-dimensional latent state $`\boldsymbol{h}(t)`$, which is then projected onto the output signal $`y(t)`$.

To interact with the model in vector form of finite dimensionality, it needs to be discretized.

### Discretization
The second equation is discretized straightforwardly: $\boldsymbol{y_k} = \boldsymbol{y}(k\Delta) \rightarrow \boldsymbol{y_k} = \boldsymbol{Ch_k}+\boldsymbol{Dx_k}$. Let's delve into the details for the first equation:

#### General solution
Let's multiply the first equation by $`e^{-\boldsymbol{A}t}`$:

$$ e^{-\boldsymbol{A}t} \boldsymbol{\dot h}(t) - e^{-\boldsymbol{A}t} \boldsymbol{Ah}(t) = e^{-\boldsymbol{A}t} \boldsymbol{Bx}(t)$$

$$ \frac{d}{dt} (e^{-\boldsymbol{A}t} \boldsymbol{h}(t)) = e^{-\boldsymbol{A}t} \boldsymbol{Bx}(t)$$

Then the general solution for the continuous model will be:

$$\boldsymbol{h}(t) = e^{\boldsymbol{A}t} \boldsymbol{h}(0) + \int_0^t{e^{\boldsymbol{A}(t-\tau)}} \boldsymbol{Bx}(\tau) d\tau$$

Discretization step $`\Delta \Rightarrow \boldsymbol{x_k} = \boldsymbol{x}(k \Delta), \boldsymbol{h_k} = \boldsymbol{h}(k \Delta)`$:

$$ \boldsymbol{h_k} = e^{\boldsymbol{A}k\Delta} \boldsymbol{h}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bx}(\tau) d\tau $$

$$ 
\boldsymbol{h_{k+1}} = e^{\boldsymbol{A}\Delta} 
\left[ e^{\boldsymbol{A}k\Delta} \boldsymbol{h}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bx}(\tau) d\tau\right] +
\int_{k\Delta}^{(k+1)\Delta} e^{\boldsymbol{A}\left[(k+1)\Delta-\tau\right]} \boldsymbol{Bx}(\tau) d\tau = $$

We substitute the expression for $`\boldsymbol{h_k}`$ and take into account that $`\boldsymbol{x}=const`$ within the interval $`\Delta`$:

$$ 
= e^{\boldsymbol{A}\Delta} \boldsymbol{h_k} + \left[ \int_0^{\Delta} e^{\boldsymbol{A}\nu} d\nu \right] \boldsymbol{Bx_k} = 
e^{\boldsymbol{A}\Delta} \boldsymbol{h_k} + \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{Bx_k}
$$

#### Alternative way
Let's write the equation directly in discrete form $`\Delta \Rightarrow \boldsymbol{x_k} = \boldsymbol{x}(k \Delta), \boldsymbol{h_k} = \boldsymbol{h}(k \Delta)`$:

$$ \frac{\boldsymbol{h_{k+1}}-\boldsymbol{h_k}}{\Delta} = \boldsymbol{Ah_k}+\boldsymbol{Bx_k} $$

$$ \boldsymbol{h_{k+1}} = (\boldsymbol{I} + \boldsymbol{A}\Delta)\boldsymbol{h_k} + \Delta \boldsymbol{Bx_k} $$

In the first approximation $e^{\boldsymbol{A}\Delta} \approx \boldsymbol{I} + \boldsymbol{A} \Delta$ or $\frac{1}{\boldsymbol{A}}(e^{\boldsymbol{A}\Delta} - \boldsymbol{I}) \approx \Delta$, then:

$$ \boldsymbol{h_{k+1}} = e^{\boldsymbol{A}\Delta} \boldsymbol{h_k} + \frac{1}{\boldsymbol{A}}(e^{\boldsymbol{A}\Delta} - \boldsymbol{I}) \boldsymbol{Bx_k} $$

Thus, we obtain a discrete SSM model:

$$\left\{ \begin{array}{lcl}
\boldsymbol{h_k} = \overline{\boldsymbol{A}} \boldsymbol{h_{k-1}} + \overline{\boldsymbol{B}} \boldsymbol{x_k}\\ 
\boldsymbol{y_k} = \boldsymbol{\overline{C} h_k} + \overline{\boldsymbol{D}} \boldsymbol{x_k}\\ 
\\
\boldsymbol{\overline{A}} = e^{\boldsymbol{A}\Delta}\\
\boldsymbol{\overline{B}} = \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{B} \approx \Delta \boldsymbol{B}\\
\end{array} \right$$

If we expand the exponent in the parameter $\boldsymbol{\overline{B}}$ to the first order, a very fortunate simplification occurs. Therefore, the authors neglect the accuracy of this, not the most important, parameter in favor of reducing computations:

$`\boldsymbol{x_k}`$ - input to the model,
$`\boldsymbol{y_k}`$  - output of the model,
$`\boldsymbol{h_k}`$ - hidden state or memory of the model,
$`\boldsymbol{\overline{A}}`$ - main parameter responsible for how we transform memory over time - or memory retention parameter,
$`\boldsymbol{\overline{B}}`$ - input transformation parameter,
$`\boldsymbol{\overline{C}}`$ - output transformation parameter,
$`\boldsymbol{\overline{D}}`$ - a kind of skip connection or skip parameter,
$`\Delta`$ - discretization step.

In the simplest case, we have the following dimensions:

$$ \boldsymbol{\overline{A}} (N, N), \\;
\boldsymbol{\overline{B}} (N, 1), \\;
\boldsymbol{\overline{C}} (1, N), \\;
\boldsymbol{\overline{D}} (1, 1), \\;
\boldsymbol{x_k} (1, 1), \\;
\boldsymbol{y_k} (1, 1), \\;
\boldsymbol{h_k}(N, 1), \\;
\Delta = const
$$

Thus, we have obtained a simple recurrent system, while retaining all the mathematical power of the state space. You can gain intuition about the standard SSM [here](https://srush.github.io/annotated-s4/).


## Selective state space model

The distinctive feature of Mamba compared to previous deep SSMs in this branch of evolution lies in the addition of selectivity. In other words, we want only significant values from all $`\boldsymbol{h_i} [i \lt k]`$ to be included in the hidden state $`\boldsymbol{h_k}`$, while the rest are filtered out.

### Glossary
$`N`$ - hidden state dimensionality <br/>
$`L`$ - input sequence length <br/>
$`b`$ - batch size <br/>
$`d`$ - model depth <br/>
$`E=2`$ - expansion factor <br/>
$`d_{in} = Ed`$ - model depth in the mamba block <br/>
$`A,B,C,D`$ - SSM parameters <br/>
$`\Delta`$ - discretization step size <br/>
$`\Delta_R = \frac{d}{16}`$ - projection dimensionality <br/>


### Parametrization
So, to allow the model to focus attention on specific elements of the input sequence, let's make three parameters dependent on the input:

$$ \boldsymbol{B} = \boldsymbol{xW_B}, \\; \boldsymbol{C} = \boldsymbol{xW_C}, \\; \Delta = Softplus[\boldsymbol{xW_{\Delta1} W_{\Delta2}}+\Delta_{bias}]$$

The parameter $`\Delta`$ governs the balance between how much to focus or ignore the current input signal. A large $`\Delta`$ resets the state $`\boldsymbol{h_k}`$ and focuses on the current input $`\boldsymbol{x_k}`$, while a small $`\Delta`$ maintains the state and ignores the current input. Parameters $`\boldsymbol{B}`$ and $`\boldsymbol{C}`$ allow for finer control, determining whether to incorporate the input $`\boldsymbol{x_k}`$ into the state $`\boldsymbol{h_k}`$ or the state into the output $`\boldsymbol{y_k}`$.

$`\boldsymbol{A}`$ and $`\boldsymbol{D}`$ remain independent of the input but become parameters themselves. $`\boldsymbol{A}`$ will be stored in logarithmic form $`\boldsymbol{A_{log}}`$ (*see [S4D](https://arxiv.org/pdf/2206.11893.pdf) initialization*):

$$ \boldsymbol{A} = -\exp^{\boldsymbol{A_{log}}}$$

Here and below, all exponents and logarithms are element-wise. Thus, the trainable parameters for the selective block are:

$$ 
\boldsymbol{A_{log}}(d_{in}, N), \boldsymbol{W_{B}}(d_{in}, N), \boldsymbol{W_{C}}(d_{in}, N), \boldsymbol{D}(d_{in}), \boldsymbol{W_{\Delta1}}(d_{in}, \Delta_R), \boldsymbol{W_{\Delta2}}(\Delta_R, d_{in}), \Delta_{bias}(d_{in})
$$

Let's introduce the remaining parameters that will be used in the architecture:

$$
\boldsymbol{W_{in}}(d, 2d_{in}), \boldsymbol{W_{out}}(d_{in}, d), 
\boldsymbol{W_{emb}}(vocab\\:size, d)=\boldsymbol{W_{vocab}}^T, \boldsymbol{W_{conv1d}}(d_{in}, 1, K)
$$

### Parameters initialization

Each of the parameters described above is initialized differently:

$$ 
\boldsymbol{A_{log}} = \ln
\begin{pmatrix}
1 & 2 & 3 & ... & N\\
1 & 2 & 3 & ... & N\\
  &   &...
\end{pmatrix}, \\; \boldsymbol{D} = \overline{\boldsymbol{1}}, \\; \Delta_{bias} = Softplus^{-1}\left[Uniform(10^{-3}, 10^{-1}) \right]
$$

The parameter $`\boldsymbol{W_{conv1d}}`$ is initialized using the standard initialization of a **conv1d** layer with **bias=True**, while all remaining weights are initialized using a **Linear** layer with **bias=False**.

### Selective SSM inference with Hardware-aware State Expansion

*Figure 1: Structure of the Selective SSM block ([Mamba](https://arxiv.org/abs/2312.00752)).*
![SSSM](https://github.com/Kirill-Shokhin/Research/assets/46619252/8485b8db-b090-4324-98df-ed6f92badfe9)

$$\boldsymbol{x}(b, L, d_{in}), \boldsymbol{h_t}(b, d_{in}, N) \rightarrow \boldsymbol{y}(b, L, d_{in})$$

To accelerate computations, the authors divided the inference of the selective SSM block into two stages: first, preparation (using three-dimensional arrays) on the regular (slower) memory of the graphics card, then discretization and recursive computation (using four-dimensional arrays) in the fast memory of the graphics card:

#### 1) Preparation (GPU HBM):
Returning $`\boldsymbol{A}`$ to its standard form:

$$\boldsymbol{A}(d_{in}, N) = -\exp^{\boldsymbol{A_{log}}}$$

Input projection:
     
 $$\begin{array}{ccc}
 \boldsymbol{B}(b, L, N) = \boldsymbol{xW_B}\\
 \boldsymbol{C}(b, L, N) = \boldsymbol{xW_C}\\
 \Delta(b, L, d_{in}) = Softplus[\boldsymbol{xW_{\Delta1} W_{\Delta2}}+\Delta_{bias}]
 \end{array}$$

#### 2) Selective scan (GPU SRAM):
   
Initialization of the hidden state:

$$\boldsymbol{h_{-1}} = \overline{\boldsymbol{0}}$$

Discretization:
     
$$\begin{array}{ccc}
\boldsymbol{\overline{A}}(b, L, d_{in}, N) = e^{\Delta \boldsymbol{A}}\\
\boldsymbol{\overline{B}x}(b, L, d_{in}, N) = \Delta \boldsymbol{Bx}
\end{array}$$


In the loop over $`t`$ along the $`L`$ axis (for each token), computation of all hidden states $`\boldsymbol{h}`$ and their corresponding outputs $`\boldsymbol{y}`$:
   
$$\begin{array}{lcl}
\boldsymbol{h_t} = \overline{\boldsymbol{A_t}} \boldsymbol{h_{t-1}} + \boldsymbol{(\overline{B} x)_t}\\ 
\boldsymbol{y_t} = \boldsymbol{C_t h_t} + \boldsymbol{Dx_t}\\ 
\end{array}$$

## Mamba Architecture

![Mamba схемы](https://github.com/Kirill-Shokhin/Research/assets/46619252/15b732bd-da9a-44db-85a6-ed3e77faf6e3)
*Figure 2: Mamba's Architecture Overview*. ($`\boldsymbol{x}(b,L,d), \boldsymbol{x_{in}}(b,L,d_{in}), \boldsymbol{W_{in}}(d,d_{in})`$ - for each branch, $`\sigma = SiLU`$)

### Mamba
The architecture of Mamba doesn't differ much from the Transformer's:
1) At the input we have a sequence of length $`L`$, which can represent whether textual tokens or image patches.
2) We vectorize the elements of the sequence with the embedding matrix $`\boldsymbol{W_{emb}}`$, obtaining the desired $`\boldsymbol{x}(b, L, d)`$.
3) We pass it through $`n_{layers}`$ Mamba layers, while preserving its dimensionality.
4) We return the dimensionality to $`(b, L, vocab\;size)`$ by matrix-multiplying it with $`\boldsymbol{W_{vocab}=W_{emb}^T}`$ - the same matrix as used at the input.
5) Finally, we obtain probabilities for each token in the vocabulary.

### Mamba Layer
The Mamba layer consists of:

1) Layer normalization
2) The Mamba block itself
3) Skip connection

### Mamba Block
The principle of the block is based on a **gated MLP**, which, with the help of an additional branch comprising a linear layer, activation function, and subsequent element-wise multiplication, can control the flow of information in the main branch. This mechanism determines which part should be preserved and which should be suppressed.

Along the main branch, there is what is called an **inverted bottleneck**:

* Expansion $`(\boldsymbol{W_{in}}) \rightarrow`$ **depthwise convolution** (In this case, it's a one-dimensional) $`\rightarrow`$ projection $`(\boldsymbol{W_{out}})`$,

with the addition of activation and the core block - **selective SSM** from the previous section.

## Conclusion
The Mamba model has successfully inherited key characteristics from Transformers, such as attention to context and multimodality, while opening new perspectives for future development. Mamba's ability to work efficiently across domains, especially in modalities where large amounts of context need to be considered, such as genomics, audio and video, sets it apart from cutting-edge developments.

Although this review focuses solely on the mathematical aspects of the new approach, the results show that Mamba could be a powerful candidate for a new general multimodal backbone. Details about the synthetic tests, results and comparisons in the areas of LLM, audio and genomics are available in the [original article](https://arxiv.org/pdf/2312.00752.pdf).

Have an interesting 2024 for us!

## Additional materials
* Official [Mamba Repository](https://github.com/state-spaces/mamba)
* Minimal [PyTorch implementation in a single file](https://github.com/johnma2006/mamba-minimal/blob/master/model.py) - recommended for consolidating
