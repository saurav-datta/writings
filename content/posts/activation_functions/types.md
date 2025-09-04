+++
date = '2024-03-05'
draft = false
title = 'Types'
tags = ['activation-functions']
+++

# Types of activation functions

## Comparison
| **Activation Function**    | **Pros**                                                                 | **Cons**                                                              | **Used In LLMs**                                                                   | **Common Use Cases**                           |
|-----------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------|
| **Sigmoid**                | Smooth, probabilistic output between (0,1)                               | Vanishing gradient, not zero-centered                                | None (not used in modern LLMs)                                                     | Binary classification (non-LLM)               |
| **Softmax**                | Normalized output probabilities                                          | Sensitive to large input values                                     | **All LLMs (e.g., GPT, BERT, LLaMA, T5, Claude, Gemini, PaLM)**                    | Final output/token prediction                 |
| **Hard Sigmoid/Swish**      | Low compute, approximates sigmoid/swish                                 | Less precise                                                         | **MobileBERT, TinyBERT, DistilBERT (quantized/edge LLMs)**                         | Edge/NLP deployment                           |
| **Tanh**                   | Zero-centered, smooth gradient                                           | Vanishing gradients at extremes                                      | Early LSTM-based LLMs (e.g., early Seq2Seq models)                                 | Historical LLMs, RNN-based                     |
| **Dynamic Tanh**           | Adaptive shape, better expressiveness                                   | More parameters, added complexity                                    | Experimental adaptive LLMs (research only)                                         | Meta-learning, adaptive time-series           |
| **ReLU**                   | Efficient, avoids vanishing gradient for positive inputs                 | Dying ReLU problem                                                    | Rare; used in early transformer variants (e.g., Transformer-XL)                    | Legacy LLM components                         |
| **Leaky ReLU**             | Mitigates dying ReLU issue                                               | Slightly more complex than ReLU                                      | Rare in LLMs; minor experimentation (e.g., early GShard)                           | Not typical in production LLMs                |
| **Parametric ReLU (PReLU)**| Learns slope of negative input                                           | Risk of overfitting, adds parameters                                 | Rare, mostly experimental in LLMs                                                  | Not standard in transformer LLMs              |
| **ELU**                    | Smooth for negative inputs, zero-centered                               | Higher compute cost                                                  | Not used in LLMs                                                                   | Not applicable                                 |
| **Swish**                  | Smooth, non-monotonic, better than ReLU in some cases                   | More computation                                                     | **T5, Switch Transformer, GShard**                                                  | NLP transformer blocks                        |
| **GELU**                   | Smooth, noise-tolerant, better convergence                               | Computationally intensive                                            | **GPT-2, GPT-3, GPT-4, BERT, RoBERTa, T5, LLaMA, OPT, BLOOM, Mistral, Falcon**     | Feedforward layers in LLMs                    |

## Sigmoid
```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

## Softmax
```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    res = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    return res
```

## Hard sigmoid/swish
```python
import math

def hard_sigmoid_swish(x):
    res = np.clip((x + 1) / 2, 0, 1)
    return res
```


## Tanh
```python
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
```
    
    
## Dynamic Tanh
```python
def DyTanh(x, gamma, alpha , beta):
     
    alpha_x = alpha * x
    res = gamma * np.tanh(alpha_x) + beta
    
    return res
# alpha is a learnable scalar parameter 
# that allows scaling the input differently based on its range, accounting  for varying x scale
# 
# gamma, beta are learnable, per-channel vector parameters, 
# the same as those used in all normalization layers
# they allow the output to scale back to any scales.
```
* **Reference**
    * [Arxiv-Transformers without Normalization](https://arxiv.org/abs/2503.10622)

* **Pros**
    * Zero-centered, smooth gradient                                

* **Cons**
    * Vanishing gradients at extremes
                                

## ReLU
```python
def relu(x):
    return np.maximum(0, x)
```

## Leaky Relu
```python
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)
```

## Parametric ReLU (PReLU)
```python
def param_relu(x):
    return np.where(x > 0, x, a * x)

# a is learnable
```

## ELU
```python
def elu(x):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

```

## Swish
```python
def elu(x):
    return (x * (1 / (1 + np.exp(-x)))
```

## GELU

```python
def gelu(x):
    res = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

    return res
```



