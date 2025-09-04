+++
date = '2024-03-05'
draft = false
title = 'sigmoid'
tags = ['sigmoid']
+++

# Sigmoid


The sigmoid function can be defined as:

f(x) = 1 / (1 + e^(-x))

Where:

* f(x) is the output of the sigmoid function for input x
* e is the Euler’s number (approximately 2.71828)

⠀ In Python, the sigmoid function can be implemented as follows:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

This function takes an input x and returns the output of the sigmoid function for that input.

## activation-functions

GELU

https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c

SWIGLU

https://deci.ai/blog/evolution-of-modern-transformer-swiglu-rope-gqa-attention-is-all-you-need/#:~:text=SwiGLU%3A%20An%20Enhanced%20Activation%20Function,All%20You%20Need'%20Transformer%20model.
