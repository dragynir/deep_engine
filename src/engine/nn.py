import random

import numpy as np

from src.engine.core import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, w=None, activation=None):
        self.w = [Value(random.uniform(-1, 1) if w is None else w) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x):
        if isinstance(x, Value):
            x = [x]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act if self.activation is None else getattr(act, self.activation)()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Linear' if self.activation is None else self.activation}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, initializer, **kwargs):

        w = None
        if initializer == 'xavier':  # symmetric functions
            range_w = np.sqrt(6 / (nin + nout))
            w = random.uniform(-range_w, range_w)
        elif initializer == 'he':  # rule etc.
            w = random.uniform(0, 2/nin)

        self.neurons = [Neuron(nin, w=w, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, activations=None, initializer=None):
        if activations is not None:
            assert len(nouts) == len(activations)

        sz = [nin] + nouts
        self.layers = [
            Layer(
                sz[i],
                sz[i + 1],
                activation=None if activations is None else activations[i],
                initializer=initializer,
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
