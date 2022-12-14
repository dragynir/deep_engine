import numpy as np



class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self.optim_state = {}
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += 1.0 / (out.data + 1e-16) * out.grad
        out._backward = _backward
        return out

    def __abs__(self):
        out = Value(abs(self.data), (self,), "abs")

        def _backward():
            if out.data == 0:
                g = 0
            else:
                g = out.data / abs(out.data)
            self.grad += g * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "int and float supported only"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def _stable_sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        x_exp = np.exp(x)
        return x_exp / (1.0 + x_exp)

    def sigmoid(self):
        out = Value(self._stable_sigmoid(self.data), (self,), "Sigmoid")

        def _backward():
            sigm = self._stable_sigmoid(out.data)
            self.grad += (sigm * (1.0 - sigm) * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), "Tanh")

        def _backward():
            self.grad += (1.0 - np.tanh(out.data) ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        # topological sort
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
