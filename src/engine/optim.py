from src.engine.core import Value


class SGD:
    def __init__(
        self,
        parameters,
        momentum=0,
        dampening=0,
        weights_decay=0,
        nesterow=False,
        maximize=False,
    ):
        self._parameters = parameters
        self._momentum = momentum
        self._weights_decay = weights_decay
        self._nesterow = nesterow
        self._dampening = dampening
        self._maximize = maximize

    def step(self, learning_rate):
        for p in self._parameters:
            gt = p.grad
            w = p.data

            # apply weights decay
            gt = gt + self._weights_decay * w

            if 'bgt' in p.optim_state:
                p.optim_state['bgt'] = (1 - self._dampening) * gt + self._momentum * p.optim_state['bgt']
            else:
                p.optim_state['bgt'] = gt

            if self._nesterow:
                gt = gt + self._momentum * p.optim_state['bgt']
            else:
                gt = p.optim_state['bgt']

            if self._maximize:
                w += learning_rate * gt
            else:
                w -= learning_rate * gt
            p.data = w


def l1_regularization(parameters, alpha=1e-4) -> Value:
    return alpha * sum(abs(p) for p in parameters)


def l2_regularization(parameters, alpha=1e-4) -> Value:
    return alpha * sum((p * p for p in parameters))
