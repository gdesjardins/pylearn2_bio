import numpy as np

from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import sharedX

class LearningRule():
    """
    A pylearn2 learning rule is an object which computes new parameter values
    given (1) a learning rate (2) current parameter values and (3) the current
    estimated gradient.
    """

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Method called by the training algorithm, which allows LearningRules to
        add monitoring channels.

        monitor: a pylearn2.monitor.Monitor object, to which the rule
        should register additional monitoring channels.
        monitoring_dataset: a Dataset instance of dictionary whose values
        are Dataset objects.
        """
        raise NotImplementedError()

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """

        Parameters
        ----------
        learning_rate: float, learning rate coefficient.
        grads: a dictionary mapping from the model's parameters
        to their gradients.
        lr_scalers: dictionary mapping from the model's parameters
        to a learning rate multiplier.

        Returns
        -------
        A dictionary mapping from the old model parameters, to their new
        values after a single iteration of the learning rule.

        e.g. for standard SGD, one would return `sgd_rule_updates` defined
        below. Note that such a LearningRule object is not implemented, as these
        updates are implemented by default when the `learning_rule` parameter
        of sgd.SGD.__init__ is None.

            sgd_rule_updates = OrderedDict()
            for (param, grad) in grads.iteritems():
                sgd_rule_updates[k] = param - learning_rate * lr_scalers.get(param, 1.) * grad
        """
        raise NotImplementedError()


class Momentum(LearningRule):
    """
    Implements momentum as described in Section 9 of 
    "A Practical Guide to Training Restricted Boltzmann Machines", Geoffrey Hinton.

    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc
    """

    def __init__(self, init_momentum):
        """
        init_momentum: initial value for the momentum coefficient. It remains
        fixed during training unless used with a
        training_algorithms.sgd.MomentumAdjustor extension.
        """
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        monitor.add_channel(
                name='momentum',
                ipt=None,
                val=self.momentum,
                data_specs=(NullSpace(), ''),
                dataset=monitoring_dataset)

    def get_updates(self, learning_rate, grads, lr_scalers=None):

        updates = OrderedDict()

        for (param, grad) in grads.iteritems():
            inc = sharedX(param.get_value() * 0.)
            if param.name is not None:
                inc.name = 'inc_'+param.name
            updated_inc = self.momentum * inc -\
                          learning_rate * lr_scalers.get(param, 1.) * grad
            updates[inc] = updated_inc
            updates[param] = param + updated_inc

        return updates


class MomentumAdjustor(TrainExtension):

    def __init__(self, final_momentum, start, saturate):
        """
            final_momentum: the momentum coefficient to use at the end
                            of learning.
            start: the epoch on which to start growing the momentum coefficient.
            saturate: the epoch on which the moment should reach its final value
        """

        if saturate < start:
            raise TypeError("Momentum can't saturate at its maximum value before it starts increasing.")

        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0

    def on_monitor(self, model, dataset, algorithm):
        if hasattr(algorithm, 'learning_rule'):
            momentum = algorithm.learning_rule.momentum
        else:
            # TODO: remove once training_algorithm.sgd.SGD(init_momentum)
            # is officially deprecated.
            momentum = algorithm.momentum

        if not self._initialized:
            self._init_momentum = momentum.get_value()
            self._initialized = True
        self._count += 1
        momentum.set_value( np.cast[config.floatX](self.current_momentum()))

    def current_momentum(self):
        w = self.saturate - self.start

        if w == 0:
            # saturate=start, so just jump straight to final momentum
            if self._count >= self.start:
                return self.final_momentum
            return self._init_momentum

        alpha = float(self._count - self.start) / float(w)
        if alpha < 0.:
            alpha = 0.
        if alpha > 1.:
            alpha = 1.
        return self._init_momentum * (1.-alpha)+alpha*self.final_momentum




class AdaDelta(LearningRule):
    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    """

    def __init__(self, decay=0.95):
        """
        Parameters
        ----------
        decay: float
            decay rate \rho in Algorithm 1 of the afore-mentioned paper.
        """
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """ TODO: add channels worth monitoring """
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):

        updates = OrderedDict()

        for param in grads.keys():

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)
            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = \
                    self.decay * mean_square_grad +\
                    (1 - self.decay) * T.sqr(grads[param])

            # Compute update
            epsilon = lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = \
                    self.decay * mean_square_dx + \
                    (1 - self.decay) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t
        
        return updates
