import numpy.random
import warnings

import theano.sparse
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

<<<<<<< HEAD
from pylearn2.costs.cost import Cost
from pylearn2.space import NullSpace

class GSNFriendlyCost(Cost):
    @staticmethod
    def cost(target, output):
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        return self.cost(X, model.reconstruct(X))

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())

class MeanSquaredReconstructionError(GSNFriendlyCost):
    @staticmethod
    def cost(a, b):
        return ((a - b) ** 2).sum(axis=1).mean()

class MeanBinaryCrossEntropy(GSNFriendlyCost):
    @staticmethod
    def cost(target, output):
        return tensor.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

class SampledMeanBinaryCrossEntropy(Cost):
    """
    CE cost that goes with sparse autoencoder with L1 regularization on activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling
    """
    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.one_ratio = ratio

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense = theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1,
                                            prob=self.one_ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P>0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        reg_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        # params = model.get_params()
        # W = params[2]

        # there is a numerical problem when using
        # tensor.log(1 - model.reconstruct(X, P))
        # Pascal fixed it.
        before_activation = model.reconstruct_without_dec_acti(X, P)

        cost = ( 1 * X_dense *
                 tensor.log(tensor.log(1 + tensor.exp(-1 * before_activation))) +
                 (1 - X_dense) *
                 tensor.log(1 + tensor.log(1 + tensor.exp(before_activation)))
               )

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * reg_units

        return cost

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


class SampledMeanSquaredReconstructionError(MeanSquaredReconstructionError):
    """
    mse cost that goes with sparse autoencoder with L1 regularization on activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling
    """
    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.ratio = ratio

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense=theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1, prob=self.ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P>0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        L1_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        #params = model.get_params()
        #W = params[2]
        #L1_weights = theano.tensor.abs_(W).sum()

        cost = ((model.reconstruct(X, P) - X_dense) ** 2)

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * L1_units

        return cost

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())

#class MeanBinaryCrossEntropyTanh(Cost):
#     def expr(self, model, data):
#        self.get_data_specs(model)[0].validate(data)
#        X = data
#        X = (X + 1) / 2.
#        return (
#            tensor.xlogx.xlogx(model.reconstruct(X)) +
#            tensor.xlogx.xlogx(1 - model.reconstruct(X))
#        ).sum(axis=1).mean()
#
#    def get_data_specs(self, model):
#        return (model.get_input_space(), model.get_input_source())


class ContractionCost(Cost):

    def expr(self, model, data, ** kwargs):
        from pylearn2.models.autoencoder import ContractiveAutoencoder
        self.get_data_specs(model)[0].validate(data)
        return model.contraction_penalty(data)

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


class WeightDecay(Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    """

    def __init__(self, coeff):
        """
        coeff: L2 coefficient for the L2 penalty on the weight matrix.
        """
        self.coeff = coeff

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        l2_cost = model.get_weight_decay(self.coeff)
        l2_cost.name ='ae_weight_decay'
        return l2_cost

    def get_data_specs(self, model):
        # This cost does not use any data
        return (NullSpace(), '')


class L1WeightDecay(Cost):
    """
    coeff * sum(abs(weights))

    for each set of weights.

    """

    def __init__(self, coeff):
        """
        coeff: L2 coefficient for the L2 penalty on the weight matrix.
        """
        self.coeff = coeff

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        l1_cost = model.get_l1_weight_decay(self.coeff)
        l1_cost.name ='ae_l1_weight_decay'
        return l1_cost

    def get_data_specs(self, model):
        # This cost does not use any data
        return (NullSpace(), '')


