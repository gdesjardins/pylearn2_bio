from theano import tensor
import theano.sparse
import warnings
from pylearn2.costs.cost import Cost
import numpy.random
from theano.tensor.shared_randomstreams import RandomStreams

class MeanSquaredReconstructionError(Cost):
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        return ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean()

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


class MeanBinaryCrossEntropy(Cost):
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        return (
            - X * tensor.log(model.reconstruct(X)) -
            (1 - X) * tensor.log(1 - model.reconstruct(X))
        ).sum(axis=1).mean()

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


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
        assert isinstance(model, ContractiveAutoencoder)
        self.get_data_specs(model)[0].validate(data)
        return model.contraction_penalty(data)

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


