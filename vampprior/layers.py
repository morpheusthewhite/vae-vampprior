import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


class Encoder(tf.keras.layers.Layer):
    def __init__(self, D, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.D = D

    def build(self, inputs_shape):
        self.flatten = layers.Flatten(input_shape=(inputs_shape[1], inputs_shape[2]),
                                      name='enc-flatten')

        self.dense0 = layers.Dense(300, name='enc-dense0', activation='sigmoid')
        self.dense1 = layers.Dense(300, name='enc-dense1', activation='sigmoid')

        self.dense_mu = layers.Dense(self.D, name='enc-out-mu')
        self.dense_logvar = layers.Dense(self.D, name='enc-out-lo')

    def call(self, inputs):
        flattened = self.flatten(inputs)

        x = self.dense0(flattened)
        x = self.dense1(x)

        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar


class Sampling(tf.keras.layers.Layer):
    """
    When called returns L samples of dimension D from the gaussians with the
    mu and logvar passed as input, using the reparametrization trick
    """
    def __init__(self, D, L, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.L = L
        self.D = D

        # the standard distribution to be used when sampling
        # needed for the reparametrization trick
        self.normal_standard = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(shape=(self.D,)),
            tf.ones(shape=(self.D,)))

    def call(self, inputs):
        mu, logvar = inputs

        # samples with the reparametrization trick
        # N(0, I) * sigma + mu
        latent_samples = self.normal_standard.sample((self.L, mu.shape[0])) * \
                tf.sqrt(tf.exp(logvar)) + mu

        # the returned samples will have shape (N, L, D)
        # where N is the size of the batch
        return tf.reshape(latent_samples, (-1, self.L, self.D))


class EncoderProb(tf.keras.layers.Layer):
    """
    Layer encoding probabilities. Used in guassian mixture to infer to which
    class the input data belong
    """
    def __init__(self, K, **kwargs):
        super(EncoderProb, self).__init__(**kwargs)
        self.K = K

    def build(self, inputs_shape):
        # inputs will have shape (N, 2*D), a mu and logvar vector
        self.dense0 = layers.Dense(300, name='encProb-dense0', activation='sigmoid')
        self.dense1 = layers.Dense(self.K, name='encProb-out', activation='softmax')

    def call(self, inputs):
        x = self.dense0(inputs)
        logits = self.dense1(x)

        return logits


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_shape, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_shape_ = output_shape

        self.dense0 = layers.Dense(300, name='dec-dense0', activation='sigmoid')
        self.dense1 = layers.Dense(300, name='dec-dense1', activation='sigmoid')
        self.reconstruct = layers.Dense(output_shape[0] * output_shape[1], name='dec-out')

    def build(self, inputs_shape):
        # transform the result into a square matrix
        # the result of a single input will be a (L, M, M) tensor
        # where M is the size of the original image
        self.reshape = layers.Reshape((-1, self.output_shape_[0],
                                       self.output_shape_[1]),
                                      name='dec-out-reshaped')

    def call(self, inputs):
        # inputs will have shape (N, L, D)
        x = self.dense0(inputs)
        x = self.dense1(x)

        reconstructed = self.reconstruct(x)

        # once reshaped it will have shape (N, L, M, M)
        return self.reshape(reconstructed)


class DecoderMixture(tf.keras.layers.Layer):
    """
    Layer decoding distributions of mixture from a D-dimensional sample
    """
    def __init__(self, D, K, **kwargs):
        super(DecoderMixture, self).__init__(**kwargs)
        self.D = D
        self.K = K

    def build(self, inputs_shape):
        self.dense0 = layers.Dense(300, name='dec-dense0', activation='sigmoid')

        # layers for decoding each of the mixture component
        self.dense1_mu_list = []
        self.dense1_logvar_list = []

        for k in range(self.K):
            dense1_mu_k = layers.Dense(self.D, name=f"decMixture-dense1-mu-{k}")
            dense1_logvar_k = layers.Dense(self.D, name=f"decMixture-dense1-logvar-{k}")

            self.dense1_mu_list.append(dense1_mu_k)
            self.dense1_logvar_list.append(dense1_logvar_k)

    def call(self, inputs):
        # TODO handle L > 1
        # inputs will have shape (N, L, D)

        x = self.dense0(inputs)

        mixture_mu = []
        mixture_logvar = []

        for k in range(self.K):
            # compute mu and logvar for each component
            mu_k = self.dense1_mu_list[k](x)
            logvar_k = self.dense1_logvar_list[k](x)

            mixture_mu.append(mu_k)
            mixture_logvar.append(logvar_k)

        return mixture_mu, mixture_logvar


class MeanReducer(tf.keras.layers.Layer):
    """
    Reduce with mean along the L axis. Meant to be used on the result of the
    decoder to aggregate the decoded L samples
    """
    def __init__(self, **kwargs):
        super(MeanReducer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs has shape (N, L, M, M)
        # output will have shape (N, M, M)
        return tf.reduce_mean(inputs, axis=1)


class HEncoder():
    # TODO
    pass


class HDecoder():
    # TODO
    pass


class MinMaxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max, name="min_value-max-constr")


class Clamp:
    def __init__(self, min_value=0., max_value=1.):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x):
        return tf.clip_by_value(x, self.min_value, self.max_value, name='hardtanh')
