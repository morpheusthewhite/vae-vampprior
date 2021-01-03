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
        self.dense_logvar = layers.Dense(self.D, name='enc-out-lo',
                                         activation=tf.keras.layers.Activation(Clamp(-6., 2.)))  # HardTanh

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


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_shape, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_shape_ = output_shape

        self.dense0 = layers.Dense(300, name='dec-dense0', activation='sigmoid')
        self.dense1 = layers.Dense(300, name='dec-dense1', activation='sigmoid')
        self.reconstruct = layers.Dense(output_shape[0] * output_shape[1], name='dec-out', activation='sigmoid')

    def build(self, inputs_shape):
        # transform the result into a square matrix
        # the result of a single input will be a (L, M, M) tensor
        # where M is the size of the original image
        self.reshape = layers.Reshape((inputs_shape[1],
                                       self.output_shape_[0], self.output_shape_[1]),
                                      name='dec-out-reshaped')

    def call(self, inputs):
        # inputs will have shape (N, L, D)
        x = self.dense0(inputs)
        x = self.dense1(x)

        reconstructed = self.reconstruct(x)

        # once reshaped it will have shape (N, L, M, M)
        return self.reshape(reconstructed)


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
