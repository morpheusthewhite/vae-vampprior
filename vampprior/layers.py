import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

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
    def __init__(self, D, L, single=False, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.L = L
        self.D = D
        self.single = single

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
        if self.single:
            return tf.reshape(latent_samples, (-1, self.D))
        else:
            return tf.reshape(latent_samples, (-1, self.L, self.D))


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


class HierarchicalEncoder(layers.Layer):  # MLP block #1   # layer insieme di altri layer
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z) for z1 and z2."""

    # DEFINE LAYER OUTPUT DIMENSIONALITY: "attributes"
    # NOTE: @param units: Positive integer, dimensionality of the output space.
    def __init__(self, D, name="encoder", **kwargs):
        super(HierarchicalEncoder, self).__init__(name=name, **kwargs)
        self.flatten = layers.Flatten(name='enc-flatten')
        # TODO move in build function
        # layers for z2,
        self.dense_1 = layers.Dense(300, activation="sigmoid", name="dense_1")
        self.dense_2 = layers.Dense(300, activation="sigmoid", name="dense_2")
        self.dense_z2_mean = layers.Dense(D, activation="sigmoid", name="dense_z2_mean")
        self.dense_z2_logvar = layers.Dense(D, name="dense_z2_logvar")  # todo: chnge activation HARD tan
        # layers for z1,
        self.dense_z1_z2 = layers.Dense(300, activation="sigmoid", name="dense_z1_z2")
        self.dense_z1_x = layers.Dense(300, activation="sigmoid", name="dense_z1_x")
        self.dense_joint = layers.Dense(300, activation="sigmoid", name="dense_joint")
        self.dense_z1_mean = layers.Dense(D, activation="sigmoid", name="dense_z1_mean")
        self.dense_z1_logvar = layers.Dense(D, name="dense_z1_logvar")  # todo: chnge activation HARD tan  #### CONSTRAINT CLASS
        # sampling
        self.sampling = Sampling(D, 1, single=True)  # don't consider L

        self.D = D

    # CONNECT LAYERS
    def call(self, inputs):
        # q(z2|x)
        flat_inputs = self.flatten(inputs)
        res = self.dense_1(flat_inputs)
        res = self.dense_2(res)
        z2_mean = self.dense_z2_mean(res)
        z2_logvar = self.dense_z2_logvar(res)

        z2 = self.sampling((z2_mean, z2_logvar))  # (N, L, D)

        # q(z1|x,z2)
        res = self.dense_z1_z2(z2)  # (N, L, 300)
        res2 = self.dense_z1_x(flat_inputs)  # (N, 1, 300)
        # var = Lambda(concat_test, name='concat_test')([var_1, var_2])
        concat_input = layers.Concatenate()([res, res2])
        res = self.dense_joint(concat_input)  # concat_input_dim = 600, a_dim = 300
        z1_mean = self.dense_z1_mean(res)
        z1_logvar = self.dense_z1_logvar(res)
        z1 = self.sampling((z1_mean, z1_logvar))
        # import pdb
        # pdb.set_trace()

        return z1_mean, z1_logvar, z1, z2_mean, z2_logvar, z2


class HierarchicalDecoder(layers.Layer):  # MLP block #2   # layer insieme di altri layer
    """Converts z1,z2, the encoded digit vectors, back into a readable digit x."""

    def __init__(self, output_shape, D, name="decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        # decoder: p(z1 | z2)
        self.dense_1 = layers.Dense(300, activation="sigmoid", name="dense_1")
        self.dense_z1new_z2 = layers.Dense(300, activation="sigmoid", name="dense_z1new_z2")
        self.dense_z1new_mean = layers.Dense(D, activation="sigmoid", name="dense_z1new_mean")
        self.dense_z1new_logvar = layers.Dense(D, name="dense_z1new_logvar")
        # sampling
        self.sampling = Sampling(D, 1, single=True)

        # decoder: p(x | z1, z2)
        self.dense_x_z1new = layers.Dense(300, activation="sigmoid", name="dense_x_z1new")
        self.dense_x_z2 = layers.Dense(300, activation="sigmoid", name="dense_x_z2")
        self.dense_joint = layers.Dense(300, activation="sigmoid", name="dense_x_mean")
        self.dense_x_mean = layers.Dense(np.prod(output_shape), activation="sigmoid", name="dense_x_mean")
        self.dense_x_logvar = layers.Dense(np.prod(output_shape), name="dense_x_logvar")  # todo: chnge activation HARD tan
        self.output_shape_ = output_shape

    def build(self, inputs_shape):
        # transform the result into a square matrix
        # the result of a single input will be a (M, M) tensor
        # where M is the size of the original image
        # input shape (N, D)
        self.reshape = layers.Reshape((self.output_shape_[0],
                                       self.output_shape_[1]),
                                      name='dec-out-reshaped')

    def call(self, inputs):
        z1_q, z2_q = inputs
        # decoder: p(z1 | z2)
        res = self.dense_1(z2_q)
        res = self.dense_z1new_z2(res)
        z1_p_mean = self.dense_z1new_mean(res)
        z1_p_logvar = self.dense_z1new_logvar(res)
        # there is no sampling for the new z1_p

        # decoder: p(x | z1, z2)
        res = self.dense_x_z1new(z1_q)
        res2 = self.dense_x_z2(z2_q)

        # joint
        # concat_input = Lambda(concat_test, name='concat_test')([var_1, var_2])
        concat_input = layers.Concatenate()([res, res2])
        joint = self.dense_joint(concat_input)

        # reconstruct X (no sampling)
        x_mean = self.dense_x_mean(joint)
        x_logvar = self.dense_x_logvar(joint)

        x_mean_reshaped = self.reshape(x_mean)
        # FIXME reshape x_logvar eventually when changing loss to log-logistic
        return x_mean_reshaped, x_logvar, z1_p_mean, z1_p_logvar

    def p_z1(self, z2):
        # decoder: p(z1 | z2)
        res = self.dense_1(z2)
        res = self.dense_z1new_z2(res)
        z1_p_mean = self.dense_z1new_mean(res)
        z1_p_logvar = self.dense_z1new_logvar(res)
        return z1_p_mean, z1_p_logvar

    def p_x(self, z1, z2):
        # decoder: p(x | z1, z2)
        res = self.dense_x_z1new(z1)
        res2 = self.dense_x_z2(z2)
        # joint
        # concat_input = Lambda(concat_test, name='concat_test')([var_1, var_2])
        concat_input = layers.Concatenate()([res, res2])
        joint = self.dense_joint(concat_input)

        # reconstruct X (no sampling)
        x_mean = self.dense_x_mean(joint)
        x_logvar = self.dense_x_logvar(joint)

        return x_mean, x_logvar


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
