import tensorflow as tf
import tensorflow_probability as tfp

from vampprior.layers import Encoder, Decoder, Sampling, MeanReducer
from vampprior.probabilities import log_normal_diag, log_normal_standard


class VAE(tf.keras.Model):
    def __init__(self, D, L, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.D = D
        self.L = L

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]))
        self.mean_reducer = MeanReducer()

    def call(self, inputs):
        mu, logvar = self.encoder(inputs)
        samples = self.sampling((mu, logvar))

        # TODO: improve numerical stability and remove epsilon
        #   to improve decoding performances

        # epsilon for avoiding log explosion in loss
        eps = 1e-18

        # loss due to regularization
        # first addend, corresponding to log( p_lambda (z_phi^l) )
        log_p_lambda = log_normal_standard(samples, reduce_dim=2, name='log-p-lambda')

        # second addend, corresponding to log( q_phi (z|x) )
        # where q_phi=N(z| mu_phi(x), sigma^2_phi(x))
        # samples have shape (N, L, D) where N is the minibatch size and D the latent var dimension
        log_q_phi = log_normal_diag(samples, mu, logvar, reduce_dim=2, name='log-q-phi')

        regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                               tf.math.reduce_mean(log_p_lambda),
                                               name='regularization-loss')
        self.add_loss(regularization_loss)

        reconstructed = self.decoder(samples)

        # return reconstructed
        return self.mean_reducer(reconstructed)

    def generate(self, N):
        normal_standard = tfp.distributions.MultivariateNormalDiag(tf.zeros((self.D,)),
                                                                   tf.ones((self.D,)))
        samples = normal_standard.sample([N])

        # inputs will have shape (N, D)
        reconstructed = self.decoder(samples)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(reconstructed)


class VampVAE(tf.keras.Model):
    def __init__(self, D, L, C, init_mean=0, init_std=0.01, **kwargs):
        super(VampVAE, self).__init__(**kwargs)
        self.D = D  # latent dimension
        self.L = L  # MC samples
        self.C = C  # number of pseudo inputs
        self.init_mean = init_mean  # pseudo inputs initialization
        self.init_std = init_std

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]))
        self.mean_reducer = MeanReducer()

        self.pseudo_inputs = tf.Variable(
            initial_value=tf.random.normal((self.C, tf.reduce_prod(inputs_shape)), self.init_mean, self.init_std),
            trainable=True,
            constraint=MinMaxConstraint(0., 1.)
        )

    def call(self, inputs, **kwargs):
        # main forward pass
        mu, logvar = self.encoder(inputs)
        samples = self.sampling((mu, logvar))  # N x L x D
        reconstructed = self.decoder(samples)

        # loss due to regularization
        # Prior: Vamp Prior
        # 1. get mean and var from pseudo_inputs
        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs)  # C x D
        z_expand = tf.expand_dims(samples, 2)  # N x L x 1 x D
        pseudo_mean_expand = tf.expand_dims(pseudo_mean, 0)  # 1 x C x D
        pseudo_logvar_expand = tf.expand_dims(pseudo_logvar, 0)  # 1 x C x D
        #
        # a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        # a_max, _ = torch.max(a, 1)  # MB x 1
        #
        # # calculte log-sum-exp
        # log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        lognormal = log_normal_diag(z_expand, pseudo_mean_expand, pseudo_logvar_expand,
                                    reduce_dim=3, name='pseudo-log-normal') - tf.math.log(self.C)
        ln_max = tf.reduce_max(lognormal, axis=2, keepdims=True)  # find max along the C values
        # get average of probabilities over C
        log_p_lambda = ln_max + tf.math.log(tf.reduce_sum(tf.math.exp(lognormal - ln_max), 2))

        # Posterior: Normal posterior
        # samples have shape (N, L, D) where N is the minibatch size and D the latent var dimension
        log_q_phi = log_normal_diag(samples, mu, logvar, reduce_dim=2, name='log-q-phi')

        regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                               tf.math.reduce_mean(log_p_lambda),
                                               name='regularization-loss')
        self.add_loss(regularization_loss)

        # return reconstructed
        return self.mean_reducer(reconstructed)

    def generate(self, N):
        normal_standard = tfp.distributions.MultivariateNormalDiag(tf.zeros((self.D,)),
                                                                   tf.ones((self.D,)))
        samples = normal_standard.sample([N])

        # inputs will have shape (N, D)
        reconstructed = self.decoder(samples)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(reconstructed)


class MixtureVAE():
    # TODO
    pass


class HVAE():
    # TODO
    pass


class MinMaxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max, name="min_value-max-constr")