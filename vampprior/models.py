import tensorflow as tf
import tensorflow_probability as tfp

from vampprior.layers import Encoder, Decoder, Sampling, MeanReducer, MinMaxConstraint
from vampprior.probabilities import log_normal_diag, log_normal_standard, log_bernoulli


class VAE(tf.keras.Model):
    def __init__(self, D, L, beta=1e-3, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.D = D
        self.L = L
        self.beta = beta

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]))
        self.mean_reducer = MeanReducer()

    def call(self, inputs):
        mu, logvar = self.encoder(inputs)
        samples = self.sampling((mu, logvar))  # (N, L, D)
        reconstructed = self.decoder(samples)

        # samples have shape (N, L, D) where N is the minibatch size and D the latent var dimension
        #   must be reshaped to (L, N, D) specifically for log_q_phi
        samples_t = tf.transpose(samples, (1, 0, 2))
        # loss due to regularization
        # first addend, corresponding to log( p_lambda (z_phi^l) )
        log_p_lambda = log_normal_standard(samples_t, reduce_dim=2, name='log-p-lambda')

        # second addend, corresponding to log( q_phi (z|x) )
        # where q_phi=N(z| mu_phi(x), sigma^2_phi(x))
        log_q_phi = log_normal_diag(samples_t, mu, logvar, reduce_dim=2, name='log-q-phi')

        regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                               tf.math.reduce_mean(log_p_lambda),
                                               name='reg-loss')
        self.add_loss(self.beta * regularization_loss)

        # # Reconstruction loss - KL
        # rec_loss = - log_bernoulli(tf.transpose(reconstructed, [1, 0, 2, 3]), inputs, reduce_dim=[2, 3],
        #                            name='rec-loss')
        # self.add_loss(tf.math.reduce_mean(rec_loss))

        # return reconstructed
        return self.mean_reducer(reconstructed)

    def generate(self, N):
        normal_standard = tfp.distributions.MultivariateNormalDiag(tf.zeros((self.D,)),
                                                                   tf.ones((self.D,)))
        # samples will have shape (N, D)
        samples = normal_standard.sample([N])
        samples_extended = samples[:, tf.newaxis, :]

        # inputs will have shape (N, 1, D)
        reconstructed = self.decoder(samples_extended)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(reconstructed)


class VampVAE(tf.keras.Model):
    def __init__(self, D, L, C, beta=1e-3, pseudo_init_mean=.5, pseudo_init_std=0.01, **kwargs):
        super(VampVAE, self).__init__(**kwargs)
        self.D = D  # latent dimension
        self.L = L  # MC samples
        self.C = C  # number of pseudo inputs
        self.beta = beta
        self.init_mean = pseudo_init_mean  # pseudo inputs initialization
        self.init_std = pseudo_init_std

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]))
        self.mean_reducer = MeanReducer()

        self.pseudo_inputs = tf.Variable(
            initial_value=tf.random.normal((self.C, inputs_shape[1], inputs_shape[2]), self.init_mean, self.init_std),
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

        lognormal = log_normal_diag(z_expand, pseudo_mean_expand, pseudo_logvar_expand,
                                    reduce_dim=3, name='pseudo-log-normal') - tf.math.log(tf.cast(self.C, tf.float32))
        ln_max = tf.reduce_max(lognormal, axis=2, keepdims=True)  # find max along the C values
        # get average of probabilities over C using log-sum-exp:
        #   https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        log_p_lambda = ln_max + tf.math.log(tf.reduce_sum(tf.math.exp(lognormal - ln_max), 2))

        # Posterior: Normal posterior
        # samples have shape (N, L, D) where N is the mini-batch size and D the latent var dimension
        # mu and logvar have shape (N, D), therefore we need to transpose samples
        samples_t = tf.transpose(samples, (1, 0, 2))
        log_q_phi = log_normal_diag(samples_t, mu, logvar, reduce_dim=2, name='log-q-phi')

        regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                               tf.math.reduce_mean(log_p_lambda),
                                               name='regularization-loss')
        self.add_loss(self.beta * regularization_loss)

        # return reconstructed
        return self.mean_reducer(reconstructed)

    def generate(self, N):

        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs[:N])  # N x D
        samples = self.sampling((pseudo_mean, pseudo_logvar))[:, :1, :]  # N x 1 x D
        # take only the first sample, but keep shape

        # inputs will have shape (N, 1, D)
        reconstructed = self.decoder(samples)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(reconstructed)

    def update_beta(self, epoch):
        # TODO use it
        self.beta.assign((epoch + 1) / self.warmup)


class MixtureVAE():
    # TODO
    pass


class HVAE():
    # TODO
    pass
