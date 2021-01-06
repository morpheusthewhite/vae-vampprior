import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from vampprior.layers import Encoder, Decoder, Sampling, MeanReducer, MinMaxConstraint, HierarchicalEncoder, \
    HierarchicalDecoder
from vampprior.probabilities import log_normal_diag, log_normal_standard, log_bernoulli


class VAEGeneric(tf.keras.Model):
    def loglikelihood(self, X, R, MB=100):
        """
        Calculate loglikelihoods for the given data
        @param R: number of iterations over X
        @param MB: minibatch size. Needed to avoid Out Of Memory errors. X.shape[0]
            needs to be divisible by this number
        @return loglikelihoods: array of loglikelihood, each corresponding to one input
        @return loglikelihood_mean: mean of loglikelihoods across passed data
        """
        # number of batches
        NB = X.shape[0] // MB

        loglikelihoods = []
        for r in range(R):
            loglikelihoods_minibatch = []
            for n in range(NB):
                # starting and ending index of the current minibatch
                minibatch_start, minibatch_end = n * MB, (n + 1) * MB

                minibatch = X[minibatch_start:minibatch_end]
                reconstructed, samples, mu, logvar = \
                    self.forward(minibatch)

                loglikelihood = self.loss_fn(minibatch,
                                             mu, logvar, samples, reconstructed,
                                             training=False, average=False)

                # append the result of the current minibatch
                loglikelihoods_minibatch.append(loglikelihood)

            loglikelihoods.append(tf.concat(loglikelihoods_minibatch, axis=0))

        loglikelihoods_stacked = tf.stack(loglikelihoods, axis=1)
        loglikelihoods_max = tf.reduce_logsumexp(loglikelihoods_stacked, axis=1) - \
                             np.log(R)

        loglikelihood_mean = tf.reduce_mean(loglikelihoods_max)
        return loglikelihoods_max, loglikelihood_mean


class VAE(VAEGeneric):
    def __init__(self, D, L, max_beta=1., warmup=0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.D = D
        self.L = L
        self.max_beta = max_beta
        self.beta = max_beta
        self.warmup = warmup

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]))
        self.mean_reducer = MeanReducer()

    def call(self, inputs, training):
        reconstructed, samples, mu, logvar = self.forward(inputs)

        loss = self.loss_fn(inputs, mu, logvar, samples, reconstructed, training)
        self.add_loss(loss)

        return self.mean_reducer(reconstructed)

    def loss_fn(self, inputs, mu, logvar, samples, reconstructed, training=True,
                average=True):
        """
        Calculate loss for the given inputs.
        @param training: if true beta is considered in the calculation of the total loss
        @param average: if true aggregate loss over inputs by averaging
        """
        # samples have shape (N, L, D) where N is the minibatch size and D the latent var dimension
        #   must be reshaped to (L, N, D) specifically for log_q_phi
        samples_t = tf.transpose(samples, (1, 0, 2))

        # loss due to regularization
        # first addend, corresponding to log( p_lambda (z_phi^l) )
        log_p_lambda = log_normal_standard(samples_t, reduce_dim=2, name='log-p-lambda')

        # second addend, corresponding to log( q_phi (z|x) )
        # where q_phi=N(z| mu_phi(x), sigma^2_phi(x))
        log_q_phi = log_normal_diag(samples_t, mu, logvar, reduce_dim=2, name='log-q-phi')

        if average:
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                                   tf.math.reduce_mean(log_p_lambda),
                                                   name='reg-loss')
        else:
            # reduce only along the L axis
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi, axis=0),
                                                   tf.math.reduce_mean(log_p_lambda, axis=0),
                                                   name='reg-loss')

        # TODO test log-bernoulli loss
        # # Reconstruction loss - KL
        # rec_loss = - log_bernoulli(tf.transpose(reconstructed, [1, 0, 2, 3]), inputs, reduce_dim=[2, 3],
        #                            name='rec-loss')
        # self.add_loss(tf.math.reduce_mean(rec_loss))

        # consider beta only in the training phase
        if training:
            loss = self.beta * regularization_loss
        else:
            loss = regularization_loss

        return loss

    def forward(self, X):
        mu, logvar = self.encoder(X)
        samples = self.sampling((mu, logvar))  # (N, L, D)
        reconstructed = self.decoder(samples)

        return reconstructed, samples, mu, logvar

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

    def update_beta(self, epoch):
        self.beta = min((epoch + 1) / self.warmup * self.max_beta, self.max_beta)


class VampVAE(VAEGeneric):
    def __init__(self, D, L, C, max_beta=1., warmup=0, pseudo_init_mean=.5, pseudo_init_std=0.01, **kwargs):
        super(VampVAE, self).__init__(**kwargs)
        self.D = D  # latent dimension
        self.L = L  # MC samples
        self.C = C  # number of pseudo inputs
        self.max_beta = max_beta
        self.beta = max_beta
        self.warmup = warmup
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

    def call(self, inputs, training, **kwargs):
        reconstructed, samples, mu, logvar = self.forward(inputs)

        loss = self.loss_fn(inputs, mu, logvar, samples, reconstructed, training)
        self.add_loss(loss)

        # return reconstructed
        return self.mean_reducer(reconstructed)

    def loss_fn(self, X, mu, logvar, samples, reconstructed,
                training=True, average=True):
        # loss due to regularization
        # Prior: Vamp Prior
        # 1. get mean and var from pseudo_inputs
        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs)  # C x D
        z_expand = tf.expand_dims(samples, 2)  # N x L x 1 x D
        pseudo_mean_expand = tf.expand_dims(pseudo_mean, 0)  # 1 x C x D
        pseudo_logvar_expand = tf.expand_dims(pseudo_logvar, 0)  # 1 x C x D

        lognormal = log_normal_diag(z_expand, pseudo_mean_expand, pseudo_logvar_expand,
                                    reduce_dim=3, name='pseudo-log-normal') - \
                    tf.math.log(tf.cast(self.C, tf.float32))
        ln_max = tf.reduce_max(lognormal, axis=2)  # find max along the C values, shape (N, L)
        # get average of probabilities over C using log-sum-exp:
        #   https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        log_p_lambda = ln_max + tf.math.log(
            tf.reduce_sum(tf.math.exp(lognormal - ln_max[:, :, tf.newaxis]), 2))  # (N, L)

        # Posterior: Normal posterior
        # samples have shape (N, L, D) where N is the mini-batch size and D the latent var dimension
        # mu and logvar have shape (N, D), therefore we need to transpose samples
        samples_t = tf.transpose(samples, (1, 0, 2))
        log_q_phi = log_normal_diag(samples_t, mu, logvar, reduce_dim=2, name='log-q-phi')

        if average:
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                                   tf.math.reduce_mean(log_p_lambda),
                                                   name='regularization-loss')
        else:
            # reduce only along the L axis
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi, axis=1),
                                                   tf.math.reduce_mean(log_p_lambda, axis=1),
                                                   name='regularization-loss')

        # consider beta only if training
        if training:
            loss = self.beta * regularization_loss
        else:
            loss = regularization_loss

        return loss

    def generate(self, N):

        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs[:N])  # N x D
        samples = self.sampling((pseudo_mean, pseudo_logvar))[:, :1, :]  # N x 1 x D
        # take only the first sample, but keep shape

        # inputs will have shape (N, 1, D)
        reconstructed = self.decoder(samples)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(reconstructed)

    def forward(self, X):
        # main forward pass
        mu, logvar = self.encoder(X)
        samples = self.sampling((mu, logvar))  # N x L x D
        reconstructed = self.decoder(samples)

        return reconstructed, samples, mu, logvar

    def update_beta(self, epoch):
        self.beta = min((epoch + 1) / self.warmup * self.max_beta, self.max_beta)


class MixtureVAE():
    # TODO
    pass


class HVAE(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
            self,
            D,
            name="autoencoder",
            **kwargs
    ):
        super(HVAE, self).__init__(name=name, **kwargs)
        self.D = D
        self.encoder = HierarchicalEncoder(D=D)
        self.sampling = Sampling(self.D, 1)

    def build(self, input_shape):
        self.decoder = HierarchicalDecoder((input_shape[1], input_shape[2]), D=self.D)

    def call(self, inputs, **kwargs):
        # variational dist from encoder
        z1_q_mean, z1_q_logvar, z1_q, z2_q_mean, z2_q_logvar, z2_q = self.encoder(inputs)

        x_mean, x_logvar, z1_p_mean, z1_p_logvar = self.decoder((z1_q, z2_q))

        # KL
        log_p_z1 = log_normal_diag(z1_q, z1_p_mean, z1_p_logvar, reduce_dim=1)
        log_q_z1 = log_normal_diag(z1_q, z1_q_mean, z1_q_logvar, reduce_dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_normal_diag(z2_q, z2_q_mean, z2_q_logvar, reduce_dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
        beta = 1e-3
        self.add_loss(beta * KL)
        return x_mean

    def log_p_z2(self, z2):
        # TODO add vamp prior if-else
        log_prior = log_normal_standard(z2, reduce_dim=1)
        return log_prior

    def generate(self, N):
        normal_standard = tfp.distributions.MultivariateNormalDiag(tf.zeros((self.D,)),
                                                                   tf.ones((self.D,)))
        # z2 will have shape (N, D)
        z2 = normal_standard.sample([N])

        # z1 from z2 with partial decoding
        z1_p_mean, z1_p_logvar = self.decoder.p_z1(z2)

        z1 = self.sampling((z1_p_mean, z1_p_logvar))  # (N, D)

        x_mean, x_logvar = self.decoder.p_x(z1, z2)

        # aggregation still needed as result will have shape (N, M, M)
        # in order to remove the 1-st axis
        return x_mean
