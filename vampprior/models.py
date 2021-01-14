import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from vampprior.layers import Encoder, Decoder, Sampling, MeanReducer, MinMaxConstraint, HierarchicalEncoder, \
    HierarchicalDecoder
from vampprior.probabilities import log_normal_diag, log_normal_standard, log_bernoulli, log_logistic256


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
        assert X.shape[0] % MB == 0

        # number of batches
        NB = X.shape[0] // MB

        loglikelihoods = []
        for r in range(R):
            loglikelihoods_minibatch = []
            for n in range(NB):
                # starting and ending index of the current minibatch
                minibatch_start, minibatch_end = n * MB, (n + 1) * MB

                minibatch = X[minibatch_start:minibatch_end]
                x_mean, x_logvar, z, z_mu, z_logvar = self.forward(minibatch)

                # TODO change sign, loss is neg-loglikelihood
                loglikelihood = self.loss_fn(minibatch,
                                             x_mean, x_logvar, z, z_mu, z_logvar,
                                             training=False, average=False)

                # append the result of the current minibatch
                loglikelihoods_minibatch.append(loglikelihood)

            loglikelihoods.append(tf.concat(loglikelihoods_minibatch, axis=0))

        loglikelihoods_stacked = tf.stack(loglikelihoods, axis=1)
        loglikelihoods_max = tf.reduce_logsumexp(loglikelihoods_stacked, axis=1) - \
                             np.log(R)

        loglikelihood_mean = tf.reduce_mean(loglikelihoods_max)
        return loglikelihoods_max.numpy(), loglikelihood_mean.numpy()

    def ELBO(self, X, MB=100):
        """
        Calculate ELBO for the given data
        @param MB: minibatch size. Needed to avoid Out Of Memory errors. X.shape[0]
            needs to be divisible by this number
        @return ELBO: the Expected Lower BOund
        """
        assert X.shape[0] % MB == 0

        # number of batches
        NB = X.shape[0] // MB

        elbos = []
        for n in range(NB):

            # starting and ending index of the current minibatch
            minibatch_start, minibatch_end = n*MB, (n+1)*MB

            minibatch = X[minibatch_start:minibatch_end]
            x_mean, x_logvar, z, z_mu, z_logvar = self.forward(minibatch)

            elbo = self.loss_fn(minibatch,
                                x_mean, x_logvar, z, z_mu, z_logvar,
                                training=False, average=True)

            # append the result of the current minibatch
            elbos.append(elbo)

        elbos_stacked = tf.stack(elbos, axis=0)
        return tf.reduce_mean(elbos_stacked).numpy()


class VAE(VAEGeneric):
    def __init__(self, D, L, max_beta=1., warmup=0, binary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.D = D
        self.L = L
        self.max_beta = max_beta
        self.beta = max_beta
        self.warmup = warmup
        self.binary = binary
        self.epoch = 0

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]), binary=self.binary)
        self.mean_reducer = MeanReducer()

    def call(self, inputs, training):
        x_mean, x_logvar, z, z_mean, z_logvar = self.forward(inputs)

        loss = self.loss_fn(inputs, x_mean, x_logvar, z, z_mean, z_logvar, training)
        self.add_loss(loss)

        # (N, 1, M, M) -> (N, M, M)
        if x_logvar is not None:
            return self.mean_reducer(x_mean), self.mean_reducer(x_logvar)
        else:
            return self.mean_reducer(x_mean), x_logvar

    def loss_fn(self, inputs, x_mean, x_logvar, z, z_mean, z_logvar, training=True, average=True):
        """
        Calculate loss for the given inputs.
        @param training: if true beta is considered in the calculation of the total loss
        @param average: if true aggregate loss over inputs by averaging
        """
        # Reconstruction loss - log p(x | z)
        x_mean_t = tf.transpose(x_mean, (1, 0, 2, 3))  # (L, N, M, M)
        if self.binary:
            # TODO check if mean over L must be computed BEFORE the reconstruction loss
            log_p_theta = log_bernoulli(x_mean_t, inputs, reduce_dim=[2, 3], name='log_p_theta')
        else:
            x_logvar_t = tf.transpose(x_logvar, (1, 0, 2, 3))
            log_p_theta = log_logistic256(inputs, x_mean_t, x_logvar_t, reduce_dim=[2, 3], name='log_p_theta')

        # samples have shape (N, L, D) where N is the minibatch size and D the latent var dimension
        #   must be reshaped to (L, N, D) specifically for log_q_phi
        samples_t = tf.transpose(z, (1, 0, 2))

        # loss due to regularization
        # first addend, corresponding to log( p_lambda (z_phi^l) )
        log_p_lambda = log_normal_standard(samples_t, reduce_dim=2, name='log-p-lambda')

        # second addend, corresponding to log( q_phi (z|x) )
        # where q_phi=N(z| mu_phi(x), sigma^2_phi(x))
        log_q_phi = log_normal_diag(samples_t, z_mean, z_logvar, reduce_dim=2, name='log-q-phi')

        if average:
            rec_loss = - tf.math.reduce_mean(log_p_theta, name='rec-loss')
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                                   tf.math.reduce_mean(log_p_lambda),
                                                   name='reg-loss')
        else:
            # reduce only along the L axis
            rec_loss = - tf.math.reduce_mean(log_p_theta, name='rec-loss', axis=0)
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi, axis=0),
                                                   tf.math.reduce_mean(log_p_lambda, axis=0),
                                                   name='reg-loss')

        # consider beta only in the training phase
        if training:
            with tf.name_scope("training_scope"):
                tf.summary.scalar("reconstruction_loss", rec_loss, step=self.epoch)
                tf.summary.scalar("regularization_loss", regularization_loss, step=self.epoch)

            loss = rec_loss + self.beta * regularization_loss
        else:
            loss = rec_loss + regularization_loss

        return loss

    def forward(self, X):
        z_mu, z_logvar = self.encoder(X)
        z = self.sampling((z_mu, z_logvar))  # (N, L, D)
        x_mean, x_logvar = self.decoder(z)  # (N, L, M, M)

        return x_mean, x_logvar, z, z_mu, z_logvar

    def generate(self, N):
        normal_standard = tfp.distributions.MultivariateNormalDiag(tf.zeros((self.D,)),
                                                                   tf.ones((self.D,)))
        # samples will have shape (N, D)
        samples = normal_standard.sample([N])
        samples_extended = samples[:, tf.newaxis, :]

        # inputs will have shape (N, 1, D)
        x_mean, _ = self.decoder(samples_extended)  # (N, 1, M, M)
        # aggregation still needed in order to remove the axis 1
        return self.mean_reducer(x_mean)

    def update_beta(self, epoch):

        self.epoch = epoch
        self.beta = min((epoch + 1) / self.warmup * self.max_beta, self.max_beta)
        with tf.name_scope("training_scope"):
            tf.summary.scalar("beta", self.beta, step=epoch)


class VampVAE(VAEGeneric):
    def __init__(self, D, L, C, max_beta=1., warmup=0, pseudo_init_mean=.5, pseudo_init_std=0.01, binary=False, **kwargs):
        super(VampVAE, self).__init__(**kwargs)
        self.D = D  # latent dimension
        self.L = L  # MC samples
        self.C = C  # number of pseudo inputs
        self.max_beta = max_beta
        self.beta = max_beta
        self.warmup = warmup
        self.init_mean = pseudo_init_mean  # pseudo inputs initialization
        self.init_std = pseudo_init_std
        self.binary = binary

    def build(self, inputs_shape):
        self.encoder = Encoder(self.D)
        self.sampling = Sampling(self.D, self.L)
        self.decoder = Decoder((inputs_shape[1], inputs_shape[2]), binary=self.binary)
        self.mean_reducer = MeanReducer()

        self.pseudo_inputs = tf.Variable(
            initial_value=tf.random.normal((self.C, inputs_shape[1], inputs_shape[2]),
                                           self.init_mean, self.init_std),
            trainable=True,
            constraint=MinMaxConstraint(0., 1.)
        )

    def call(self, inputs, training, **kwargs):
        x_mean, x_logvar, samples, z_mean, z_logvar = self.forward(inputs)

        loss = self.loss_fn(inputs, x_mean, x_logvar, samples, z_mean, z_logvar, training)
        self.add_loss(loss)

        # (N, 1, M, M) -> (N, M, M)
        if x_logvar is not None:
            return self.mean_reducer(x_mean), self.mean_reducer(x_logvar)
        else:
            return self.mean_reducer(x_mean), x_logvar

    def loss_fn(self, inputs, x_mean, x_logvar, z, z_mean, z_logvar, training=True, average=True):

        # Reconstruction loss - log p(x | z)
        x_mean_t = tf.transpose(x_mean, (1, 0, 2, 3))  # (L, N, M, M)
        if self.binary:
            log_p_theta = log_bernoulli(x_mean_t, inputs, reduce_dim=[2, 3], name='log_p_theta')
        else:
            x_logvar_t = tf.transpose(x_logvar, (1, 0, 2, 3))
            log_p_theta = log_logistic256(inputs, x_mean_t, x_logvar_t, reduce_dim=[2, 3], name='log_p_theta')

        # loss due to regularization
        # Prior: Vamp Prior
        # 1. get mean and var from pseudo_inputs
        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs)  # C x D
        z_expand = tf.expand_dims(z, 2)  # N x L x 1 x D
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
        samples_t = tf.transpose(z, (1, 0, 2))
        log_q_phi = log_normal_diag(samples_t, z_mean, z_logvar, reduce_dim=2, name='log-q-phi')

        if average:
            rec_loss = - tf.math.reduce_mean(log_p_theta, name='rec-loss')
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi),
                                                   tf.math.reduce_mean(log_p_lambda),
                                                   name='regularization-loss')
        else:
            # reduce only along the L axis
            rec_loss = - tf.math.reduce_mean(log_p_theta, name='rec-loss', axis=1)
            regularization_loss = tf.math.subtract(tf.math.reduce_mean(log_q_phi, axis=1),
                                                   tf.math.reduce_mean(log_p_lambda, axis=1),
                                                   name='regularization-loss')

        # consider beta only if training
        if training:
            loss = rec_loss + self.beta * regularization_loss
        else:
            loss = rec_loss + regularization_loss

        return loss

    def generate(self, N):

        pseudo_mean, pseudo_logvar = self.encoder(self.pseudo_inputs[:N])  # N x D
        samples = self.sampling((pseudo_mean, pseudo_logvar))[:, :1, :]  # N x 1 x D
        # take only the first sample, but keep shape

        # inputs will have shape (N, 1, D)
        x_mean, x_logvar = self.decoder(samples)

        # aggregation still needed as result will have shape (N, 1, M, M)
        # in order to remove the 1-st axis
        return self.mean_reducer(x_mean)

    def forward(self, X):
        # main forward pass
        mu, logvar = self.encoder(X)
        samples = self.sampling((mu, logvar))  # N x L x D
        x_mean, x_logvar = self.decoder(samples)

        return x_mean, x_logvar, samples, mu, logvar

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
            binary=True,
            **kwargs
    ):
        super(HVAE, self).__init__(name=name, **kwargs)
        self.D = D
        self.encoder = HierarchicalEncoder(D=D)
        self.sampling = Sampling(self.D, 1)
        self.mean_reducer = MeanReducer()
        self.binary=binary

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

        # Reconstruction loss - log p(x | z)
        #  import pdb; pdb.set_trace()
        #  x_mean_t = tf.transpose(x_mean, (1, 0, 2, 3))  # (L, N, M, M)
        x_mean_t = x_mean
        KL = tf.math.reduce_mean(KL)

        if self.binary:
            # TODO check if mean over L must be computed BEFORE the reconstruction loss
            log_p_theta = log_bernoulli(x_mean_t, inputs, reduce_dim=[1, 2], name='log_p_theta')
        else:
            x_logvar_t = tf.transpose(x_logvar, (1, 0, 2, 3))
            log_p_theta = log_logistic256(inputs, x_mean_t, x_logvar_t,
                                          reduce_dim=[2, 3], name='log_p_theta')
        rec_loss = - tf.math.reduce_mean(log_p_theta, name='rec-loss')

        self.add_loss(rec_loss + beta * KL)

        return x_mean, x_logvar

    def log_p_z2(self, z2):
        # TODO add vamp prior if-else
        log_prior = log_normal_standard(z2, reduce_dim=1)
        return log_prior

    # generate HVAE
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
