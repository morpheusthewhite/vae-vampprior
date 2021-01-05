import tensorflow as tf
import tensorflow_probability as tfp

from vampprior.layers import Encoder, Decoder, Sampling, MeanReducer, MinMaxConstraint, \
        EncoderProb, DecoderMixture
from vampprior.probabilities import log_normal_diag, log_normal_standard, log_bernoulli


class VAE(tf.keras.Model):
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

        # TODO test log-bernoulli loss
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

    def update_beta(self, epoch):
        self.beta = min((epoch + 1) / self.warmup * self.max_beta, self.max_beta)


class VampVAE(tf.keras.Model):
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
        self.beta = min((epoch + 1) / self.warmup * self.max_beta, self.max_beta)


class MixtureHVAE(tf.keras.Model):
    def __init__(self, D, L, K, warmup=0, **kwargs):
        super(MixtureHVAE, self).__init__(**kwargs)
        self.D = D
        self.L = L
        self.K = K

        # number of epochs of warmup
        self.warmup = warmup

        if warmup > 0:
            self.beta = tf.Variable(0, trainable=False, dtype=tf.float32)
        else:
            self.beta = tf.Variable(1, trainable=False, dtype=tf.float32)

    def build(self, inputs_shape):
        # encoders for the 3 latent variables
        self.encoder_z = Encoder(self.D, name='encoder-z')
        self.encoder_w = Encoder(self.D, name='encoder-w')
        self.encoder_y = EncoderProb(self.K, name='encoder-y')

        self.sampling = Sampling(self.D, self.L)

        self.decoder_mixture = DecoderMixture(self.D, self.K, name="decoder-mixture")
        self.decoder_x = Decoder((inputs_shape[1], inputs_shape[2]),
                                 name="decoder-x")

        self.reshaper = tf.keras.layers.Reshape((inputs_shape[1], inputs_shape[2]))

        self.mean_reducer = MeanReducer()

    def call(self, inputs):
        # encode both z and w from inputs
        encoded_z = self.encoder_z(inputs)
        encoded_w = self.encoder_w(inputs)

        mu_z, logvar_z = encoded_z
        mu_w, logvar_w = encoded_w

        # sample from the encoded distributions of z and w
        sampled_z = self.sampling(encoded_z)
        sampled_w = self.sampling(encoded_w)

        # concatenate z and w to be passed to the y encoder
        sampled_zw = tf.concat([sampled_z, sampled_w], axis=2)
        y_probs = self.encoder_y(sampled_zw)

        # epsilon for avoiding log explosion in loss
        # TODO: eventually remove epsilon
        eps = 1e-20

        y_logprobs = tf.math.log(eps + y_probs)

        # decode mixtures (needed for loss later) and output x
        mixture_mu_list, mixture_logvar_list = self.decoder_mixture(sampled_w)
        reconstructed = self.decoder_x(sampled_z)

        #
        # losses
        #

        # probability p(z|x)
        # z_distributions = tfp.distributions.MultivariateNormalDiag(mu_z, tf.sqrt(tf.exp(logvar_z)))
        # log_q_z = tf.linalg.diag_part(tf.math.log(eps + z_distributions.prob(sampled_z)))

        # sampled_z has shape (N, L, D)
        # mu_z and logvar_z have shape (N, D)
        log_q_z_l = log_normal_diag(tf.transpose(sampled_z, (1,0,2)),
                                    mu_z, logvar_z, reduce_dim=2)
        # sum over L
        # TODO: is it correct? probably not, a sum of log does not make senses
        log_q_z = tf.reduce_sum(log_q_z_l, axis=0)

        # mean over L axis
        mean_sampled_z = tf.reduce_mean(sampled_z, axis=1)
        mean_y_probs = tf.reduce_mean(y_probs, axis=1)

        # array of K elements
        # where each elements is the (N,) tensor representing the
        # probability that mean_sampled_z belongs to the k-th mixture
        components_prob = []
        for k in range(self.K):
            # mixture_mu_list[k] and mixture_logvar_list[k] have shape (L, N, D)
            # TODO: use custom function instead, without it exploding

            k_normal = tfp.distributions.MultivariateNormalDiag(mixture_mu_list[k],
                                                                tf.sqrt(tf.exp(mixture_logvar_list[k])))
            components_prob.append(tf.linalg.diag_part(k_normal.prob(mean_sampled_z)))

            # k_component_prob_l = log_normal_diag(mean_sampled_z,
            #                                      tf.transpose(mixture_mu_list[k], (1,0,2)),
            #                                      tf.transpose(mixture_logvar_list[k], (1,0,2)),
            #                                      reduce_dim=2)
            # components_prob.append(tf.reduce_sum(k_component_prob_l, axis=0))

        # create a single tensor of shape (N, K) and multiply by inferred probs
        stacked_probs = tf.stack(components_prob, axis=1) * mean_y_probs

        log_p_z = tf.reduce_sum(tf.math.log(eps + stacked_probs), axis=1)

        # loss regularizing inferred z to one of the mixture components
        E_loss_1 = log_q_z - log_p_z

        # KL(q(w|x)||p(w))
        # loss to regularize w as a Normal standard

        # KL_w = tfp.distributions.MultivariateNormalDiag(mu_w,
        #                                                 tf.sqrt(tf.exp(logvar_w))).log_prob(sampled_w) - \
        #        tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu_w),
        #                                                tf.ones_like(logvar_w)).log_prob(sampled_w)

        KL_w = log_normal_diag(tf.transpose(sampled_w, (1,0,2)), mu_w, logvar_w, reduce_dim=2) - \
                log_normal_standard(tf.transpose(sampled_w, (1,0,2)), reduce_dim=2)

        # E[KL(q(y|wz)||p(y))]
        # loss to regularize y as a uniform categorical
        KL_y = tf.reduce_sum(y_logprobs - tf.math.log(1/self.K), axis=1)

        regularization_loss = tf.reduce_mean(E_loss_1) / 10000+ \
                tf.reduce_mean(KL_w) + tf.reduce_mean(KL_y)
        # regularization_loss = tf.reduce_mean(KL_w) + tf.reduce_mean(KL_y)
        # regularization_loss = tf.reduce_mean(KL_w)

        self.add_loss(tf.multiply(self.beta, regularization_loss))

        return self.reshaper(self.mean_reducer(reconstructed))

    def generate(self, N):
        normal_dist = tfp.distributions.MultivariateNormalDiag(tf.zeros(self.D,),
                                                               tf.ones(self.D,))
        # samples will have shape (N, D)
        w_samples = normal_dist.sample([N])

        mixture_mu_list, mixture_logvar_list = self.decoder_mixture(w_samples)
        mixture_samples = []

        # sample from each of the components of the mixture
        for k in range(self.K):
            k_mixture_sample = tfp.distributions.MultivariateNormalDiag(
                mixture_mu_list[k],
                tf.sqrt(tf.exp(mixture_logvar_list[k]))
            ).sample(1)

            mixture_samples.append(k_mixture_sample)

        mixture_samples = tf.stack(mixture_samples, axis=1)
        reconstructed = self.decoder_x(mixture_samples)

        return tf.reshape(reconstructed, (N*self.K, 28, 28))[:N]

    def update_beta(self, epoch):
        self.beta.assign((epoch + 1)/self.warmup)


class HVAE():
    # TODO
    pass

