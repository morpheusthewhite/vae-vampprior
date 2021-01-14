import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
import datetime

from vampprior.models import VAE, VampVAE, HVAE
from vampprior.datasets import load_frey, load_fashion_mnist

parser = argparse.ArgumentParser(description='VAE+VampPrior')
# Model params
parser.add_argument('--model-name', '-mn', type=str, default='vae', metavar='model_name',
                    help='model name: vae, vamp', choices=['vae', 'vamp', 'hvae'])
parser.add_argument('-C', '--pseudo-inputs', type=int, default=500, metavar='C', dest='C',
                    help='number of pseudo-inputs with vamp prior')
parser.add_argument('-D', type=int, default=40, metavar='D',
                    help='number of stochastic hidden units, i.e. z size (same for z1 and z2 with HVAE)')
parser.add_argument('--dataset', '-ds', type=str, default='mnist', metavar='dataset',
                    help='used dataset: mnist, frey', choices=['mnist', 'frey', 'fashion'])
# Training params
parser.add_argument('--epochs', '-e', type=int, default=1, metavar='epochs',
                    help='number of epochs')
parser.add_argument('-bs', '--batch-size', type=int, default=100, metavar='batch_size',
                    help='size of training mini-batch')
parser.add_argument('-L', type=int, default=1, metavar='L',
                    help='number of MC samples')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, metavar='lr', dest='lr',
                    help='learning rate')
parser.add_argument('-wu', '--warm-up', type=int, default=0, metavar='warmup', dest='warmup',
                    help='number of warmup epochs')
parser.add_argument('--max-beta', type=float, default=1., metavar='max_beta',
                    help='maximum value of the regularization loss coefficient')
# Debugging params
parser.add_argument('-tb', '--tensorboard', action='store_true', dest='tb',
                    help='save training log in ./ for tensorboard inspection')
parser.set_defaults(tb=False)
parser.add_argument('-d', '--debug', action='store_true', dest='debug',
                    help='show images')
parser.set_defaults(debug=False)

args = parser.parse_args()

log_dir = './logs'  # save tensorboard logs in the current dir


def train_test_vae(vae, x_train, x_test, epochs, batch_size,
                   model_name, warmup, args, show=True, tb=False):
    """
    Train model and visualize result
    """

    # set grey colormap
    plt.set_cmap('Greys')

    # add callbacks for tensorboard or/and warmup beta update
    callbacks = None
    if tb or warmup > 0:
        callbacks = []
    if tb:
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    if warmup > 0:
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: vae.update_beta(epoch)))

    # TRAINING =======================
    history = vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_test, x_test), callbacks=callbacks)
    # ================================

    # create folder to save results if it does not exists
    res_dir = "results"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    # create folder dedicated to this single experiment
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(os.path.join(res_dir, current_time))
    # store current args for later inspection
    with open(os.path.join(res_dir, current_time, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Plot training/val loss
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    epochs = np.arange(len(train_losses)) + 1
    with plt.style.context('bmh'):
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label='train')
        ax.plot(epochs, val_losses, label='val')
        ax.set(xlabel='epoch',  # ylabel='loss (neg-LB)',
               title='Training over epochs')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        fig.savefig(os.path.join(res_dir, current_time, f"{model_name}-losses.png"))

    print("Now testing reconstruction")
    reconstructions, _ = vae(x_test[:5])

    plt.figure().suptitle(f"Reconstruction for {model_name}")
    for i, reconstruction in enumerate(reconstructions):
        plt.subplot(2, 5, 1 + i)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(x_test[i])

        plt.subplot(2, 5, 6 + i)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(reconstruction)

    if show:
        plt.show()
    else:
        # save reconstruction
        plt.savefig(os.path.join(res_dir, current_time, f"{model_name}-reconstructions.png"))

    print("Now testing generation")
    generations = vae.generate(10)

    plt.figure().suptitle(f"Generations for {model_name}")
    for i, generation in enumerate(generations):
        plt.subplot(2, 5, 1 + i)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(generation)

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(res_dir, current_time, f"{model_name}-generations.png"))

    if "vamp" in model_name:
        # visualize pseudoinputs only for vamp-priors models
        print("Retrieving pseudo-inputs")

        # take just 10 of them
        assert vae.C > 10
        pseudo_inputs = vae.pseudo_inputs[:10].numpy()

        plt.figure().suptitle(f"Pseudoinputs for {model_name}")
        for i, pseudo_input in enumerate(pseudo_inputs):
            plt.subplot(2, 5, 1 + i)
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.imshow(pseudo_input)

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(res_dir, current_time, f"{model_name}-pseudoinputs.png"))

    print("Estimating likelihood")
    loglikelihoods, loglikelihood_mean = vae.loglikelihood(x_test, 4)

    print(f"Loglikelihood: {loglikelihood_mean}")

    with plt.style.context('ggplot'):
        plt.figure().suptitle(f"Loglikelihood histogram for {model_name}")
        plt.hist(loglikelihoods, bins=100)

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(res_dir, current_time, f"{model_name}-loglikelihood-hist.png"))

    return vae.ELBO(x_test)


def main():
    binary = False
    if args.dataset == "mnist":
        mnist = tf.keras.datasets.mnist
        (mnist_train, _), (mnist_test, _) = mnist.load_data()

        binary = True
        # simple workaround for working with binary data
        # where each pixel is either 0 or 1
        x_train = np.array((mnist_train / 255.) > 0.5, dtype=np.float32)
        x_test = np.array((mnist_test / 255.) > 0.5, dtype=np.float32)
    elif args.dataset == "frey":
        # freyfaces dataset, only continous
        x_train, x_test = load_frey(MB=args.batch_size)
    elif args.dataset == "fashion":
        x_train, x_test = load_fashion_mnist()
    else:
        raise Exception("Wrong dataset name")

    if args.model_name == 'vae':
        # simple VAE, normal standard prior
        model = VAE(args.L, D=args.D, warmup=args.warmup, max_beta=args.max_beta,
                    binary=binary, name=args.model_name)
    elif args.model_name == 'vamp':
        # VAE with Vamp prior
        model = VampVAE(args.L, args.C, D=args.D, warmup=args.warmup,
                        max_beta=args.max_beta, binary=binary, name=args.model_name)
    elif args.model_name == 'hvae':
        model = HVAE(D=args.D, warmup=args.warmup, max_beta=args.max_beta,
                     binary=binary, name=args.model_name)
    else:
        raise Exception('Wrong model name!')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr))

    elbo = train_test_vae(model, x_train, x_test,
                          args.epochs, args.batch_size,
                          model_name=args.model_name,
                          warmup=args.warmup, args=args,
                          show=args.debug, tb=args.tb)
    print(f"ELBO: {elbo}")

    return


if __name__ == "__main__":
    main()
