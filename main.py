import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from vampprior.models import VAE

def train_test_vae(vae, x_train, x_test, epochs,
                   model_name, show=True):
    """
    Train model and visualize result
    """
    vae.fit(x_train, x_train, epochs=epochs)

    print("Now testing reconstruction")
    reconstructions = vae(x_test[:5])

    plt.title(f"Reconstruction for {model_name}")
    for i, reconstruction in enumerate(reconstructions):
        plt.subplot(2, 5, 1+i)
        plt.imshow(x_test[i])

        plt.subplot(2, 5, 6+i)
        plt.imshow(reconstruction)

    if show:
        plt.show()
    else:
        # create folder to save images if it does not exists
        if not os._exists("img"):
            os.mkdir("img")

        plt.savefig(os.path.join("img", f"{model_name}-reconstructions.png"))

    print("Now testing generation")
    generations = vae.generate(5)

    plt.title(f"Generations for {model_name}")
    for i, generation in enumerate(generations):
        plt.subplot(1, 5, 1+i)
        plt.imshow(generation)

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join("img", f"{model_name}-generations.png"))


def main():
    mnist = tf.keras.datasets.mnist
    (mnist_train, _), (mnist_test, _) = mnist.load_data()

    # simple workaround for working with binary data
    # where each pixel is either 0 or 1
    # TODO: use correct dataset
    mnist_train = np.array((mnist_train / 255.) > 0.5, dtype=np.float32)
    mnist_test = np.array((mnist_test / 255.) > 0.5, dtype=np.float32)

    # simple VAE, normal standard prior
    standard_vae = VAE(40, 1)
    standard_vae.compile(optimizer='adam',
                         loss=tf.nn.sigmoid_cross_entropy_with_logits)

    train_test_vae(standard_vae, mnist_train, mnist_test,
                   1, model_name="standard-vae", show=False)

    return

if __name__ == "__main__":
    main()
