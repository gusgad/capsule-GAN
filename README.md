# Capsule GAN

[Link to the paper](https://drive.google.com/file/d/1LoaWLTq1mbEVRsYLqn-QjulPb229n9J9/view?usp=sharing)

Code for my Master thesis on "Capsule Layer as a Discriminator in Generative Adversarial Networks". In order to reproduce results, follow the "capsule_gan" Jupyter notebook that contains:
* Dataset loading and preprocessing
* Both Discriminator and Generator structures
* Training, loss functions
* Image outputs
* Metrics visualization

***

But first you may want to install [Miniconda](https://conda.io/miniconda.html) and corresponding dependencies from environment.yml:
`conda env create -f environment.yml` within the project directory as well as install [required tools for GPU computing](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support). If no GPU is going to be used - delete the `tensorflow-gpu` line from environment.yml.


### Generated images
![MNIST_output](/out_metrics/mnist_output_sample.png?raw=true)
![CIFAR10_output](/out_metrics/cifar10_output_sample.png?raw=true)

[All generated MNIST images over 30k epochs](https://www.amazon.com/clouddrive/share/BKlDzoKMhrFnIQu9LqH7KcrYvP9ZKoZF1oc5wjMPFRc)
[All generated CIFAR10 images over 30k epochs](https://www.amazon.com/clouddrive/share/V99W1XhuDg0U7ZABNXwtHBVaacMzqdUCKkI6m9Vp4HG)


[Generator weights for MNIST](https://www.amazon.com/clouddrive/share/wSRq5KX7IrWKaxvdsyGZubh9WcffzrfEFW89mEgQdLC)
[Generator weights for CIFAR10](https://www.amazon.com/clouddrive/share/2CKaZZdGlJqyT5WXnYu1Zbka1Vldtr9yCNffYWwz8Wn)

Thanks to @eriklindernoren (<https://github.com/eriklindernoren/Keras-GAN>) who I borrowed the Keras implementation of DCGAN from and @XifengGuo (<https://github.com/XifengGuo/CapsNet-Keras>) who I took the squashing function from.
