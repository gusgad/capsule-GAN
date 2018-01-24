# Capsule GAN

Code for my Master thesis on "Capsule Layer as a Discriminator in Generative Adversarial Networks". In order to reproduce results, follow the "capsule_gan" Jupyter notebook that contains:
* Dataset loading and preprocessing
* Both Discriminator and Generator structures
* Training, loss functions
* Image outputs
* Metrics visualization

### Generated images
![MNIST_output](/out_metrics/mnist_output_sample.png?raw=true)
![CIFAR10_output](/out_metrics/cifar10_output_sample.png?raw=true)

Thanks to @eriklindernoren (<https://github.com/eriklindernoren/Keras-GAN>) who I borrowed the Keras implementation of DCGAN from and @XifengGuo (<https://github.com/XifengGuo/CapsNet-Keras>) who I took the squashing function from.