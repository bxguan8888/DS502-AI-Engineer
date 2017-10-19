import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, kernel=(5, 5), num_filter=20, is_pool=False, pool_kernel=(2,2)):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?

    conv = mx.sym.Convolution(data=input_layer, kernel=kernel, num_filter=num_filter)
    relu = mx.sym.Activation(data=conv, act_type="relu")
    if is_pool:
        return mx.sym.Pooling(data=relu, pool_type="max", kernel=pool_kernel, stride=pool_kernel)
    else:
        return relu


# # Optional
def inception_layer(input_layer):
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    covn1 = conv_layer(input_layer, kernel=(1, 1))
    covn1_flatten = mx.symbol.Flatten(data=covn1)

    covn2 = conv_layer(input_layer, kernel=(1, 1))
    covn2 = conv_layer(covn2, kernel=(3, 3))
    covn2_flatten = mx.symbol.Flatten(data=covn2)

    covn3 = conv_layer(input_layer, kernel=(1, 1))
    covn3 = conv_layer(covn3, kernel=(4, 4))
    covn3_flatten = mx.symbol.Flatten(data=covn3)

    covn4 = mx.sym.Pooling(data=input_layer, pool_type="max", kernel=(3, 3), stride=(3, 3))
    covn4 = conv_layer(covn4, kernel=(1, 1))
    covn4_flatten = mx.symbol.Flatten(data=covn4)


    # inception_output = mx.sym.Concat(*[covn1, covn2, covn3, covn4])
    inception_output = mx.sym.concat(covn1_flatten, covn2_flatten, covn3_flatten, covn4_flatten)
    return inception_output



def get_conv_sym(use_inception=False):

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
    pass

    conv1 = conv_layer(data, is_pool=True)
    conv2 = conv_layer(conv1, is_pool=True)

    if use_inception:
        intermediate_output = inception_layer(conv2)
    else:
        intermediate_output = conv2

    flatten = mx.symbol.Flatten(data=intermediate_output)
    fc1 = mlp_layer(input_layer=flatten, n_hidden=128, activation="relu", BN=True)
    fc2 = mlp_layer(input_layer=fc1, n_hidden=64, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=fc2, num_hidden=10)
    # Softmax with cross entropy loss
    cnn = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return cnn