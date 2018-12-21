import os
import time
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model
from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.nasnet import NASNetLarge
from keras.utils import multi_gpu_model
from loss import *


def build_model(embedding_dim=60, arch='resnet', weights="imagenet", triplet_mining="online", multigpu=True):
    """Create a keras model instance base on resnet or nasnet architecture to use for transfer learning

    Args:
        input_shape: tuple (height, width, channels)
        arch: model architecture, default 'resnet', alternative 'nasnet'
        weights: optional custom weights, default 'imagenet' 

    Returns:
        mask: A keras model instance
    """

    now = time.time()

    # Creating network architecture
    if arch == "nasnet":
        print("Creating NasNet model, retrieving weights...")
        input_shape = (331, 331, 3)
        freeze_breakpoint = "activation_166"
        conv_model = NASNetLarge(input_shape=input_shape, weights=weights)
    elif arch == "resnet":
        freeze_breakpoint = "activation_43"
        input_shape = (224, 224, 3)
        conv_model = ResNet50(input_shape=input_shape, weights=weights)
    else:
        print("model architecture not specified or an unknown architecture has been specified")
        return

    # Removing top layers
    for i in range(3):
        conv_model.layers.pop()

    # Freezing first layers
    print("Setting trainable layers...")
    conv_model.trainable = True
    after_checkpoint = False
    for layer in conv_model.layers:
        if layer.name == freeze_breakpoint:
            after_checkpoint = True
        layer.trainable = after_checkpoint
        print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))
    print("Embedding model created in %f s. Printing summary..." % (time.time() - now))

    # Adding encoding layer
    x = conv_model.output
    x = Dropout(0.5)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
    embedding_model = Model(conv_model.input, x, name="embedding")

    # Merging embedding and triplet model

    if triplet_mining == "online":
        triplet_model = embedding_model
    
    elif triplet_mining == "offline":
        anchor_input = Input(input_shape, name='anchor_input')
        anchor_embedding = embedding_model(anchor_input)
        positive_input = Input(input_shape, name='positive_input')
        negative_input = Input(input_shape, name='negative_input')

        positive_embedding = embedding_model(positive_input)
        negative_embedding = embedding_model(negative_input)

        inputs = [anchor_input, positive_input, negative_input]
        outputs = [anchor_embedding, positive_embedding, negative_embedding]

        triplet_model = Model(inputs, outputs)

    else:
        print("No valid strategy for training found, no model created")
        return

    if multigpu:
        model = multi_gpu_model(triplet_model, gpus=2)
    model.summary()
    return model


# Only for test purposes
if __name__ == "__main__":
    model = build_model(weights="../weights/resnet50-imagenet.hdf5")
