import os
import time
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model
# from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input


def build_model(input_shape=None, embedding_dim=60, dropout=0.5, weights="imagenet"):

    if input_shape is None:
        input_shape = (331, 331, 3)

    now = time.time()
    print("Retrieving NASNet weights...")
    conv_model = NASNetLarge(input_shape=input_shape, weights=weights)

    print("Setting trainable layers...")
    conv_model.trainable = True
    for layer in conv_model.layers:
        if layer.name == 'activation_166':
            layer.trainable = True
        else:
            layer.trainable = False
        print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))
    conv_model.summary()
    print("Embedding model created in %f s. Printing summary..." % (time.time() - now))

    x = conv_model.output
    x = Dropout(dropout)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
    embedding_model = Model(conv_model.input, x, name="embedding")

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))

    return embedding_model, triplet_model


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)
