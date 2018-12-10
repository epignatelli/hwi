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


def build_model(input_shape=(331, 331, 3), embedding_dim=60, dropout=0.5, weights="imagenet"):
    now = time.time()
    print("Retrieving NASNet weights...")
    conv_model = NASNetLarge(input_shape=input_shape, weights=weights)

    print("Setting trainable layers...")
    conv_model.trainable = True
    after_checkpoint = False
    for layer in conv_model.layers:
        if layer.name == 'activation_166':
            after_checkpoint = True
        layer.trainable = after_checkpoint
        print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))
    print("Embedding model created in %f s. Printing summary..." % (time.time() - now))
    conv_model.summary()

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
    triplet_model.summary()

    return embedding_model, triplet_model


def build_model_resnet(input_shape=(224, 224, 3), embedding_dim=60, dropout=0.5, weights="imagenet"):
    now = time.time()
    print("Retrieving ResNet weights...")
    conv_model = ResNet50(input_shape=input_shape, weights=weights)
    for i in range(2):
        conv_model.layers.pop()

    print("Setting trainable layers...")
    conv_model.trainable = True
    after_checkpoint = False
    for layer in conv_model.layers:
        layer.trainable = after_checkpoint
        if layer.name == 'activation_43':
            after_checkpoint = True
        print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))

    print("Embedding model created in %f s. Printing summary..." % (time.time() - now))
    conv_model.summary()

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


# Only for test purposes
if __name__ == "__main__":
    conv_model, triplet_model = build_model_resnet(weights="../weights/resnet50-imagenet.hdf5")
