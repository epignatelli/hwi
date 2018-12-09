import os
import argparse
import time
import datetime
import json
import socket
import getpass
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import create_model
import preprocess


def train(model, train_dir, test_dir=None, out_dir=None,
          input_shape=(331, 331, 3),
          weights="../weights/nasnet-imagenet_weights.h5",
          embedding_dim=60,
          epochs=10,
          batch_size=32,):
    
    print("Setting up variables...")
    timestamp = int(time.time())

    # setting folders
    destination_dir = os.path.join(train_dir, "..", "bin")
    training_out_dir = os.path.join(destination_dir, "sessions", "training", "training_" + str(timestamp))
    checkpoint_name = "_epoch{epoch:02d}_acc{val_acc:.2f}_loss{val_loss:4f}.hdf5"
    checkpoint_path = os.path.join(training_out_dir, str(timestamp) + checkpoint_name)
    os.makedirs(training_out_dir)

    # setting input_shape
    if len(input_shape) <= 2:
        input_shape = (input_shape[0], input_shape[1], 3)
    elif input_shape[0] == 3:
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # setting model
    if isinstance(model, str):
        if model == "":
            encoder, classifier = create_model.build_model(input_shape=input_shape, embedding_dim=embedding_dim, weights=weights)
        else:
            classifier = keras.load_model(model)
    else:
        encoder, classifier = model

    classifier.compile(loss=None, optimizer=Adam(0.01))

    # setting callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor="val_acc",
                                                          save_best_only=False,
                                                          period=1,
                                                          verbose=1
                                                          )

    # Getting data
    print("Getting data...")
    img_gen = ImageDataGenerator(height_shift_range=20,
                                 horizontal_flip=True,
                                 preprocessing_function=None,
                                 )
    pairs = preprocess.get_pairs_dict("../train.csv")
    triplets = preprocess.get_triplets(pairs)
    train_gen = preprocess.triplets_generator(triplets, img_gen)

    # fitting data
    print("Now fitting data...")
    history = classifier.fit_generator(train_gen,
                                       epochs=epochs,
                                       steps_per_epoch=1000,
                                       callbacks=[checkpoint_callback]
                                       )

    # Saving model and history
    print("Saving results...")
    classifier.save(os.path.join(training_out_dir, str(timestamp) + ".h5"))
    classifier.save(os.path.join(destination_dir, "latest_model.h5"))

    classifier.save(os.path.join(training_out_dir, str(timestamp) + ".h5"))
    classifier.save(os.path.join(destination_dir, "latest_model.h5"))

    data = {
        "Model": str(timestamp) + ".h5",
        "train_history": history.history,
        "__Tag__": "training",
        "__Time__": datetime.datetime.utcnow().isoformat(),
        "__Origin__": getpass.getuser() + "@" + socket.gethostname()
    }

    with open(os.path.join(training_out_dir, str(timestamp) + "_session.json"), 'w+') as outfile:
        json.dump(data, outfile)
    return

