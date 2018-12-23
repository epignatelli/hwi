import os
import time
import datetime
import json
import socket
import getpass
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from hyperparameters import Hparams
import create_model
import loss


def train(model=None, train_dir="../data/train", train_csv="../train.csv", test_dir="../data/test", hyperparam_path="../hyperparam.json"):
    
    print("Setting up variables...")
    timestamp = int(time.time())

    # Retrieving hyperparameters
    hparams = Hparams(hyperparam_path)

    # setting folders
    destination_dir = os.path.join(train_dir, "..", "bin")
    training_out_dir = os.path.join(destination_dir, "sessions", "training", "training_" + str(timestamp))
    checkpoint_name = "_epoch{epoch:02d}_loss{loss:6f}.hdf5"
    checkpoint_path = os.path.join(training_out_dir, str(timestamp) + checkpoint_name)
    os.makedirs(training_out_dir)

    # setting model
    if model is None:
        classifier = create_model.build_model(embedding_dim=hparams.embedding_size)
    else:
        classifier = model

    if hparams.triplet_mining == "online":
        loss_function = loss.batch_hard_triplet_loss
    elif hparams.triplet_mining == "offline":
        loss_function = loss.triplet_loss_offline

    classifier.compile(loss=loss_function, optimizer=Adam(hparams.learning_rate), metrics=['accuracy'])

    # setting callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor="val_acc",
                                                          save_best_only=False,
                                                          period=1,
                                                          verbose=1
                                                          )
    # Data augumentation and preprocessing
    print("Getting data...")
    img_gen = ImageDataGenerator(height_shift_range=20,
                                 horizontal_flip=True,
                                 preprocessing_function=preprocess_input,   
                                 )


    # Retrieving images from train csv
    df_train = pd.read_csv(train_csv)
    classes_map = {k: v for k, v in zip(df_train["Image"], df_train["Id"])}
    train_gen = img_gen.flow_from_dataframe(dataframe=df_train,
                                            directory=train_dir,
                                            x_col="Image",
                                            y_col="Id",
                                            class_mode="sparse",
                                            target_size=(hparams.input_shape[0], hparams.input_shape[1]),
                                            batch_size=hparams.batch_size,
                                            )

    # Fitting model
    print("Now fitting data...")
    history = classifier.fit_generator(train_gen,
                                       use_multiprocessing=True,
                                       epochs=hparams.epochs,
                                       steps_per_epoch=2,
                                       callbacks=[checkpoint_callback],
                                       verbose=1
                                       )

    
    # Saving model and history
    print("Saving results...")
    classifier.save(os.path.join(training_out_dir, str(timestamp) + ".h5"))
    classifier.save(os.path.join(destination_dir, "latest_model.h5"))

    config = {
        "Model": str(timestamp) + ".h5",
        "session": str(timestamp),
        "hparams": hparams.__dict__,
        "train_history": history.history,
        "classes_map": classes_map,
        "__Tag__": "training",
        "__Time__": datetime.datetime.utcnow().isoformat(),
        "__Origin__": getpass.getuser() + "@" + socket.gethostname()
    }

    with open(os.path.join(training_out_dir, str(timestamp) + "_session.json"), 'w+') as outfile:
        json.dump(config, outfile)

    with open(os.path.join("..", "classes_map.json"), 'w+') as outfile:
        json.dump(classes_map, outfile)

    return config, history 
    