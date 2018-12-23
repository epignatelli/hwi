import os
import time
import csv
import json
import pandas as pd
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import NearestNeighbors
from hyperparameters import Hparams
import loss


def predict(target="test",
            model="../data/bin/latest_model.h5",
            data_dir="../data",
            hyperparam_path="../hyperparam.json"):

    # Retrieving hyper parameters
    hparams = Hparams(hyperparam_path)

    # If model is a path, loading it
    if isinstance(model, str):
        model = load_model(model, custom_objects={"batch_hard_triplet_loss": loss.batch_hard_triplet_loss})

    # Generating embedding from trained model
    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    iterator = img_gen.flow_from_directory(data_dir,
                                           classes=[target],
                                           batch_size=hparams.batch_size,
                                           target_size=(hparams.input_shape[0], hparams.input_shape[1]))
    preds = model.predict_generator(iterator,
                                    steps=(iterator.samples // hparams.batch_size) + 1,
                                    verbose=1)
    filenames = [name.split("/")[-1] for name in iterator.filenames]

    return preds, filenames


def recognize(model, classes_map):
    # Creating databse to search from. Saving it ti hwi/train_preds.csv
    train_preds, train_names = predict(model=model, target="train")

    # Creating embedding of test files
    test_preds, test_names = predict(model=model, target="test")

    # Searching for the test[i] nearest point in the entire train set
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(train_preds)
    distances_test, neighbors_test = neigh.kneighbors(test_preds)
    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

    # Co
    preds_str = []
    for filepath, distance, neighbour_ in zip(test_names, distances_test, neighbors_test):
        sample_result = []
        sample_classes = []
        for d, n in zip(distance, neighbour_):
            train_file = train_names[n]
            class_train = classes_map[train_file]
            sample_classes.append(class_train)
            sample_result.append((class_train, d))

        if "new_whale" not in sample_classes:
            sample_result.append(("new_whale", 0.1))
        sample_result.sort(key=lambda x: x[1])
        sample_result = sample_result[:5]
        preds_str.append(" ".join([x[0] for x in sample_result]))

        df = pd.DataFrame(preds_str, columns=["Id"])
        df['Image'] = test_names
        df.to_csv("sub_humpback.csv", index=False)

    return preds_str


if __name__ == "__main__":
    predict()
