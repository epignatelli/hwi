import os
import csv
import random
import numpy as np
import cv2
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def triplets_from_dataframe(dataframe, generator, directory, target_size, batch_size):
    Xa = generator.flow_from_dataframe(dataframe=dataframe, directory=directory,
                                       batch_size=batch_size, target_size=target_size,
                                       x_col="anchor_input", y_col="id",)
    Xp = generator.flow_from_dataframe(dataframe=dataframe, directory=directory,
                                       batch_size=batch_size, target_size=target_size,
                                       x_col="positive_input", y_col="id",)
    Xn = generator.flow_from_dataframe(dataframe=dataframe, directory=directory,
                                       batch_size=batch_size, target_size=target_size,
                                       x_col="negative_input", y_col="id",)
    while True:
        triplet = {'anchor_input': Xa.next()[0], 'positive_input': Xp .next()[0], 'negative_input': Xn .next()[0]}
        label = None
        # print(triplet)
        yield (triplet, label)


def get_triplets(pairs_dict, csv_out_path="../triplets.csv"):
    """
    Inputs a list of key value pairs (key, value) and returns a list of triplets.
    """

    # TODO: The current implementation gets jsut one negative example for each anchor
    # Thus, the number of triplets is equal to the number of images with at least a positive example

    print("Creating triplets from images list...")
    with open(csv_out_path, "w+", newline='') as csvout:
        w = csv.writer(csvout)
        w.writerow(["anchor_input", "positive_input", "negative_input", "id"])

        keys = list(pairs_dict.keys())
        triplets = []
        for key, img_list in pairs_dict.items():
            if len(img_list) <= 1:  # safe guard if no positive example
                print("Skipping %s. No positive example found" % key)
                continue

            for img in img_list:
                anchor_name = img_list.pop()
                positive_name = img_list.pop()
                # TODO: Make the selection of negative examples smarter
                # Consider clustering or cosine similarity with the positive example
                # to distinghish between hard, easy and semi-hard negatives
                negative_key = random.choice(keys)
                negative_img_list = pairs_dict[negative_key]

                while (len(negative_img_list) <= 0) or (negative_key == key):
                    # Got empty key, probably because it's the anchor one, let's pick another one
                    negative_key = random.choice(keys)
                    negative_img_list = pairs_dict[negative_key]

                negative_name = random.choice(pairs_dict[negative_key])
                triplet = (anchor_name, positive_name, negative_name, key)

                triplets.append(triplet)
                w.writerow(triplet)
    return triplets


def get_pairs_dict(csv_path="../train.csv"):
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        d = {}
        for row in csv_reader:
            key = row[1]
            value = row[0]
            if key in d:
                d[key].append(value)
            else:
                d[row[1]] = [row[0]]

        print([len(val) for val in d.values()])
        print(np.mean(np.array([len(val) for val in d.values()])))
        print(np.std(np.array([len(val) for val in d.values()])))
        print(len([len(val) for val in d.values()]))
    return d


# This is only for testing purposes
if __name__ == "__main__":
    import pandas as pd

    pairs = get_pairs_dict("../train.csv")
    triplets = get_triplets(pairs)

    data_gen = ImageDataGenerator()
    df = pd.read_csv("../triplets.csv", )
    print(df.head())
    train_gen = triplets_from_dataframe(dataframe=df,
                                       generator=data_gen,
                                       directory="../data/train/",
                                       target_size=(331, 331),
                                       batch_size=64,
                                       )

    next(train_gen)
