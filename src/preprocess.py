import os
import csv
import random
import numpy as np
import cv2
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def triplets_generator(triplets, img_size=None, generator=None, data_folder="../data/train", batch_size=64):
    """
    triplets is a tuple (anchor, positive, negative) of image names
    """

    print("Creating triplets generator for batch yielding...")
    if img_size is None:
        img_size = (331, 331)
    n_batches = len(triplets) // batch_size
    for j in range(n_batches):
        anchor_list = []
        neg_list = []
        pos_list = []

        for i in range(batch_size):
            anchor_name, pos_name, neg_name = triplets[(j * batch_size) + i]
            anchor_path = os.path.join(data_folder, anchor_name)
            pos_path = os.path.join(data_folder, pos_name)
            neg_path = os.path.join(data_folder, neg_name)

            # read
            anchor_img = cv2.imread(anchor_path)
            positive_img = cv2.imread(pos_path)
            negative_img = cv2.imread(neg_path)

            # resize
            anchor_img = cv2.resize(anchor_img, (331, 331), interpolation=cv2.INTER_AREA)
            positive_img = cv2.resize(positive_img, (331, 331), interpolation=cv2.INTER_AREA)
            negative_img = cv2.resize(negative_img, (331, 331), interpolation=cv2.INTER_AREA)

            anchor_list.append(anchor_img)
            pos_list.append(positive_img)
            neg_list.append(negative_img)

        A = np.array(anchor_list, dtype="float32")
        B = np.array(pos_list, dtype="float32")
        C = np.array(neg_list, dtype="float32")

        A = preprocess_input(A)
        B = preprocess_input(B)
        C = preprocess_input(C)

        label = None
        trip = {'anchor_input': A, 'positive_input': B, 'negative_input': C}
        yield (trip, label)


def get_triplets(pairs_dict, csv_out_path="../triplets.csv"):
    """
    Inputs a list of key value pairs (key, value) and returns a list of triplets.
    """

    # TODO: The current implementation gets jsut one negative example for each anchor
    # Thus, the number of triplets is equal to the number of images with at least a positive example

    print("Creating triplets from images list...")
    with open(csv_out_path, "w+") as csvout:
        w = csv.writer(csvout)

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
                triplet = (anchor_name, positive_name, negative_name)

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
    pairs = get_pairs_dict("../train.csv")
    triplets = get_triplets(pairs)

    data_gen = ImageDataGenerator()
    train_gen = triplets_generator(triplets, generator=data_gen)
    batch = next(train_gen)
