import os
import csv
import random
import numpy as np


def get_triples(pairs_dict, csv_out_path="../triplets.csv"):
    """
    Inputs a list of key value pairs (key, value) and returns a list of triplets
    """
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

                while len(negative_img_list) <= 0:
                    negative_key = random.choice(keys)
                    negative_img_list = pairs_dict[negative_key]                    

                negative_name = random.choice(pairs_dict[negative_key])
                triplet = (anchor_name, positive_name, negative_name)

                triplets.append(triplet)
                w.writerow(triplet)
    return


def get_pairs_dict(csv_path="../train.csv"):
    with open(csv_path, 'r+') as csvfile:
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


def get_image(id, img_id_dict, data_folder="../data/train"):
    cv2.imread(os.path.join(data_folder, img_id_dict))


def gen(triplet_gen, batch_size=64):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            path_pos1 = join(path_train, positive_example_1)
            path_neg = join(path_train, negative_example)
            path_pos2 = join(path_train, positive_example_2)

            positive_example_1_img = read_and_resize(path_pos1)
            negative_example_img = read_and_resize(path_neg)
            positive_example_2_img = read_and_resize(path_pos2)

            positive_example_1_img = augment(positive_example_1_img)
            negative_example_img = augment(negative_example_img)
            positive_example_2_img = augment(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        A = preprocess_input(np.array(list_positive_examples_1))
        B = preprocess_input(np.array(list_positive_examples_2))
        C = preprocess_input(np.array(list_negative_examples))

        label = None

        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, label)


# Test
if __name__ == "__main__":
    pairs = get_pairs_dict()
    get_triples(pairs)