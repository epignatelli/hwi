import os
import time
import csv
import json
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import loss

def predict(model="../data/bin/latest_model.h5", 
            image_dir="../data",
            out_dir="../data/bin/sessions/eval",
            hyperparam_path="../hyperparam.json"):
    
    start = int(time.time())

    # Sorting folders
    target_dir = os.path.join(out_dir, str(start))
    target_path = os.path.join(target_dir, str(start) + ".csv")

    try:
        os.mkdir(target_dir)
    except OSError as e:
        print("Warning: " + str(e))

    # Retrieving hyper parameters
    with open(hyperparam_path, "r") as h_json:
        hparams = json.load(h_json)

    if len(hparams["input_shape"]) <= 2:
        input_shape = (hparams["input_shape"][0], hparams["input_shape"][1], 3)
    elif hparams["input_shape"][0] == 3:
        input_shape = (hparams["input_shape"][1], hparams["input_shape"][2], hparams["input_shape"][0])
    else:
        input_shape = hparams["input_shape"]

    # Retrieving model
    if isinstance(model, str):
        model = load_model(model, custom_objects={"batch_hard_triplet_loss": loss.batch_hard_triplet_loss})

    with open(target_path, "w+") as outcsv:
        csv_writer = csv.writer(outcsv)

        img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_gen = img_gen.flow_from_directory(image_dir,
                                               classes=["test"],
                                               target_size=(input_shape[0], input_shape[1]))

        result = model.predict_generator(test_gen,
                                         steps=1,
                                         verbose=1)
        csv_writer.writerow(["Image", "Id"])
        for pred in result:
            print(pred)
            print("%d Images" % test_gen.samples)
            print("Result shape", pred.shape)


if __name__ == "__main__":
    predict()
