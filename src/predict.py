import os
import time
import csv
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import loss

def predict(model="../data/bin/latest_model.h5", 
            image_dir="../data/test", out_dir="../data/sessions/eval"):
    
    start = int(time.time())

    # Sorting folders
    target_dir = os.path.join(out_dir, str(start))
    target_path = os.path.join(target_dir, str(start) + "eval.csv")

    try:
        os.mkdir(target_dir)
    except OSError as e:
        print("Warning: " + str(e))

    # Retrieving model
    if isinstance(model, str):
        model = load_model(model_path)

    model.compile(custom_objects={"batch_all_triplet_loss": loss.batch_all_triplet_loss})

    with open(target_path, "w+") as outcsv:
        csv_writer = csv.writer(outcsv)

        img_gen = ImageDataGenerator(preprocess_function=preprocess_input)

        test_gen = img_gen.flow_from_directory(image_dir)

        result = model.predict_generator(test_gen, steps=1, verbose=1)
        csv_writer.writerow("Image", "Id")
        for pred in result:
            print(pred)


if __name__ == "__main__":
    predict()
