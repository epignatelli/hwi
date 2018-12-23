import create_model
import predict
import train
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib


# if not an imported module
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device")
    args = parser.parse_args()
    device = args.device

    if device is not None:
        if "GPU" in [x.name for x in device_lib.list_local_devices()]:
            dev = "/gpu:" + device
        else:
            dev = "/cpu:0"

        with tf.device(dev):
            model = create_model.build_model()
            config, history = train.train(model)
            result = predict.recognize(model, config["classes_map"])
    else:
        model = create_model.build_model()
        config, history = train.train(model)
        result = predict.recognize(model, config["classes_map"])

