import create_model
import predict
import train
import argparse
import tensorflow as tf

# if not an imported module
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device")
    args = parser.parse_args()
    device = args.device

    if device is None:
    	device = "0"

    dev = "/gpu:" + device

    with tf.device(dev):
    	model = create_model.build_model()
    	trained = train.train(model)
    	result = predict.predict(model)
