import create_model,
import predict
import train

# if not an imported module
if __name__ == "__main__":
    model = create_model.build_model(weights="../weights/nasnet-imagenet_weights.h5")
    trained = train.train(model)
    result = predict.predict()
