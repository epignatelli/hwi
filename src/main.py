import create_model, preprocess, predict, train, postprocess, display_data

# if not an imported module
if __name__ == "__main__":
    # model = create_model.build_model(weights="../weights/nasnet-imagenet_weights.h5")
    model = create_model.build_model_resnet(weights="../weights/resnet50-imagenet.hdf5")
    trained = train.train(model, "../triplets.csv", "../data/train/", input_shape=(224, 224, 1))
