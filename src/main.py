import create_model, preprocess, predict, train, postprocess, display_data

# if not an imported module
if __name__ == "__main__":
    create_model.build_model(weights="../weights/nasnet-imagenet_weights.h5")