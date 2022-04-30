from fastai.vision.all import *


def get_dataset(balance=True):
    """Gets the dataset from folder and loads as a ImageDataLoaders Object"""
    PATH = Path("data/processed/")
    trfm = [CropPad(170, "zeros")]
    fnames = [
        photo
        for folder in [PATH / "train", PATH / "val"]
        for classes in folder.ls()
        for photo in classes.ls()
    ]
    if balance:
        dls = ImageDataLoaders.from_path_func(
            PATH, fnames, (lambda x: x.parent.name), seed=100, item_tfms=trfm
        )
    else:
        dls = ImageDataLoaders.from_folder(
            PATH, train="train", valid="val", seed=100, item_tfms=trfm
        )
    # dls.show_batch()
    return dls


def train(dls, model_name):
    """
    Trains the model given ImageDataLoaders and a CNN Architecture
    Additionally it saves the model and prints model quality info
    """
    learner = vision_learner(dls, model_name, metrics=error_rate)
    learner.fine_tune(10, freeze_epochs=4)
    learner.save(Path(f"models/{str(model_name)}"))

    diag = ClassificationInterpretation.from_learner(learner)
    diag.print_classification_report()
    print("\n\n\n")


def main():
    dls = get_dataset()
    train(dls, models.resnet34)


if __name__ == "__main__":
    main()
