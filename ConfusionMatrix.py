import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import seaborn
import matplotlib.pyplot as plt


def get_generator(path, split):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=split)
    training_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=16,
        seed=42,
        shuffle=False,
        subset="training",
        target_size=(64, 64))
    validation_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=16,
        seed=42,
        shuffle=False,
        subset="validation",
        target_size=(64, 64))
    return training_generator, validation_generator


def plot_confusion_matrix(data, labels, output_filename):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=0.5)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='g')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


def cm_for_model(names, models, train=False, validation=False, test=True):
    training_generator, validation_generator = get_generator('train/', 0.20)
    test_generator, _ = get_generator('newtest/', 0)
    labels = test_generator.class_indices
    if train:
        train_labels = training_generator.labels
        train_predictions = predict(models, training_generator)
        train_cm = confusion_matrix(train_labels, train_predictions)
        plot_confusion_matrix(train_cm, labels, "CMs/" + names[0] + "_train.png")
    if validation:
        validation_labels = validation_generator.classes
        validation_predictions = predict(models, validation_generator)
        validation_cm = confusion_matrix(validation_labels, validation_predictions)
        plot_confusion_matrix(validation_cm, labels, "CMs/" + names[0] + "_validation.png")
    if test:
        test_labels = test_generator.classes
        test_predictions = predict(models, test_generator)
        test_cm = confusion_matrix(test_labels, test_predictions)
        acc = np.mean(test_labels == test_predictions)
        print(acc)
        plot_confusion_matrix(test_cm, labels, "CMs/" + names[0] + ".png")


def predict(models, generator):
    prediction = []
    for model in models:
        if len(prediction):
            prediction += model.predict(generator, verbose=1)
        else:
            prediction = model.predict(generator, verbose=1)
    predictions = np.argmax(prediction, axis=1)
    return predictions


def load_models(names):
    models = []
    for name in names:
        models.append(load_model("models/" + name + ".h5"))
    return models


def main():
    names = ["cnn64ddd12_99_94"]
    models = load_models(names)
    cm_for_model(names, models)


if __name__ == '__main__':
    main()
