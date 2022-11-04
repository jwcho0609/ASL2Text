import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
tf.config.list_physical_devices('GPU')


def get_generator(path, split):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=split)
    training_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=64,
        seed=42,
        shuffle=True,
        subset="training",
        target_size=(200, 200))
    validation_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=64,
        seed=42,
        shuffle=True,
        subset="validation",
        target_size=(200, 200))
    return training_generator, validation_generator


def get_model(name):
    try:
        print("Loading model...")
        return load_model("models/" + name)
    except:
        print("New model")
        model = Sequential()
        model.add(Flatten(input_shape=(200, 200, 3)))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(29, activation="softmax"))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(lr=0.001))
        return model


if __name__ == '__main__':
    name = "nn"
    training_generator, validation_generator = get_generator('train/', 0.2)
    test_generator, _ = get_generator('test/', 0)
    model = get_model(name + ".h5")
    history = model.fit(training_generator, validation_data=validation_generator, epochs=20)
    model.save("models/" + name + ".h5")
