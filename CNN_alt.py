import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D#, Dropout
import numpy as np

#tf.config.list_physical_devices('GPU')


def get_generator(path, split):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=split,
        zoom_range=[0.9, 1.6],
        horizontal_flip=False,
        width_shift_range=0.20,
        height_shift_range=0.20,
        rotation_range=10,
        brightness_range=[0.9, 1.1])
    training_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="training",
        target_size=(224, 224))
    validation_generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="validation",
        target_size=(224, 224))
    return training_generator, validation_generator


def get_test_generator(path):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    generator = image_data_generator.flow_from_directory(
        directory=path,
        batch_size=32,
        shuffle=False,
        target_size=(224, 224))
    return generator


def get_model(name):
    try:
        print("Loading model...")
        return load_model("models/" + name)
    except:
        print("New model")
        # model = Sequential()
        # model.add(Conv2D(input_shape=(None, None, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        # model.add(Flatten())
        # model.add(Dense(29, activation="softmax"))
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = Conv2D(3, 7, activation='relu')(inputs)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Flatten()(x)
        outputs = Dense(29, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        return model


if __name__ == '__main__':
    name = "cnn64_alt"
    training_generator, validation_generator = get_generator('train/', 0.1)
    test_generator = get_test_generator('newtest/')
    model = get_model(name + ".h5")
    print(model.summary())
    for e in range(30):
        history = model.fit(training_generator, validation_data=validation_generator, epochs=1)
        test_labels = test_generator.classes
        test_predictions = np.argmax(model.predict(test_generator, verbose=1), axis=1)
        test_acc = round(np.mean(test_labels == test_predictions) * 100)
        val_acc = round(history.history['val_accuracy'][0] * 100)
        model.save("models/" + name + str(e) + "_" + str(val_acc) + "_" + str(test_acc) + ".h5")
