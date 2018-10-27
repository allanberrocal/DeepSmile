import tensorflow as tf
from tensorflow import keras


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(input_shape=(64, 64, 1), activation=tf.nn.relu, filters=4, kernel_size=(3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    return model


def create_model2():
    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(input_shape=(64, 64, 1), activation=tf.nn.relu, filters=4, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(activation=tf.nn.relu, filters=4, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(activation=tf.nn.relu, filters=4, kernel_size=(3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    return model