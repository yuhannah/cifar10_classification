# -- coding: utf-8 --
# train1: 一个 5 层的 CNN
# input:[batch, 32, 32, 3], conv1: [3, 3, 32], conv2: [3, 3, 32], conv3: [3, 3, 64], FC: [64], softmax: 10

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras import backend as K
from load_data import load_data
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

print("Tensorflow version " + tf.__version__)
print("Tensorflow keras version " + tf.keras.__version__)

classes_num = 10
batch_size = 64
epochs_num = 300


def quality_classify_model():
    model = Sequential([
        Conv2D(kernel_size=3, filters=32, padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(kernel_size=3, filters=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(classes_num, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def train():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5" # 指定序号的GPU训练

    # 数据载入：网络 or 本地
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = load_data()

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    # 多分类标签生成
    y_train = to_categorical(y_train, classes_num)
    y_test = to_categorical(y_test, classes_num)
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)

    # 生成训练数据
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    model = quality_classify_model()
    model.summary()
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num, validation_data=(x_test, y_test),
                     shuffle=True)

    model.save('./model/simple_CNN/cifar10_model.hdf5')
    model.save_weights('./model/simple_CNN/cifar10_model_weight.hdf5')

    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    print("train acc: ", train_acc)
    print("validation acc: ", val_acc)

    # 绘图
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(epochs, train_acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title('Training and validation accuracy')
    plt.subplot(122)
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.legend()
    plt.ylabel('costs')
    plt.xlabel('iterations')
    plt.title('Training and validation loss')
    plt.savefig("output/simple_CNN_accuracy_loss.png")


if __name__ == '__main__':
    train()

# accuracy = 72%
