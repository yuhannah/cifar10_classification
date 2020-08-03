# -- coding: utf-8 --
# train1: 使用 vgg16 模型提取特征，用自定义的全连接层进行训练

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
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
epochs_num = 200


def quality_classify_model():
    model = Sequential([
        Flatten(input_shape=(4, 4, 512)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(classes_num, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def train():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5" # 指定序号的GPU训练

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
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    datagen = ImageDataGenerator(rescale=1. / 255)

    # 加载预训练好的卷积基
    conv_base = VGG16(include_top=False, weights='imagenet')

    # 用预训练好的卷积基处理训练集提取特征
    sample_count = len(y_train)
    train_features = np.zeros(shape=(sample_count, 4, 4, 512))
    train_labels = np.zeros(shape=(sample_count, classes_num))
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = conv_base.predict(inputs_batch)
        train_features[i * batch_size: (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    # train_features = np.reshape(train_features, (sample_count, 4*4*512))

    # 用预训练好的卷积基处理验证集提取特征
    sample_count = len(y_test)
    test_generator = datagen.flow(x_test, y_test, batch_size=batch_size)
    test_features = np.zeros(shape=(sample_count, 4, 4, 512))
    test_labels = np.zeros(shape=(sample_count, classes_num))
    i = 0
    for inputs_batch, labels_batch in test_generator:
        features_batch = conv_base.predict(inputs_batch)
        test_features[i * batch_size: (i + 1) * batch_size] = features_batch
        test_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    # test_features = np.reshape(test_features, (sample_count, 4*4*512))

    model = quality_classify_model()

    # hist = model.fit_generator(train_datagan.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = 8000, epochs = epochs_num, validation_data=(x_test,y_test), shuffle=True)
    hist = model.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs_num,
                     validation_data=(test_features, test_labels))

    model.save('./model/vgg16_extract_features/cifar10_model.hdf5')
    model.save_weights('./model/vgg16_extract_features/cifar10_model_weight.hdf5')

    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    print("train acc: ", train_acc)
    print("validation acc: ", val_acc)

    # 绘图
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title('Training and validation accuracy')
    plt.subplot(122)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.legend()
    plt.ylabel('costs')
    plt.xlabel('iterations')
    plt.title('Training and validation loss')
    plt.savefig("output/vgg16_extract_features_accuracy_loss.png")


if __name__ == '__main__':
    train()
