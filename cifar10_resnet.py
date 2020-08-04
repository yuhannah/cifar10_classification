# -- coding: utf-8 --
# train1: 使用 resnet 的网络

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
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


def resnet_block(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if (activation):
        x = Activation('relu')(x)
    return x


# 建一个20层的ResNet网络
def resnet(input_shape):
    inputs = Input(shape=input_shape)  # Input层，用来当做占位使用

    # 第一层
    x = resnet_block(inputs)
    print('layer1, xshape:', x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs=x)
        b = resnet_block(inputs=a, activation=None)
        x = tf.keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=32)
        else:
            a = resnet_block(inputs=x, num_filters=32)
        b = resnet_block(inputs=a, activation=None, num_filters=32)
        if i == 0:
            x = Conv2D(32, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = tf.keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out:16*16*32
    # 第14~19层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=64)
        else:
            a = resnet_block(inputs=x, num_filters=64)

        b = resnet_block(inputs=a, activation=None, num_filters=64)
        if i == 0:
            x = Conv2D(64, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = tf.keras.layers.add([x, b])  # 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:8*8*64
    # 第20层
    x = AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(10, activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 初始化模型
    # 之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# 动态变化学习率
def lr_sch(epoch):
    # 200 total
    if epoch < 50:
        return 1e-3
    if 50 <= epoch < 100:
        return 1e-4
    if epoch >= 100:
        return 1e-5


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
    x_train /= 255
    x_test /= 255

    # train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, fill_mode='wrap')
    # test_datagen = ImageDataGenerator(rescale=1./255)

    model = resnet((32, 32, 3))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./resnet/cifar10_resnet_ckpt.h5', monitor='val_accuracy', verbose=1,
                                 save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, mode='max', min_lr=1e-3)
    callbacks = [checkpoint, lr_scheduler, lr_reducer]
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num, validation_data=(x_test, y_test),
                     verbose=1, callbacks=callbacks)

    # hist = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = 8000, epochs = epochs_num, validation_data=(x_test,y_test), verbose=1, callbacks=callbacks)

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
    plt.savefig("output/resnet_accuracy_loss.png")


if __name__ == '__main__':
    train()
