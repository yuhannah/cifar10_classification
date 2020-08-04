## cifar-10 开启深度学习图像分类

PS：项目中用到了 datasets 和 models 等辅助资料不在仓库中备份。

cifar10是一个图像数据集（[官网](https://www.cs.toronto.edu/~kriz/cifar.html)），包含10种类别的32*32大小的图像共60000张。另外还有cifar100，包含100种类别的更多图像。因此，cifar10分类就是一个图像多分类任务。

### 环境

代码运行环境为 python 3.7.4，tensorflow 1.15.0，keras 2.2.4，Win10 系统，pyCharm IDE。

### 目录

本文记录使用 keras 训练 cifar-10 数据集的过程。内容包括：

- [五层的 CNN](#五层的 CNN)
- [五层的 CNN 加数据增强](#五层的 CNN 加数据增强)
- [使用 VGG16 提取特征](#使用 VGG16 提取特征)
- [深层 CNN](#深层 CNN)
- [ResNet](#ResNet)

### 五层的 CNN 

网络模型为：

input:(32,32,3)-->conv1:(3,3,32)-->relu-->maxpooling-->conv2:(3,3,32)-->relu-->maxpooling-->FC:(8)-->relu-->dropout-->softmax:(10)

训练结果：

![](output/simple_CNN_accuracy.png)

参考代码：cifar10_five_layers_cnn.py

### 五层的 CNN 加数据增强

在加载 datasets 后，对图像数据进行裁剪、旋转等预处理，扩充原始的数据集，能够提高训练的准确率。

训练结果：



参考代码：cifar10_data_augmentation.py

### 使用 VGG16 提取特征

使用 vgg16 模型提取特征，用自定义的全连接层进行训练。去掉 vgg16 模型的输出层，换成自定义的两层全连接层。在训练前，先得到 vgg16 模型的除去最后一层的训练结果，再将结果输入到自定义的两层网络中进行训练。利用训练好的网络模型加快训练速度。（迁移学习）

训练结果：



参考代码：cifar10_vgg16_extract_features.py

### 基于 VGG16 进行模型微调



### 深层 CNN

构建一个深层网络：

训练结果：



参考代码：cifar10_deep_cnn.py

### ResNet

构建一个 resnet。

训练结果：



参考代码：cifar10_resnet.py