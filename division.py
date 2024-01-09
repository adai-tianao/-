# 针对生成的网络模型进行可视化
# 针对测试数据中的某一张图进行随机测试
# 给出测试结果图

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing import image

# 加载预训练的 LeNet 模型
model = load_model('cifra10_model.h5')

# 加载 CIFAR-10 数据集
(_, _), (test_images, test_labels) = cifar10.load_data()

# 选择一张测试图像
image_index = np.random.randint(0, test_images.shape[0])
test_image = test_images[image_index]
true_label = test_labels[image_index]

# 预处理图像，使其符合 LeNet 模型的输入要求
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # 归一化

# 进行模型推理
predictions = model.predict(test_image)
predicted_label = np.argmax(predictions)

# 显示原始图像和模型的预测结果
plt.figure(figsize=(8, 4))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(test_images[image_index])
plt.title(f'True Label: {true_label}')

# 显示模型的预测结果
plt.subplot(1, 2, 2)
plt.bar(range(10), predictions[0])
plt.xticks(range(10), [str(i) for i in range(10)])
plt.title(f'Predicted Label: {predicted_label}')

plt.show()
