import tensorflow as tf
from keras import layers, models

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images, test_images = train_images / 255.0, test_images / 255.0

# 构建LeNet模型
model = models.Sequential([
    # 卷积层
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    # 池化层
    layers.MaxPooling2D((2, 2)),
    # 卷积层
    layers.Conv2D(16, (5, 5), activation='relu'),
    # 池化层
    layers.MaxPooling2D((2, 2)),
    # 展平
    layers.Flatten(),
    # 全连接层
    layers.Dense(120, activation='relu'),
    # 全连接层
    layers.Dense(84, activation='relu'),
    # 全连接层
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', # Adaptive Moment Estimation 自适应学习率优化算法
              loss='sparse_categorical_crossentropy', # 多类别交叉熵函数：处理标签是整数的情况
              metrics=['accuracy'])

# 将数据reshape成(样本数, 高度, 宽度, 通道数)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
