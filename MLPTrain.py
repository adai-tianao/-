import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# 加载 MNIST 手写字体数据集
# 28x28像素灰度图像 像素值0~255 附带有标签值
# 60000张训练用图
# 10000张测试用图
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
# images是三维的numpy数组 形状 (60000, 28, 28) 
# 首先将images每个图转化为长度为784(28*28)的一维数组
# 再将图像数据类型转换为float32
# 再归一化到0~1区间内(像素值都是非负值)
train_images = train_images.reshape((60000,28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255

# 数据预处理
# 把标签数据转换为独热码比如: 
# 0 -> [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 1 -> [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# 9 -> [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 神经网络模型定义
# MLP共三层: 输入层 1层隐藏层 输出层
# 定义初始化权重矩阵及偏置值
# 权重矩阵使用正态分布随机数进行初始化, 原因:
# 1. 随机性避免神经元学习陷入对称性问题
# 2. 避免初始化为0值：如果权重矩阵为0矩阵，会使反向传播时权重梯度相等
# 3. 数值范围控制在一个较小的范围，避免梯度爆炸或消失
# 4. 中心极限定理：独立随机变量的和趋于正态分布
def initialize_model(input_size, hidden_size, output_size):
    np.random.seed(42)
    w1 = np.random.randn(hidden_size, input_size) * 0.1 # 权重矩阵1初始化
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * 0.1 # 权重矩阵2初始化
    b2 = np.zeros((output_size, 1))
    return {"w1":w1,"b1":b1, "w2":w2, "b2":b2}

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z, axis = 0, keepdims=True)

# 前向传播
# dot: 计算矩阵乘法(或者矩阵与向量的乘法)
# 这里是权重矩阵与输入向量的乘法
def forward_propagation(X, model):
    neth1 = np.dot(model["w1"], X.T) + model["b1"]
    outh1 = np.maximum(0, neth1) # RELU 激活函数
    neto1 = np.dot(model["w2"], outh1) + model["b2"] # 列向量
    # outo1 = softmax(neto1) # RELU 激活函数 -> softmax
    outo1 = np.maximum(0, neto1)
    return {"neth1":neth1, "outh1":outh1, "neto1":neto1, "outo1":outo1}


# 计算损失函数
# loss = Sigma(Y - Y_pre)^2 / 2
def compute_loss(outo1, Y):
    m = Y.shape[0] # 分类数量 -> 10
    loss = np.sum(np.square(Y - outo1)) / (2)
    return loss

# 反向传播 求出梯度
def backward_propagation(model, buffer, X, Y):
    m = X.shape[0]
    dz2 = buffer["outo1"] - Y # 列向量 ** 不知道为什么softmax的求导没有体现出来 **
    dw2 = np.dot(dz2, buffer["outh1"].T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m    
    dz1 = np.dot(model["w2"].T, dz2) * (buffer["neth1"] > 0)
    dw1 = np.dot(dz1, X) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return {"dw1":dw1, "db1":db1, "dw2":dw2, "db2":db2}

# 权重更新 实际也更新偏置
def update_weight(model, grads, learning_rate = 0.01):
    model["w1"] -= learning_rate * grads["dw1"]
    model["b1"] -= learning_rate * grads["db1"]
    model["w2"] -= learning_rate * grads["dw2"]
    model["b2"] -= learning_rate * grads["db2"]
    return model

# 定义神经网络架构
input_size = 28*28
hidden_size = 32
output_size = 10

# 初始化参数
model = initialize_model(input_size,hidden_size,output_size)

# 训练模型
num_epochs = 3
learning_rate = 0.02


# 训练过程
for epoch in range(num_epochs):
    all_loss = 0
    for i in range(len(train_images)):
        X = train_images[i].reshape((1,-1))
        Y = train_labels[i].reshape((-1,1))

        # 前传
        buf = forward_propagation(X, model)

        # 计算损失
        loss = compute_loss(buf["outo1"],Y)
        all_loss += loss
        # 反向传播 计算梯度
        grads = backward_propagation(model,buf,X,Y)

        # 权重更新
        model = update_weight(model, grads, learning_rate)
    ave_loss = all_loss / len(train_images)
    print(f"Epoch {epoch + 1}/{num_epochs}, ave loss:{ave_loss}")


# 推理过程
correct = 0
for i in range(len(test_images)):
    X_test = test_images[i].reshape((1,-1))
    Y_test = test_labels[i].reshape((-1,1))

    # 前向传播
    buf = forward_propagation(X_test,model)

    # 预测类别：以概率值最大的作为预测结果
    pre_res = np.argmax(buf["outo1"])
    true_res = np.argmax(Y_test)
    if pre_res == true_res:
        correct += 1

print(f"Accuracy = {(correct / len(test_images)) * 100} %")




