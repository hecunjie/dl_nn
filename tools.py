# # tools.py
# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import json
# import os
# import logging

# # 配置 logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='training.log',
#     filemode='a'
# )

# # 创建 logger 对象
# logger = logging.getLogger(__name__)

# # 添加控制台输出（可选）
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# # 数据加载
# def load_cifar10_torch():
#     """使用 PyTorch 下载 CIFAR-10 数据并转换为 NumPy 数组"""
#     logger.info("Loading CIFAR-10 dataset...")
#     transform = transforms.ToTensor()
#     trainset = torchvision.datasets.CIFAR10(root=r'D:\download\python_lianxi\data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.CIFAR10(root=r'D:\download\python_lianxi\data', train=False, download=True, transform=transform)

#     label_names = trainset.classes
#     X_train_full = []
#     y_train_full = []
#     for img, label in trainset:
#         img_np = img.numpy().transpose(1, 2, 0)
#         X_train_full.append(img_np)
#         y_train_full.append(label)

#     X_test = []
#     y_test = []
#     for img, label in testset:
#         img_np = img.numpy().transpose(1, 2, 0)
#         X_test.append(img_np)
#         y_test.append(label)

#     X_train_full = np.array(X_train_full)  # (50000, 32, 32, 3)
#     y_train_full = np.array(y_train_full)  # (50000,)
#     X_test = np.array(X_test)              # (10000, 32, 32, 3)
#     y_test = np.array(y_test)              # (10000,)

#     # 保存原始数据用于可视化
#     X_train_raw = X_train_full.copy()  # (50000, 32, 32, 3)
#     X_test_raw = X_test.copy()         # (10000, 32, 32, 3)

#     # 重塑为扁平化向量并标准化
#     X_train_full = X_train_full.reshape(50000, -1).astype(np.float32)  # (50000, 3072)
#     X_test = X_test.reshape(10000, -1).astype(np.float32)              # (10000, 3072)

#     mean = np.mean(X_train_full, axis=0)
#     std = np.std(X_train_full, axis=0) + 1e-7
#     X_train_full = (X_train_full - mean) / std
#     X_test = (X_test - mean) / std

#     # 划分训练集和验证集
#     num_train = 45000
#     indices = np.random.permutation(X_train_full.shape[0])
#     train_idx, val_idx = indices[:num_train], indices[num_train:]

#     X_train = X_train_full[train_idx]  # (45000, 3072)
#     y_train = y_train_full[train_idx]  # (45000,)
#     X_val = X_train_full[val_idx]      # (5000, 3072)
#     y_val = y_train_full[val_idx]      # (5000,)

#     # 划分原始数据（从原始 X_train_raw 中取）
#     X_train_r = X_train_raw[train_idx]  # (45000, 32, 32, 3)
#     X_val_raw = X_train_raw[val_idx]      # (5000, 32, 32, 3)

#     logger.info("CIFAR-10 dataset loaded successfully.")
#     return X_train, y_train, X_val, y_val, X_test, y_test, X_train_r, X_val_raw, X_test_raw, mean, std, label_names

# # 激活函数
# class Activation:
#     @staticmethod
#     def relu(x):
#         return np.maximum(0, x)

#     @staticmethod
#     def relu_deriv(x):
#         return (x > 0).astype(float)

#     @staticmethod
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     @staticmethod
#     def sigmoid_deriv(x):
#         s = Activation.sigmoid(x)
#         return s * (1 - s)

# def softmax(x):
#     # 改进 Softmax 的数值稳定性
#     x = np.clip(x, -500, 500)  # 限制输入范围，避免 exp 溢出
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# # 模型定义
# class ThreeLayerNN:
#     def __init__(self, input_size=3072, hidden1_size=512, hidden2_size=256, output_size=10, activation='relu'):
#         # 初始化权重和偏置
#         self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)  # He 初始化
#         self.b1 = np.zeros((1, hidden1_size))
#         self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
#         self.b2 = np.zeros((1, hidden2_size))
#         self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
#         self.b3 = np.zeros((1, output_size))

#         # 选择激活函数
#         if activation == 'relu':
#             self.activation = Activation.relu
#             self.activation_deriv = Activation.relu_deriv
#         elif activation == 'sigmoid':
#             self.activation = Activation.sigmoid
#             self.activation_deriv = Activation.sigmoid_deriv
#         else:
#             raise ValueError("Unsupported activation function. Use 'relu' or 'sigmoid'.")

#     def forward(self, X):
#         # 前向传播
#         self.Z1 = X @ self.W1 + self.b1
#         self.A1 = self.activation(self.Z1)
#         self.Z2 = self.A1 @ self.W2 + self.b2
#         self.A2 = self.activation(self.Z2)
#         self.Z3 = self.A2 @ self.W3 + self.b3
#         self.A3 = softmax(self.Z3)
#         return self.A3

#     def clip_weights(self, max_norm=1.0):
#         # 权重裁剪，限制权重的最大值
#         self.W1 = np.clip(self.W1, -max_norm, max_norm)
#         self.W2 = np.clip(self.W2, -max_norm, max_norm)
#         self.W3 = np.clip(self.W3, -max_norm, max_norm)

#     def clip_gradients(self, grad, max_norm=1.0):
#         # 梯度裁剪
#         norm = np.linalg.norm(grad)
#         if norm > max_norm:
#             grad = grad * (max_norm / (norm + 1e-6))
#         return grad

#     def backward(self, X, y, output, learning_rate, l2_lambda=0.01):
#         # 反向传播
#         batch_size = X.shape[0]
        
#         # 转换为 one-hot 编码
#         y_one_hot = np.zeros((batch_size, 10))
#         y_one_hot[np.arange(batch_size), y] = 1

#         # 输出层梯度
#         dZ3 = output - y_one_hot
#         dW3 = self.A2.T @ dZ3 + l2_lambda * self.W3  # L2 正则化
#         dW3 = self.clip_gradients(dW3)  # 梯度裁剪
#         db3 = np.sum(dZ3, axis=0, keepdims=True)

#         # 第二隐藏层梯度
#         dA2 = dZ3 @ self.W3.T
#         dZ2 = dA2 * self.activation_deriv(self.Z2)
#         dW2 = self.A1.T @ dZ2 + l2_lambda * self.W2
#         dW2 = self.clip_gradients(dW2)  # 梯度裁剪
#         db2 = np.sum(dZ2, axis=0, keepdims=True)

#         # 第一隐藏层梯度
#         dA1 = dZ2 @ self.W2.T
#         dZ1 = dA1 * self.activation_deriv(self.Z1)
#         dW1 = X.T @ dZ1 + l2_lambda * self.W1
#         dW1 = self.clip_gradients(dW1)  # 梯度裁剪
#         db1 = np.sum(dZ1, axis=0, keepdims=True)

#         # 更新权重和偏置
#         self.W3 -= learning_rate * dW3
#         self.b3 -= learning_rate * db3
#         self.W2 -= learning_rate * dW2
#         self.b2 -= learning_rate * db2
#         self.W1 -= learning_rate * dW1
#         self.b1 -= learning_rate * db1

#         # 权重裁剪
#         self.clip_weights(max_norm=1.0)

#     def compute_loss(self, output, y, l2_lambda=0.01):
#         # 交叉熵损失 + L2 正则化
#         batch_size = y.shape[0]
#         log_probs = -np.log(output[np.arange(batch_size), y] + 1e-10)
#         cross_entropy_loss = np.mean(log_probs)
#         l2_loss = l2_lambda * 0.5 * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
#         return cross_entropy_loss + l2_loss

#     def save_weights(self, filepath):
#         # 保存模型权重
#         weights = {
#             'W1': self.W1.tolist(),
#             'b1': self.b1.tolist(),
#             'W2': self.W2.tolist(),
#             'b2': self.b2.tolist(),
#             'W3': self.W3.tolist(),
#             'b3': self.b3.tolist()
#         }
#         with open(filepath, 'w') as f:
#             json.dump(weights, f)

#     def load_weights(self, filepath):
#         # 加载模型权重
#         with open(filepath, 'r') as f:
#             weights = json.load(f)
#         self.W1 = np.array(weights['W1'])
#         self.b1 = np.array(weights['b1'])
#         self.W2 = np.array(weights['W2'])
#         self.b2 = np.array(weights['b2'])
#         self.W3 = np.array(weights['W3'])
#         self.b3 = np.array(weights['b3'])

# # 训练函数
# def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, learning_rate=0.001, lr_decay=0.1, decay_epochs=10, l2_lambda=0.01, save_path='best_model.json'):
#     num_samples = X_train.shape[0]
#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
#     best_val_acc = 0.0

#     for epoch in range(epochs):
#         # 学习率下降
#         current_lr = learning_rate * (lr_decay ** (epoch // decay_epochs))

#         # 打乱训练数据
#         indices = np.random.permutation(num_samples)
#         X_train_shuffled = X_train[indices]
#         y_train_shuffled = y_train[indices]

#         # 按批次训练
#         for i in range(0, num_samples, batch_size):
#             X_batch = X_train_shuffled[i:i+batch_size]
#             y_batch = y_train_shuffled[i:i+batch_size]

#             # 前向传播
#             output = model.forward(X_batch)
            
#             # 计算损失
#             loss = model.compute_loss(output, y_batch, l2_lambda)
            
#             # 反向传播
#             model.backward(X_batch, y_batch, output, current_lr, l2_lambda)

#         # 计算训练集和验证集的损失和准确率
#         train_output = model.forward(X_train)
#         train_loss = model.compute_loss(train_output, y_train, l2_lambda)
#         train_acc = np.mean(np.argmax(train_output, axis=1) == y_train)

#         val_output = model.forward(X_val)
#         val_loss = model.compute_loss(val_output, y_val, l2_lambda)
#         val_acc = np.mean(np.argmax(val_output, axis=1) == y_val)

#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)

#         # 使用 logger 记录训练信息
#         logger.info(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

#         # 保存最优模型
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             model.save_weights(save_path)
#             logger.info(f"Saved best model with Val Acc: {best_val_acc:.4f}")

#     return history

# # 测试函数
# def evaluate(model, X_test, y_test, weights_path=None):
#     if weights_path:
#         model.load_weights(weights_path)
#     output = model.forward(X_test)
#     test_acc = np.mean(np.argmax(output, axis=1) == y_test)
#     logger.info(f"Test Accuracy: {test_acc:.4f}")
#     return test_acc

# # 超参数搜索
# def hyperparameter_search(X_train, y_train, X_val, y_val, X_test, y_test):
#     # 定义超参数范围
#     learning_rates = [0.001, 0.0001]
#     hidden1_sizes = [256, 512]
#     hidden2_sizes = [128, 256]
#     l2_lambdas = [0.01, 0.1]

#     results = []

#     for lr in learning_rates:
#         for h1_size in hidden1_sizes:
#             for h2_size in hidden2_sizes:
#                 for l2_lambda in l2_lambdas:
#                     logger.info(f"Testing: LR={lr}, Hidden1={h1_size}, Hidden2={h2_size}, L2_lambda={l2_lambda}")
#                     model = ThreeLayerNN(hidden1_size=h1_size, hidden2_size=h2_size, activation='relu')
#                     save_path = f'model_lr{lr}_h1{h1_size}_h2{h2_size}_l2{l2_lambda}.json'
#                     history = train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, learning_rate=lr, l2_lambda=l2_lambda, save_path=save_path)
#                     test_acc = evaluate(model, X_test, y_test, weights_path=save_path)
#                     results.append({
#                         'learning_rate': lr,
#                         'hidden1_size': h1_size,
#                         'hidden2_size': h2_size,
#                         'l2_lambda': l2_lambda,
#                         'val_acc': max(history['val_acc']),
#                         'test_acc': test_acc
#                     })

#     # 保存超参数搜索结果
#     with open('hyperparameter_results.json', 'w') as f:
#         json.dump(results, f, indent=4)
#     return results

# tools.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import json
import os
import logging

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training_1.log',
    filemode='a'
)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 数据增强函数
def data_augmentation(X, flip_prob=0.5, brightness_factor=0.2):
    """简单的数据增强：水平翻转和亮度调整"""
    X_aug = X.copy()
    batch_size = X.shape[0]
    
    # 水平翻转
    for i in range(batch_size):
        if np.random.rand() < flip_prob:
            X_aug[i] = np.fliplr(X_aug[i])
    
    # 亮度调整
    brightness = np.random.uniform(1 - brightness_factor, 1 + brightness_factor)
    X_aug = X_aug * brightness
    X_aug = np.clip(X_aug, 0, 1)  # 确保值在 [0, 1] 范围内
    
    return X_aug

# 数据加载
def load_cifar10_torch():
    logger.info("Loading CIFAR-10 dataset...")
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root=r'D:\download\python_lianxi\data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=r'D:\download\python_lianxi\data', train=False, download=True, transform=transform)

    label_names = trainset.classes
    X_train_full = []
    y_train_full = []
    for img, label in trainset:
        img_np = img.numpy().transpose(1, 2, 0)
        X_train_full.append(img_np)
        y_train_full.append(label)

    X_test = []
    y_test = []
    for img, label in testset:
        img_np = img.numpy().transpose(1, 2, 0)
        X_test.append(img_np)
        y_test.append(label)

    X_train_full = np.array(X_train_full)  # (50000, 32, 32, 3)
    y_train_full = np.array(y_train_full)  # (50000,)
    X_test = np.array(X_test)              # (10000, 32, 32, 3)
    y_test = np.array(y_test)              # (10000,)

    X_train_raw = X_train_full.copy()
    X_test_raw = X_test.copy()

    X_train_full = X_train_full.reshape(50000, -1).astype(np.float32)  # (50000, 3072)
    X_test = X_test.reshape(10000, -1).astype(np.float32)              # (10000, 3072)

    mean = np.mean(X_train_full, axis=0)
    std = np.std(X_train_full, axis=0) + 1e-7
    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std

    num_train = 45000
    indices = np.random.permutation(X_train_full.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    X_train_r = X_train_raw[train_idx]
    X_val_raw = X_train_raw[val_idx]

    logger.info("CIFAR-10 dataset loaded successfully.")
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_r, X_val_raw, X_test_raw, mean, std, label_names

# 激活函数
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

def softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 模型定义
class ThreeLayerNN:
    def __init__(self, input_size=3072, hidden1_size=1024, hidden2_size=512, output_size=10, dropout_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate

        # Adam 优化器参数
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.t = 0  # 时间步

    def dropout(self, X, rate):
        mask = np.random.binomial(1, 1 - rate, size=X.shape) / (1 - rate)
        return X * mask, mask

    def forward(self, X, training=True):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = Activation.relu(self.Z1)
        if training:
            self.A1, self.mask1 = self.dropout(self.A1, self.dropout_rate)
        else:
            self.A1 *= (1 - self.dropout_rate)  # 测试时缩放

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = Activation.relu(self.Z2)
        if training:
            self.A2, self.mask2 = self.dropout(self.A2, self.dropout_rate)
        else:
            self.A2 *= (1 - self.dropout_rate)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def backward(self, X, y, output, learning_rate, l2_lambda=0.01):
        batch_size = X.shape[0]
        y_one_hot = np.zeros((batch_size, 10))
        y_one_hot[np.arange(batch_size), y] = 1

        # 输出层梯度
        dZ3 = output - y_one_hot
        dW3 = self.A2.T @ dZ3 + l2_lambda * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # 第二隐藏层梯度
        dA2 = dZ3 @ self.W3.T
        dA2 *= self.mask2  # 应用 Dropout 掩码
        dZ2 = dA2 * Activation.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 + l2_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 第一隐藏层梯度
        dA1 = dZ2 @ self.W2.T
        dA1 *= self.mask1  # 应用 Dropout 掩码
        dZ1 = dA1 * Activation.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 + l2_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Adam 优化器更新
        self.t += 1
        # W3
        self.m_W3 = self.beta1 * self.m_W3 + (1 - self.beta1) * dW3
        self.v_W3 = self.beta2 * self.v_W3 + (1 - self.beta2) * (dW3 ** 2)
        m_hat_W3 = self.m_W3 / (1 - self.beta1 ** self.t)
        v_hat_W3 = self.v_W3 / (1 - self.beta2 ** self.t)
        self.W3 -= learning_rate * m_hat_W3 / (np.sqrt(v_hat_W3) + self.epsilon)
        # b3
        self.m_b3 = self.beta1 * self.m_b3 + (1 - self.beta1) * db3
        self.v_b3 = self.beta2 * self.v_b3 + (1 - self.beta2) * (db3 ** 2)
        m_hat_b3 = self.m_b3 / (1 - self.beta1 ** self.t)
        v_hat_b3 = self.v_b3 / (1 - self.beta2 ** self.t)
        self.b3 -= learning_rate * m_hat_b3 / (np.sqrt(v_hat_b3) + self.epsilon)
        # W2
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
        m_hat_W2 = self.m_W2 / (1 - self.beta1 ** self.t)
        v_hat_W2 = self.v_W2 / (1 - self.beta2 ** self.t)
        self.W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.epsilon)
        # b2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - self.beta1 ** self.t)
        v_hat_b2 = self.v_b2 / (1 - self.beta2 ** self.t)
        self.b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + self.epsilon)
        # W1
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
        m_hat_W1 = self.m_W1 / (1 - self.beta1 ** self.t)
        v_hat_W1 = self.v_W1 / (1 - self.beta2 ** self.t)
        self.W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.epsilon)
        # b1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - self.beta1 ** self.t)
        v_hat_b1 = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + self.epsilon)

    def compute_loss(self, output, y, l2_lambda=0.01):
        batch_size = y.shape[0]
        log_probs = -np.log(output[np.arange(batch_size), y] + 1e-10)
        cross_entropy_loss = np.mean(log_probs)
        l2_loss = l2_lambda * 0.5 * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return cross_entropy_loss + l2_loss

    def save_weights(self, filepath):
        weights = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'W3': self.W3.tolist(),
            'b3': self.b3.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            weights = json.load(f)
        self.W1 = np.array(weights['W1'])
        self.b1 = np.array(weights['b1'])
        self.W2 = np.array(weights['W2'])
        self.b2 = np.array(weights['b2'])
        self.W3 = np.array(weights['W3'])
        self.b3 = np.array(weights['b3'])

# 训练函数
def train(model, X_train, y_train, X_val, y_val, X_train_raw, epochs=50, batch_size=128, learning_rate=0.001, lr_decay=0.1, decay_epochs=20, l2_lambda=0.01, save_path='best_model.json'):
    num_samples = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        current_lr = learning_rate * (lr_decay ** (epoch // decay_epochs))
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        X_train_raw_shuffled = X_train_raw[indices]

        for i in range(0, num_samples, batch_size):
            X_raw_batch = X_train_raw_shuffled[i:i+batch_size]
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # 数据增强
            X_raw_batch = data_augmentation(X_raw_batch)
            X_batch = X_raw_batch.reshape(len(X_batch), -1).astype(np.float32)
            mean = np.mean(X_batch, axis=0)
            std = np.std(X_batch, axis=0) + 1e-7
            X_batch = (X_batch - mean) / std

            # 前向传播
            output = model.forward(X_batch, training=True)
            loss = model.compute_loss(output, y_batch, l2_lambda)
            model.backward(X_batch, y_batch, output, current_lr, l2_lambda)

        # 计算训练集和验证集的损失和准确率
        train_output = model.forward(X_train, training=False)
        train_loss = model.compute_loss(train_output, y_train, l2_lambda)
        train_acc = np.mean(np.argmax(train_output, axis=1) == y_train)

        val_output = model.forward(X_val, training=False)
        val_loss = model.compute_loss(val_output, y_val, l2_lambda)
        val_acc = np.mean(np.argmax(val_output, axis=1) == y_val)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(save_path)
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    return history

# 测试函数
def evaluate(model, X_test, y_test, weights_path=None):
    if weights_path:
        model.load_weights(weights_path)
    output = model.forward(X_test, training=False)
    test_acc = np.mean(np.argmax(output, axis=1) == y_test)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# 超参数搜索
def hyperparameter_search(X_train, y_train, X_val, y_val, X_test, y_test, X_train_raw, X_val_raw=None):
    learning_rates = [0.001, 0.0001]
    hidden1_sizes = [1024, 2048]
    hidden2_sizes = [512, 1024]
    l2_lambdas = [0.01, 0.05]

    results = []

    for lr in learning_rates:
        for h1_size in hidden1_sizes:
            for h2_size in hidden2_sizes:
                for l2_lambda in l2_lambdas:
                    logger.info(f"Testing: LR={lr}, Hidden1={h1_size}, Hidden2={h2_size}, L2_lambda={l2_lambda}")
                    model = ThreeLayerNN(hidden1_size=h1_size, hidden2_size=h2_size, dropout_rate=0.5)
                    save_path = f'model_lr{lr}_h1{h1_size}_h2{h2_size}_l2{l2_lambda}.json'
                    history = train(model, X_train, y_train, X_val, y_val, X_train_raw, epochs=20, batch_size=128, learning_rate=lr, l2_lambda=l2_lambda, save_path=save_path)
                    test_acc = evaluate(model, X_test, y_test, weights_path=save_path)
                    results.append({
                        'learning_rate': lr,
                        'hidden1_size': h1_size,
                        'hidden2_size': h2_size,
                        'l2_lambda': l2_lambda,
                        'val_acc': max(history['val_acc']),
                        'test_acc': test_acc
                    })

    with open('hyperparameter_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results