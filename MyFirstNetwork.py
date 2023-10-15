import numpy as np
import createDataAndPlot as cp

NETWORK_SHAPE = [2, 50, 100, 50, 2]

# 标准化函数
def normalize(array):
    max_number = np.max(np.absolute(array),axis=1, keepdims=True)
    # if max_number=0 ---> return 0
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm  = array * scale_rate
    return norm 

# 权重矩阵和偏置值
def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs,n_neurons)

def create_biases(n_neurons):
    return np.random.randn(n_neurons)

# 分类函数
def classify(probabilities):
    classification = np.rint(probabilities[:,1])
    return classification

# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0,inputs)

# softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    slided_inputs = inputs - max_values     #为了防止指数爆炸，在指数函数的负区间
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_values = exp_values/norm_base

    return norm_values

# 定义一层
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # self.output = activation_ReLU(sum1)
        return self.output
    
# 定义网络类
class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)

    # 前馈运算函数
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers) - 1:
                layer_output = activation_ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        return outputs

def main():
    data = cp.create_data(10)
    print(data)
    inputs = data[:,(0,1)]
    print(inputs)
    network = Network(NETWORK_SHAPE)
    network.network_forward(inputs)

main()