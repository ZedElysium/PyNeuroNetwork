import numpy as np
import createDataAndPlot as cp
import copy

NETWORK_SHAPE = [2, 50, 100, 50, 2]
BATCH_SIZE = 5
LEARNING_RATE = 0.01

# 标准化函数
def normalize(array):
    max_number = np.max(np.absolute(array),axis=1, keepdims=True)
    # if max_number=0 ---> return 0
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm  = array * scale_rate
    return norm 
# 向量标准化函数
def vector_normalize(array):
    max_number = np.max(np.absolute(array))
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

# 损失函数
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:,1] = real
    real_matrix[:,0] = 1 - real
    product = np.sum(predicted*real_matrix, axis=1)
    return 1 - product

# 需求函数
def get_final_layer_preAct_damands(predicted_values, target_vector):
    target = np.zeros((len(target_vector),2))
    target[:, 1] = target_vector
    target[:, 0] = 1-target_vector

    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target


# 定义一层
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # self.output = activation_ReLU(sum1)
        return self.output
    
    def layer_backward(self, preWeights_values,afterWeighs_demands):
        preWeights_demands = np.dot(afterWeighs_demands, self.weights.T)

        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        preActs_demands = value_derivatives * preWeights_demands
        norm_preActs_demands = normalize(preActs_demands)

        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeighs_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)

        return (norm_preActs_demands, norm_weight_adjust_matrix)

    # 调整矩阵
    def get_weight_adjust_matrix(self, preWeights_values, aftWeights_demands):
        plain_weights = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)
        plain_weights_T = plain_weights.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T * preWeights_values[i,:]).T * aftWeights_demands[i,:]
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix
    
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
    
    # 反向传播函数
    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self)
        preAct_demands = get_final_layer_preAct_damands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - i - 1] #倒序
            if i != 0:
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0)
                layer.biases = vector_normalize(layer.biases)
            outputs = layer_outputs[len(layer_outputs) - i - 2] #再前一层
            result_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = result_list[0]
            weights_adjust_matrix = result_list[1]
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)
        
        return backup_network



def main():
    data = cp.create_data(10)
    print(data)
    inputs = data[:,(0,1)]
    print(inputs)
    network = Network(NETWORK_SHAPE)
    network.network_forward(inputs)

main()