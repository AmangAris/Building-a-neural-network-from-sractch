import numpy as np


import pandas as pd
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt 

class Node:
    """
    Each node in neural networks will have these attributes and methods
    """
    def __init__(self,inputs=[]):
        """
        if the node is operator of "ax + b" , the inputs will be x node , and the outputs
        of this is its successors , and the value is *ax + b*
        """
        self.inputs = inputs 
        self.outputs = []
        self.value = None
        self.gradients = { }
        
        for node in self.inputs:
            node.outputs.append(self) # bulid a connection relationship
            
    def forward(self):
        """Forward propogation
        
        compute the output value based on input nodes and store the value
        into *self.value*
        
        """
        # 虚类
        # 如果一个对象是它的子类，就必须要重新实现这个方法
        raise NotImplemented
        
    def backward(self):
        """Backward propogation
        
        compute the gradient of each input node and store the value
        into *self.gradients*
        
        """
        # 虚类
        # 如果一个对象是它的子类，就必须要重新实现这个方法       
        raise NotImplemented

class Input(Node):
    def __init__(self, name=''):
        Node.__init__(self, inputs=[])
        self.name = name
    
    def forward(self, value=None):
        if value is not None:
            self.value = value
        
    def backward(self):
        self.gradients = {}
        
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost
    
    def __repr__(self):
        return 'Input Node: {}'.format(self.name)

class Linear(Node):
    def __init__(self, nodes, weights, bias):
        self.w_node = weights
        self.x_node = nodes
        self.b_node = bias
        Node.__init__(self, inputs=[nodes, weights, bias])
    
    def forward(self): 
        """compute the wx + b using numpy"""
        self.value = np.dot(self.x_node.value, self.w_node.value) + self.b_node.value
        
    
    def backward(self):
        
        for node in self.outputs:
            #gradient_of_loss_of_this_output_node = node.gradient[self]
            grad_cost = node.gradients[self]
            
            self.gradients[self.w_node] = np.dot(self.x_node.value.T, grad_cost) # loss对w的偏导 = loss对self的偏导 * self对w的偏导
            self.gradients[self.b_node] = np.sum(grad_cost * 1, axis=0, keepdims=False)
            self.gradients[self.x_node] = np.dot(grad_cost, self.w_node.value.T)

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
        self.x_node = node
    
    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))
    
    def forward(self):
        self.value = self._sigmoid(self.x_node.value)
    
    def backward(self):
        y = self.value
        
        self.partial = y * (1 - y)
        
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x_node] = grad_cost * self.partial

class MSE(Node):
    def __init__(self, y_true, y_hat):
        self.y_true_node = y_true
        self.y_hat_node = y_hat
        Node.__init__(self, inputs=[y_true, y_hat])
    
    def forward(self):
        y_true_flatten = self.y_true_node.value.reshape(-1, 1)
        y_hat_flatten = self.y_hat_node.value.reshape(-1, 1)
        
        self.diff = y_true_flatten - y_hat_flatten
        
        self.value = np.mean(self.diff**2)
        
    def backward(self):
        n = self.y_hat_node.value.shape[0]
        
        self.gradients[self.y_true_node] = (2 / n) * self.diff
        self.gradients[self.y_hat_node] =  (-2 / n) * self.diff

def topological_sort(data_with_value):
    feed_dict = data_with_value 
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            ## if n is Input Node, set n'value as 
            ## feed_dict[n]
            ## else, n's value is caculate as its
            ## inbounds

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def training_one_batch(topological_sorted_graph):
    # graph 是经过拓扑排序之后的 一个list
    for node in topological_sorted_graph:
        node.forward()
        
    for node in topological_sorted_graph[::-1]:
        node.backward()

def sgd_update(trainable_nodes, learning_rate=1e-2):
    for t in trainable_nodes:
        t.value += -1 * learning_rate * t.gradients[t]
        
def run(dictionary):
    return topological_sort(dictionary)

def plot_loss(losses):
    sns.set()
    plt.figure(figsize=(8,5))
    plt.xlabel('timestamp')
    plt.ylabel('loss')
    plt.plot(range(len(losses)),losses)

if __name__ == "__main__":
    

    data = pd.read_csv('国家统计局月度数据统计.csv',encoding='gbk')
    print('Shape:',data.shape)
    print('before fillna:',data.isna().sum().sum())
    data.fillna(0,inplace=True)
    print('after fillna:',data.isna().sum().sum())

    X_names = data['时间']
    Y_names = '客运量当期值(万人)'
    X_ = data[[i for i in data.columns if i not in ['时间','客运量当期值(万人)']]]
    Y_ = data['客运量当期值(万人)'].values

    X_ = (X_ - np.mean(X_ , axis=0)) / np.std(X_ , axis = 0)
    print(X_.shape)

    n_features = X_.shape[1]
    n_hidden = 10
    n_hidden_2 = 10

    W1_ = np.random.randn(n_features , n_hidden)
    b1_ = np.zeros(n_hidden)

    W2_ = np.random.randn(n_hidden,1)
    b2_ = np.zeros(1)

    X, Y = Input(name='X'), Input(name='y')  # tensorflow -> placeholder
    W1, b1 = Input(name='W1'), Input(name='b1')
    W2, b2 = Input(name='W2'), Input(name='b2')

    linear_output = Linear(X, W1, b1)
    sigmoid_output = Sigmoid(linear_output)
    Yhat = Linear(sigmoid_output, W2, b2)
    loss = MSE(Y, Yhat)

    input_node_with_value = { # -> feed_dict
        X:X_,
        Y:Y_,
        W1:W1_,
        W2:W2_,
        b1:b1_,
        b2:b2_
    }
    graph = topological_sort(input_node_with_value)

    epoch_s = []
    losses = []
    epochs = 50000

    batch_size = 64
    steps_per_epoch = X_.shape[0] // batch_size
    learning_rate = 0.2

    for i in range(epochs):
        loss = 0
        
        for batch in range(steps_per_epoch):

            X_batch, Y_batch = resample(X_, Y_, n_samples=batch_size)
            
            X.value = X_batch
            Y.value = Y_batch
            
            
            training_one_batch(graph)
            sgd_update(trainable_nodes=[W1, W2, b1, b2], learning_rate=learning_rate)
            
            loss += graph[-1].value
            
        if i % 100 == 0:
            print('Epoch: {}, loss = {:.3f}'.format(i+1, loss/steps_per_epoch))
            epoch_s.append(i+1)
            losses.append(loss)

    plot_loss(losses)