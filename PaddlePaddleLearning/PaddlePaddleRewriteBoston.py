#加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

#数据划分函数不依赖库，还是自己读
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
   
    max_values = maximums
    min_values = minimums
    
    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data



#模型定义的实质是定义线性回归的网络结构，飞桨建议通过创建Python类的方式完成模型网络的定义
#该类需要继承paddle.nn.Layer父类，并且在类中定义init函数和forward函数。
#forward函数是框架指定实现前向计算逻辑的函数，程序在调用模型实例时会自动执行，forward函数中使用的网络层需要在init函数中声明。

class Regressor(paddle.nn.Layer):
    #self代表对象自身
    def __init__(self):
        #初始化父类的参数
        super(Regressor, self).__init__()
        #定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=13, out_features=1)

    #网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x
    
# 验证数据集读取程序的正确性
training_data, test_data = load_data()
print(training_data)
#声明定义好的线性回归模型
model = Regressor()
#开启模型训练模式，一个模型有两种模式，一种是预测模式，一种是训练模式，一个是产出，一个是投入
model.train()

#定义优化算法，使用随机梯度下降SGD
#设置学习率

opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

#训练过程采用二层循环嵌套方式
#内层循环： 负责整个数据集的一次遍历，采用分批次方式（batch）。
#假设数据集样本数量为1000，一个批次有10个样本,则遍历一次数据集的批次数量是1000/10=100，即内层循环需要执行100次。
#for iter_id, mini_batch in enumerate(mini_batches):

#外层循环： 定义遍历数据集的次数，通过参数EPOCH_NUM设置。
# for epoch_id in range(EPOCH_NUM):

#正式的训练中，需要做几件事：
# 1.数据准备：将一个批次的数据先转换成nparray格式，再转换成Tensor格式
# 2.前向计算：将一个批次的样本数据灌入网络中，计算出结果
# 3.计算损失函数：以前向计算的结果和真是房价作为输入，通过算是函数sqare_error_cost计算出损失函数。
# 4.反向传播：执行梯度反向传播backward函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数(opt.step函数)。

EPOCH_NUM = 10   # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:]) # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor的格式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        
        # 前向计算
        predicts = model(house_features)
        
        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}".format(epoch_id))
            print("iter: {}".format(str(iter_id)))
            print("loss is : {}".format(float(avg_loss)))
        
        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()


# 结束模型训练之后，可以保存模型参数，文件名为LR_model.pdparams
# paddle.save(model.state_dict(), 'LR_model.pdparams')
# print("模型保存成功，模型参数保存在LR_model.pdparams中")
        
def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])
    idx = -10
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label        

# 参数为保存模型参数的文件地址
#读取保存模型
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data, label = load_one_example()
# 将数据转为动态图的variable格式 
one_data = paddle.to_tensor(one_data)
#model是定义的模型，这个model(one_data)实际上是对one_Data进行了一次前向传播
predict = model(one_data)


# 因为这个predict的值实际上是做了归一化处理的，所以这里需要进行反归一化处理
predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]
# 对label数据做反归一化处理
label = label * (max_values[-1] - min_values[-1]) + min_values[-1]

#模型预测值是22.72234,，实际值是19.700000762939453
print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))