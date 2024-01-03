# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear

## 组网
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST
#定义LeNet网络结构

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet,self).__init__()
        #创建卷积层和池化层
        #创建第一个卷积层
        self.conv1 = Conv2D(in_channels=1,out_channels=6,kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        #尺寸的逻辑：池化层未改变通道数，当前通道为6
        #创建第二个卷积层
        self.conv2 = Conv2D(in_channels=6,out_channels=16,kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        #创建第三个卷积层
        self.conv3 = Conv2D(in_channels=16,out_channels=120,kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
# 飞桨会根据实际图像数据的尺寸和卷积核参数自动推断中间层数据的W和H等，只需要用户表达通道数即可。
# 下面的程序使用随机数作为输入，查看经过LeNet-5的每一层作用之后，输出数据的形状。

# 输入数据形状是 [N, 1, H, W]
# 这里用np.random创建一个随机数组作为输入数据
x = np.random.randn(*[3,1,28,28])
x = x.astype('float32')

# 创建LeNet类的实例，指定模型名称和分类的类别数目
model = LeNet(num_classes=10)

# 通过调用LeNet从基类继承的sublayers()函数，
# 查看LeNet中所包含的子层
print(model.sublayers())
x = paddle.to_tensor(x)

for item in model.sublayers():
    #item是LeNet类中的一个子层
    #查看经过子层之后的输出数据形状
    try:
        x = item(x)
    except:
        x = paddle.reshape(x, [x.shape[0], -1])
        x = item(x)
    if len(item.parameters())==2:
        # 查看卷积和全连接层的数据和参数的形状，
        # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
        print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
    else:
        # 池化层没有参数
        print(item.full_name(), x.shape)

# 设置迭代轮数
EPOCH_NUM = 5
#定义训练过程 
def train(model,opt,train_loader,valid_loader):
    print("start training ... ")
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1] 
            #计算模型输出
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 2000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))

            #反向传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []

        for batch_id, data in enumerate(valid_loader()):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            pred = F.softmax(logits)

            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')    

# 创建模型
model = LeNet(num_classes=10)
# 设置迭代轮数
EPOCH_NUM = 5
# 设置优化器为Momentum，学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# 定义数据读取器
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
# 启动训练过程
train(model, opt, train_loader, valid_loader)