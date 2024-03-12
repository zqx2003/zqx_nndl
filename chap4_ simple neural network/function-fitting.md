# 函数拟合

## 函数定义

​	使用三角函数sin(x)作为被拟合函数，定义域为-5.0~5.0

## 数据采集

​	训练集使用tf.range在下界-5.0，上界5.0，步长0.01的方式采样，测试集使用linspace在下界-5.0，上界5.0，随机采样100个样本。数据采集代码如下

```python
# 数据生成
goal_func = tf.math.sin
min_val, max_val, step = -5.0, 5.0, 0.01
num_samples = 100

train_X = tf.expand_dims(tf.range(min_val, max_val, step), axis=-1)
train_y = tf.map_fn(goal_func, train_X)
test_X = tf.expand_dims(tf.sort(tf.linspace(min_val, max_val, num_samples)), axis=-1)
test_y = tf.map_fn(goal_func, test_X)
```



## 模型描述

​	使用两层ReLu网络，输入层大小为1，输出层大小为1，隐藏层大小为256，模型代码如下

```python
# 模型定义
class Model:
    def __init__(self, num_hidden):
        # 初始化权重和参数
        self.W1 = tf.Variable(tf.random.normal(shape=(1, num_hidden)), name='Weight1')
        self.b1 = tf.Variable(tf.zeros(shape=(1, num_hidden)), name='bias1')
        
        self.W2 = tf.Variable(tf.random.normal(shape=(num_hidden, 1)), name='Weight2')
        self.b2 = tf.Variable(tf.zeros(shape=(1, 1)), name='bias2')
        
        self.trainable_variables = [self.W1, self.b1, self.W2, self.b2]
  
    def forward(self, x):
        # 前向传播
        h1 = tf.nn.relu(x @ self.W1 + self.b1)
        out = h1 @ self.W2 + self.b2
        return out

# 创建模型
num_hidden = 256
model = Model(num_hidden)

# 定义损失函数和优化器
loss_fn = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam(learning_rate=0.01)
```



## 拟合效果

​	使用Adam优化器，学习率为0.01，训练4000轮，训练集loss和测试集loss变化均趋于平缓，预测曲线的走势与真实曲线的走势基本一致。

![Sin Wave](/sin.png)
