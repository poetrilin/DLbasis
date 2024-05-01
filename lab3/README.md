# Lab3
- 刘兆宸
- PB21061373

## 实验要求
使用 pytorch 或者 tensorflow 的相关神经网络库，编写图卷积神经网络模型 (GCN)，并在相应的图结构
数据集上完成节点分类和链路预测任务，最后分析自环、层数、DropEdge 、PairNorm 、激活函数等
因素对模型的分类和预测性能的影响 。

## 基本理论
1. 预处理
   用PyG(`torch_geometric.data`)加载数据集,其中数据以`Data`格式存储,包含`x`节点特征,`edge_index`边索引,`y`标签等属性

2. Graph卷积公式
$$ 
(f*h)_G = U[(U^Tx) \odot (U^Tg)] = Ug_{\theta}U^Tx 
$$
其中$g_{\theta}=diag(U^Tg)$ ,在GCN中,用一阶Chebyshev多项式近似,i.e

$Y = (𝑰+𝑫^{−1/2} 𝑨𝑫^{−1/2})X\theta$或$Y= (\hat 𝑫^{−1/2} \hat{𝑨}\hat{𝑫}^{−1/2})X\theta$,其中$\hat{D} = D+I,\hat{A}=A+I$

我们要实现graph convolutional layer,相当于需要根据边矩阵重构"邻接矩阵"
简要Code:
```python
def edge2adj(self, edge_index: Tensor, num_nodes: int, add_self_loops: bool = True) -> Tensor:
        """
        get and normalize the adjacency matrix from edge_index whose shape is (2, num_edges)
        if add_self_loops:
            \hat A = A + I, \hat D = D + I, return \hat D^{-1/2} \hat A \hat D^{-1/2}
        else: 
            return I+D^{-1/2} A D^{-1/2}
        default add self loops
        """
        A = torch.zeros(num_nodes, num_nodes)
        for i in range(edge_index.size(1)):
            A[edge_index[0, i], edge_index[1, i]] += 1
        D = A.sum(dim=1)
        if add_self_loops:
            D = D + 1
            A = A+torch.eye(num_nodes)
        D_inv_sqrt = torch.sqrt(1.0/D)
        D_inv = D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(1, -1)
        adj_t = D_inv * A  # element-wise product
        if add_self_loops:
            return adj_t
        else:
            return torch.eye(num_nodes) + adj_t
```

3. 链路负采样

对于链路预测任务,我们采用Encoder-Decoder框架,其中Encoder部分是两层简单的的GCN,Decoder部分是对边矩阵一个简单的内积.

我们需要负采样来训练模型,即对于每一个正样本,我们随机生成一个负样本,使得负样本的边不存在于图中。
采集负样本，负样本与正样本相对，比如数据集中两个节点间存在链接，那么为一个正样本，两个节点间在已知数据集中不存在链接，那么构成一个负样本。

根据数据集中的正样本，可以反向获取到大量的负样本，从中随机采集一部分负样本用于训练。


## 实验内容
### 网络结构
1. 分类任务
基于上面的GraphConvolution操作可以实现自定义的GCN,主要包括以下2个部分:
  1. **GCN Layer** 结构是DropEdge,GraphConv,PairNorm,激活函数的堆叠
  2. **out_layer** 是一个简单的Dropout和GraphConv,最后取argmax得出分类
2. 链路预测
   1. **Encoder** 是两层GCNConv
   2. **Decoder** 是一个简单的内积，类似于边的相似度

### 超参数调整
#### 自环
   一开始没开自环,结果很差Val_acc 几乎训完只有20多,应该是梯度出了问题(可能是梯度爆炸了)，开了自环后,效果明显提升，

   可能是因为卷积对局部一致性的要求,自环可以保证每个节点的特征不会丢失,
   并且理论也有证明add_self_loops可以保证归一化的邻接矩阵特征值更加稳定
![](./img/1.png)

```shell
At last, Epoch: 299
----------
Train loss: 0.6633 | Train acc: 0.9143
  Val loss: 1.2330 |   Val acc: 0.7380
```
但是从图中发现还是存在明显的过拟合现象,于是想用些tricks来解决这个问题

#### 层数
- l = 2 可以见上图 

- l = 3 
![](./img/l-3.png)
```shell
At last, Epoch: 299
----------
Train loss: 1.2579 | Train acc: 0.9143
  Val loss: 1.6209 |   Val acc: 0.7260
```

- l=4
![](./img/l-4.png)
```shell
At last, Epoch: 296
----------
Train loss: 0.2236 | Train acc: 0.9643
  Val loss: 1.1535 |   Val acc: 0.6660
```
  在l=4时可以发现过拟合更加严重了,从上看上去l=2是最好的选择
  (收敛较快且过拟合较轻),理论上分析也确实如此,邻接矩阵的特征值分布会随着层数的增加而变得更加分散,导致梯度爆炸或者消失,
  而较浅的网络可以更好的保留特征信息。

#### PairNorm 
   PairNorm是一种归一化方法,可以保证卷积后特征的稳定性,类似CNN里的BatchNorm.

$$
\begin{aligned}
\mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\
\mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}
\end{aligned}
$$
开启PairNorm后,效果如下:
![](./img/pair_norm.png)
```shell
Epoch: 280
----------
Train loss: 0.3690 | Train acc: 0.9857
  Val loss: 1.6320 |   Val acc: 0.5260
```
效果反而变差了,可能是因为数据集本身的特性,因为初始化时训练集：验证集：测试集=140：500：1000

于是按8:1:1重新划分了训练集,效果如下:
- 开启PairNorm
  ![](./img/pair_norm.png)
  ```shell

#### DropEdge 



#### 激活函数

 