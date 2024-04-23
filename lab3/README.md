# Lab3
- 刘兆宸
- PB21061373

## 实验要求
使用 pytorch 或者 tensorflow 的相关神经网络库，编写图卷积神经网络模型 (GCN)，并在相应的图结构
数据集上完成节点分类和链路预测任务，最后分析自环、层数、DropEdge 、PairNorm 、激活函数等
因素对模型的分类和预测性能的影响 。

## GCN 理论
1. Graph卷积公式
$$ 
(f*h)_G = U(U^Th) \odot (U^Uf)
$$
为降低复杂度,利用Chebyshev多项式作为卷积核,
$$ 
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$

## 实验内容
