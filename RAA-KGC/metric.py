import torch

from typing import List


def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
'''
这段代码是用于计算模型输出的预测结果在前 k 个预测中的准确率。以下是代码的解释：

accuracy 函数接受三个参数：

output：模型的输出张量，包含了对每个样本的预测值。
target：目标标签张量，包含了每个样本的真实标签。
topk：一个元组，包含了要计算准确率的前 k 个预测值，默认为 (1, 3)。
进入 torch.no_grad() 上下文管理器，确保在计算准确率时不进行梯度计算。

获取 topk 中的最大值，确定要计算准确率的前 k 个预测。

获取目标标签的批量大小，并存储在 batch_size 中。

使用 output.topk(maxk, 1, True, True) 方法获取模型输出中前 k 个最大值的索引，其中参数含义为：

maxk：要获取的最大值的数量。
1：指定在哪个维度上进行检索，这里是在第 1 个维度（即列）上进行检索。
True, True：指定是否返回值和索引，默认都为 True。
将预测的结果进行转置，使得每个预测结果占据一行。

使用 torch.eq() 方法比较预测结果和目标标签是否相等，得到一个布尔张量 correct，其中每个元素表示预测是否正确。

循环遍历 topk 中的每个 k 值，对每个 k 值计算准确率，并存储在列表 res 中。

对于每个 k 值，首先取出前 k 行的预测结果，并将其展平成一维张量。
使用 torch.sum() 方法计算正确预测的数量，并除以批量大小，乘以 100 得到百分比准确率。
将计算得到的准确率存储在列表 res 中。
返回包含了每个 k 值对应准确率的列表 res。
'''