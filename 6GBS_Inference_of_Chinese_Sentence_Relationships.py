import torch
from datasets import load_dataset
import random
import passKeyInformation
import GPUandCPU_allocation



#1. 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        # huggingface加载数据集，本地数据集会先load，然后放到.cache文件夹下面去
        # 本地数据集位置：/Users/fangzhou/.cache/huggingface/datasets/lansinuote___parquet/lansinuote--ChnSentiCorp-eaea6a9750cb0fe7/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec
        # split: 如果知道数据的结构，在load的时候就可以用split只load进来一部分数据；
        datasets = load_dataset(path='lansinuote/ChnSentiCorp') #lansinuote/ChnSentiCorp是已经经过标注的中文情感分析数据集
        print(datasets)  # 查看数据的结构
        dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)
        print(dataset)  # 查看数据的结构

        def f(data):
            return len(data['text']) > 40 #长度大于40个字因为我们需要把一句话分为前半句和后半句
        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']

        #切分一句话为前半句和后半句
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = 0

        #有一半的概率把后半句替换为一句无关的话
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            sentence2 = self.dataset[j]['text'][20:40]
            label = 1

        return sentence1, sentence2, label

# dataset = Dataset('train')
# sentence1, sentence2, label = dataset[0]
# print("=====1.len(dataset), sentence1, sentence2, label======")
# print(len(dataset), sentence1, sentence2, label)
# print("==========")



#2. 加载字典和分词工具
# from transformers import BertTokenizer
# token = BertTokenizer.from_pretrained('bert-base-chinese')
# print("=====2.token=====")
# print(token)
# print("==========")


#3. 定义批处理函数
#数据集中数据是一条一条的，在训练的时候应该是一批一批来处理这些数据
#我们可以使用 Hugging Face 提供的默认 collate_fn 函数，它会将每个样本转换成一个字典，并将所有样本组合成一个字典列表。这个字典中包含了每个样本的 input_ids、attention_mask 和 labels。
def collate_fn(data):
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True, #当句子长度大于max_length时,截断
                                   padding='max_length', #一律补pad到max_length长度
                                   max_length=45,
                                   return_tensors='pt',
                                   return_length=True,
                                   add_special_tokens=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    #token_type_ids:第一个句子和特殊符号的位置是0,第二个句子的位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


#数据加载器
# loader = torch.utils.data.DataLoader(dataset=dataset,
#                                      batch_size=8,#输入的 dataset 分成大小为 8 的批次
#                                      collate_fn=collate_fn,#使用 collate_fn 函数来处理每个批次中的样本
#                                      shuffle=True, #在训练时打乱样本的顺序
#                                      drop_last=True) #表示当最后一批数据的样本数量不足 8 个时，会丢弃这些样本。
#
# for i, (input_ids, attention_mask, token_type_ids,
#         labels) in enumerate(loader):
#     break
#
# print("=====3=====")
# print("len(loader):",len(loader))
# print("token.decode(input_ids[0]):",token.decode(input_ids[0]))
# print("input_ids.shape: ", input_ids.shape, "attention_mask.shape: ", attention_mask.shape, "token_type_ids.shape: ",token_type_ids.shape, "labels: ",labels)
# print("==========")


#4. 加载预训练模型
# from transformers import BertModel
# pretrained = BertModel.from_pretrained('bert-base-chinese')


#不训练,不需要计算梯度False
# for param in pretrained.parameters():
#     param.requires_grad_(False)
#输出Inference_of_Chinese_Sentence_Relationships传递过来的parms
# print("=====4=====")


#模型试算
# out = pretrained(input_ids=input_ids,
#            attention_mask=attention_mask,
#            token_type_ids=token_type_ids)
#
# print("out.last_hidden_state.shape: ", out.last_hidden_state.shape) #每个批次8句话，每句话45字，每个字768维度
# print("==========")



#5. 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2) #只包含单层全连接层，2表示2分类

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0]) #只用第0个词特征就可以分类了

        out = out.softmax(dim=1)

        return out

# model = Model()
# print("=====5.model=====")
# print(model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).shape)
# print("==========")


# print("=====更新参数=====")
# temp = 0 #控制打印的参数个数
# for name, parms in model.named_parameters():
#     temp += 1
#     if temp == 2:
#         break
#     parms = passKeyInformation.passParms()
#     loaded_parms = passKeyInformation.passParms();
# print(loaded_parms)
# print("==========")

#6. 训练
# from transformers import AdamW
# '''
# 优化器的类型是AdamW，它将被用于调整神经网络模型中的参数，以尽可能地减小训练误差。
#
# 具体来说，代码中的model.parameters()表示我们希望优化的参数来自于神经网络模型中的所有可学习参数。
# lr=5e-4则是指定了优化器的学习率，即每次更新模型参数时所使用的步长大小。在这个例子中，学习率为0.0005。
#
# AdamW 是一种基于梯度下降的优化算法，它结合了Adam和L2正则化两种方法。
# 其中Adam是一种自适应性优化算法，可以根据每个参数的历史梯度信息调整学习率，从而加速收敛。
# 而L2正则化则可以通过惩罚较大的权重值，避免模型过拟合的问题。
# AdamW是Adam算法的一种改进版本，它在Adam的基础上引入了权重衰减（weight decay）的概念，类似于L2正则化的作用，可以进一步提高模型的泛化能力。
# '''
# optimizer = AdamW(model.parameters(), lr=5e-4)
#
#
# '''
# 这段代码使用了PyTorch中的交叉熵损失函数(CrossEntropyLoss)，用于测量模型在分类问题中的性能。
# 交叉熵损失函数通常用于多分类问题，其中模型需要将输入的数据划分到多个不同的类别中。
#
# CrossEntropyLoss函数接收两个参数：模型的输出和目标类别。
# 在分类问题中，模型的输出通常是一个向量，每个元素表示该样本属于相应类别的概率得分，而目标类别表示样本实际所属的类别。
#
# 交叉熵损失函数会计算模型预测与实际标签之间的距离，并返回一个标量作为损失值，同时也可以用于反向传播，通过优化器更新网络参数。
# 在这里，我们使用torch.nn.CrossEntropyLoss()来定义损失函数criterion，它已经被PyTorch封装好，无需手动编写。
# '''
# criterion = torch.nn.CrossEntropyLoss()
#
#
# '''
# model.train() 是 Hugging Face 库中的一个方法，用于将模型设置为训练模式。
# 调用此方法时，模型会启用一些仅在训练期间使用的功能，例如dropout正则化和梯度计算。
#
# 在训练期间，模型的参数是根据模型预测值与真实标签之间的损失计算而更新的，使用反向传播算法实现。
# model.train() 确保这个过程所需的计算得以启用。
#
# 需要注意的是，在训练完成后，应该调用 model.eval() 方法将模型切换回评估模式，以禁用这些功能以获得准确的预测结果。
# '''
# model.train()
#
#
# '''
# 首先，使用 enumerate() 函数对数据进行迭代，获取每个 batch 的输入数据 input_ids、attention_mask、token_type_ids 和标签 labels。
# 然后，将这些数据传入模型中，得到模型的输出 out。
# 接着，计算模型的预测值和真实标签之间的损失 loss，并通过调用 loss.backward() 计算梯度。
# 之后，使用优化器 optimizer 对模型参数进行更新，以最小化损失函数。调用 optimizer.step() 来执行此更新操作。
# 最后，使用 optimizer.zero_grad() 来清除梯度缓存，确保每个 batch 的梯度计算都是独立的。
# 这个过程会不断重复，直到完成所有 epoch 的训练。
# '''
# print("=====6.i, loss.item(), accuracy=====")
# for i, (input_ids, attention_mask, token_type_ids,
#         labels) in enumerate(loader):
#     out = model(input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids)
#
#     loss = criterion(out, labels)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
#     if i % 5 == 0:
#         out = out.argmax(dim=1) #将模型输出 out 的每一行的最大值所对应的下标作为预测结果，并将其储存在 out 变量中
#         accuracy = (out == labels).sum().item() / len(labels) #这里使用 .item() 方法将张量转换为 Python 标量。
#
#         print(i, loss.item(), accuracy)
#
#     if i == 50:
#         break
# print("==========")



#7. 测试
def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)
    # print("=====7.test=====")
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break

        print(i)
        '''
        首先，使用 torch.no_grad() 上下文管理器，禁用梯度计算。
        这是因为在测试过程中，我们只关心模型的预测准确率，而不需要更新模型参数。

        然后，将测试数据 input_ids、attention_mask 和 token_type_ids 传入模型中，得到模型输出 out。

        接着，对 out 的每一行取最大值所对应的下标作为预测结果，并与真实标签 labels 进行比较。
        正确预测的样本数量被累加到变量 correct 中，总样本数量被累加到变量 total 中。

        在测试完成后，通过计算 correct/total 得到测试集上的模型精度，并将其作为函数的返回值返回。

        需要注意的是，在测试集上进行评估时，不能调用 loss.backward() 和 optimizer.step()，因为这些操作会更新模型参数，
        而验证集上并不需要这么做。
        '''
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)
# test()
# print("==========")




# 执行
print("=====6GBS_Inference_of_Chinese_Sentence_Relationships=====")

# 分配资源
device = GPUandCPU_allocation.allocation()

#1. 定义数据集
dataset = Dataset('train')
sentence1, sentence2, label = dataset[0]
print("=====1.len(dataset), sentence1, sentence2, label======")
print(len(dataset), sentence1, sentence2, label)
print("==========")

#2. 加载字典和分词工具
from transformers import BertTokenizer
token = BertTokenizer.from_pretrained('bert-base-chinese')
print("=====2.token=====")
print(token)
print("==========")

#3. 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,#输入的 dataset 分成大小为 8 的批次
                                     collate_fn=collate_fn,#使用 collate_fn 函数来处理每个批次中的样本
                                     shuffle=True, #在训练时打乱样本的顺序
                                     drop_last=True) #表示当最后一批数据的样本数量不足 8 个时，会丢弃这些样本。

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print("=====3=====")
print("len(loader):",len(loader))
print("token.decode(input_ids[0]):",token.decode(input_ids[0]))
print("input_ids.shape: ", input_ids.shape, "attention_mask.shape: ", attention_mask.shape, "token_type_ids.shape: ",token_type_ids.shape, "labels: ",labels)
print("==========")

#4. 加载预训练模型
from transformers import BertModel
pretrained = BertModel.from_pretrained('bert-base-chinese')
#不训练,不需要计算梯度False
for param in pretrained.parameters():
    param.requires_grad_(False)
#输出Inference_of_Chinese_Sentence_Relationships传递过来的parms
print("=====4=====")

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)

print("out.last_hidden_state.shape: ", out.last_hidden_state.shape) #每个批次8句话，每句话45字，每个字768维度
print("==========")


#5. 定义下游模型
model = Model()
print("=====5.model=====")
print(model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)
print("==========")

print("=====更新参数=====")
temp = 0 #控制打印的参数个数
for name, parms in model.named_parameters():
    temp += 1
    if temp == 2:
        break
    pki = passKeyInformation.passClass()
    parms = pki.get_parms()
    loaded_parms = pki.get_parms();
print(loaded_parms)
print("==========")

#6. 训练
from transformers import AdamW
'''
优化器的类型是AdamW，它将被用于调整神经网络模型中的参数，以尽可能地减小训练误差。

具体来说，代码中的model.parameters()表示我们希望优化的参数来自于神经网络模型中的所有可学习参数。
lr=5e-4则是指定了优化器的学习率，即每次更新模型参数时所使用的步长大小。在这个例子中，学习率为0.0005。

AdamW 是一种基于梯度下降的优化算法，它结合了Adam和L2正则化两种方法。
其中Adam是一种自适应性优化算法，可以根据每个参数的历史梯度信息调整学习率，从而加速收敛。
而L2正则化则可以通过惩罚较大的权重值，避免模型过拟合的问题。
AdamW是Adam算法的一种改进版本，它在Adam的基础上引入了权重衰减（weight decay）的概念，类似于L2正则化的作用，可以进一步提高模型的泛化能力。
'''
optimizer = AdamW(model.parameters(), lr=5e-4)


'''
这段代码使用了PyTorch中的交叉熵损失函数(CrossEntropyLoss)，用于测量模型在分类问题中的性能。
交叉熵损失函数通常用于多分类问题，其中模型需要将输入的数据划分到多个不同的类别中。

CrossEntropyLoss函数接收两个参数：模型的输出和目标类别。
在分类问题中，模型的输出通常是一个向量，每个元素表示该样本属于相应类别的概率得分，而目标类别表示样本实际所属的类别。

交叉熵损失函数会计算模型预测与实际标签之间的距离，并返回一个标量作为损失值，同时也可以用于反向传播，通过优化器更新网络参数。
在这里，我们使用torch.nn.CrossEntropyLoss()来定义损失函数criterion，它已经被PyTorch封装好，无需手动编写。
'''
criterion = torch.nn.CrossEntropyLoss()


'''
model.train() 是 Hugging Face 库中的一个方法，用于将模型设置为训练模式。
调用此方法时，模型会启用一些仅在训练期间使用的功能，例如dropout正则化和梯度计算。

在训练期间，模型的参数是根据模型预测值与真实标签之间的损失计算而更新的，使用反向传播算法实现。
model.train() 确保这个过程所需的计算得以启用。

需要注意的是，在训练完成后，应该调用 model.eval() 方法将模型切换回评估模式，以禁用这些功能以获得准确的预测结果。
'''
model.train()


'''
首先，使用 enumerate() 函数对数据进行迭代，获取每个 batch 的输入数据 input_ids、attention_mask、token_type_ids 和标签 labels。
然后，将这些数据传入模型中，得到模型的输出 out。
接着，计算模型的预测值和真实标签之间的损失 loss，并通过调用 loss.backward() 计算梯度。
之后，使用优化器 optimizer 对模型参数进行更新，以最小化损失函数。调用 optimizer.step() 来执行此更新操作。
最后，使用 optimizer.zero_grad() 来清除梯度缓存，确保每个 batch 的梯度计算都是独立的。
这个过程会不断重复，直到完成所有 epoch 的训练。
'''
print("=====6.i, loss.item(), accuracy=====")
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1) #将模型输出 out 的每一行的最大值所对应的下标作为预测结果，并将其储存在 out 变量中
        accuracy = (out == labels).sum().item() / len(labels) #这里使用 .item() 方法将张量转换为 Python 标量。

        print(i, loss.item(), accuracy)

    if i == 10:
        break
print("==========")

#7. 测试
print("=====7.test=====")
test()
print("==========")

print("=====end=====")