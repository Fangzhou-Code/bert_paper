'''
Merkle树是一种哈希树，它可以用来验证大量数据的完整性。在Merkle树中，每个叶子节点表示一个数据块的哈希值，而每个非叶子节点表示两个相邻的子节点的哈希值的拼接结果。根节点的哈希值即为整个Merkle树的哈希值。

该程序中的compress_data函数将数据按照固定大小划分成块，并计算每个块的哈希值，然后构建Merkle树并返回根节点的哈希值以及数据块数量。具体来说，compress_data函数的实现步骤如下：

将输入的字符串数据按照固定大小（这里设为1024字节）划分成若干个块。
对每个块计算SHA-256哈希值，并将所有哈希值放入一个列表中。
使用Merkle树算法（该程序中使用迭代方法）构建哈希树，得到根节点的哈希值。
返回根节点的哈希值以及数据块数量。

decompress_data函数则是对压缩后的数据进行解压缩的操作，它需要传入根节点的哈希值以及数据块数量。具体来说，decompress_data函数的实现步骤如下：
    构建一个由空字符串组成的列表，其长度为数据块数量加一。
    将根节点的哈希值放入列表最后一个位置。
    逆向迭代计算每个块的哈希值，直到计算出第一个块的哈希值。
    将所有块拼接起来并返回结果。
    需要注意的是，该程序中在使用Merkle树算法时可能会出现递归深度过大的问题，这里我们根据实际需求将代码修改为了使用迭代方式构建Merkle树。
'''
import hashlib

class MerkleTree:

    #初始化
    def __init__(self, transactions):
        self.transactions = transactions #其中包含了当前所有的交易记录。
        self.past_transaction = [] #用于存储之前的交易记录。
        self.tree = [] #将用于存储构建出的树形数据结构。
        self.build()


    #构建merkle tree
    def build(self):
        transactions = self.transactions
        print(type(self.transactions), len(self.transactions), transactions)
        past_transaction = self.past_transaction
        tree = self.tree
        n = len(transactions)
        print("n=",n)
        if n % 2 == 1: #判断交易记录数是否为奇数，如果是则在列表末尾添加一个空字符串。
            transactions.append('')
            n += 1
        # 迭代构建Merkle树
        while len(transactions) > 1:
            #在每次迭代中，遍历当前交易记录，每两个记录一组进行哈希运算，并将计算结果加入到 tree 列表中。
            if len(transactions) == 2:
                current = transactions[0]
                current_right = transactions[1]
                past_transaction.append(current)
                past_transaction.append(current_right)
                hash_value = hashlib.sha256((str(current) + str(current_right)).encode('utf-8')).hexdigest()
                tree.append(hash_value)
            else:
                for index in range(0, len(transactions)-2, 2):
                    current = transactions[index]
                    current_right = transactions[index+1]
                    past_transaction.append(current)
                    past_transaction.append(current_right)
                    hash_value = hashlib.sha256((str(current) + str(current_right)).encode('utf-8')).hexdigest()
                    tree.append(hash_value)
            transactions = tree.copy() #每次把之前产生的哈希值给到transactions重新迭代
            print(type(self.transactions), len(self.transactions), transactions)
            past_transaction.clear()
            tree.clear()

        # 将最终的根哈希值赋值给实例变量
        self.transactions = transactions
        print(type(self.transactions), len(self.transactions),transactions)

    def get_root(self):
        return self.transactions

'''
这个函数首先通过列表解析将data分割成若干块，并将这些块存储在一个列表中。
接着，使用MerkleTree类创建一个Merkle树。这个类是一个自定义类，它实现了Merkle树的构建和相关方法。
最后，函数返回Merkle树的根节点哈希值以及块的数量。
'''
def compress_data(data):
    # 将数据按照固定大小划分成块
    block_size = 1024
    blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]

    # 计算每个块的哈希值并构建Merkle树
    merkle_tree = MerkleTree(blocks)

    # 返回Merkle树的根节点哈希值和数据块数量
    return merkle_tree.get_root(), len(blocks)


# def decompress_data(root_hash, num_blocks):
#     # 构建由空字符串组成的列表，其长度为块数加一
#     blocks = [''] * (num_blocks + 1)
#     # 将Merkle树的根节点哈希值放入最后一个位置
#     blocks[-1] = root_hash
#     print("len(blocks):",len(blocks))
#     print("num_blocks:",num_blocks)
#     # 逆向迭代计算每个块的哈希值
#     for i in range(num_blocks-1, -1, -1):
#         if i*2+1 < len(blocks):
#             current_block_hash = hashlib.sha256((blocks[i*2] + blocks[i*2+1]).encode('utf-8')).hexdigest()
#             blocks[i] = current_block_hash
#
#     # 如果至少有两个块，则计算并返回根哈希值
#     if len(blocks) >= 2:
#         return ''.join(blocks[:-1])
#     else:
#         return blocks[0]






# def decompress_data(root_hash, num_blocks):
#     # 构建由 None 组成的列表，其长度为块数
#     blocks = [None] * num_blocks
#
#     # 将 Merkle 树的根哈希值放入最后一个位置
#     blocks[-1] = root_hash
#
#     # 逆向迭代计算每个块的哈希值
#     for i in range(num_blocks-1, 0, -1):
#         left_hash = blocks[i]
#         right_hash = blocks[i+1] if i % 2 == 0 else blocks[i-1]
#         block_data = f"{left_hash}{right_hash}".encode('utf-8')
#         block_hash = hashlib.sha256(block_data).hexdigest()
#         blocks[i//2] = block_hash
#     # 将所有块（除了最后一个）拼接起来并返回结果
#     return ''.join([block for block in blocks[:-1]])



if __name__ == '__main__':
    data = 'Hello world' * 100

    # 压缩数据并输出压缩率
    compressed_root_hash, num_blocks = compress_data(data)

    compressed_size = (num_blocks + 1) * 32
    original_size = len(data.encode('utf-8'))
    compression_ratio = compressed_size / original_size
    print(f'Compression ratio: {compression_ratio:.2f}')

    # 解压缩数据并比较原始数据
    # decompressed_data = decompress_data(compressed_root_hash, num_blocks)
    # print("1:",decompressed_data)
    # print("2:",data)
    # assert decompressed_data == data
