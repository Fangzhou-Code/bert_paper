'''
用来传递参数给区块链
'''
import json
import MerkleTree_Based_Data_Compression
import MBS_Inference_of_Chinese_Sentence_Relationships
from torch import tensor
from web3 import Web3


print("=====passKeyInfoToSolidity=====")
# Connect to Remix via its HTTP provider
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Set the default account to use
web3.eth.default_account = "0xE0c9fBa99dC9480d92B7cd6Bec2d3c8385EC483c"

# Get the current account's balance
account_balance = web3.eth.get_balance(web3.eth.default_account)
print(f"Current account balance: {account_balance}")

# set configuration
abi =[
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": True,
				"internalType": "string",
				"name": "newData",
				"type": "string"
			}
		],
		"name": "DataStored",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "newData",
				"type": "string"
			}
		],
		"name": "storeData",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "getData",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]
contract_address =  "0xbA1cc809f63CF8152714888ec9A0524bCFBC80b6"
my_contract = web3.eth.contract(address=contract_address, abi=abi)

# 获取参数发往solidty
# pass_parms = tensor([[ 0.0074,  0.0166, -0.0037,  0.0162,  0.0184,  0.0066],
#         [-0.0414, -0.0013, -0.0070,  -0.0107,  0.0395, -0.0037]],
#        requires_grad=True)
pass_parms = MBS_Inference_of_Chinese_Sentence_Relationships.get_parms()
# 获取 Parameter 对象对应的张量
pass_parms_data = pass_parms.data
print(pass_parms_data)

# 将张量转换为 NumPy 数组
pass_parms_array = pass_parms_data.numpy()
print(len(pass_parms_array))
print(pass_parms_array)

# 遍历数组并读取每个元素
data = ""
for i in range(pass_parms_array.shape[0]):
    for j in range(pass_parms_array.shape[1]):
        # 使用 Python 的哈希函数计算散列值
        # hash_value = hash(pass_parms_array[i, j])
		#tx_hash = my_contract.functions.receiveData(hash_value).transact()
        # print("Element at position ({}, {}): {}".format(i, j, hash_value))
                data = data+str(pass_parms_array[i,j])
print("data:",data)
# 压缩数据并输出压缩率
compressed_root_hash, num_blocks = MerkleTree_Based_Data_Compression.compress_data(data)
compressed_size = (num_blocks + 1) * 32
original_size = len(data.encode('utf-8'))
compression_ratio = compressed_size / original_size
print("compressed_root_hash",compressed_root_hash)
print("type(compressed_root_hash)",type(compressed_root_hash[0]))
print(f'Compression ratio: {compression_ratio:.2f}')

# send data to smart contract
tx_hash = my_contract.functions.storeData(compressed_root_hash[0]).transact()

# web3.eth.waitForTransactionReceipt(tx_hash)
receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

print('Data set to', my_contract.functions.getData().call())
print(receipt)
print("=====end=====")