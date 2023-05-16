'''
用来传递参数给区块链
对比框架加密的数据传输，将不加密的参数进行传输
'''
import json
import MerkleTree_Based_Data_Compression
import MBS_Inference_of_Chinese_Sentence_Relationships
from torch import tensor
from web3 import Web3
import time
import hashlib

#test delay
start = time.time()

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
pass_parms = MBS_Inference_of_Chinese_Sentence_Relationships.get_parms()

# 获取 Parameter 对象对应的张量
pass_parms_data = pass_parms.data
print("pass_parms_data: ",pass_parms_data)

# 将张量转换为 NumPy 数组
pass_parms_array = pass_parms_data.numpy()
print("len(pass_parms_array): ",len(pass_parms_array))
print("pass_parms_array: ",pass_parms_array)

# 遍历数组并读取每个元素
hash = ""
for i in range(pass_parms_array.shape[0]):
    for j in range(pass_parms_array.shape[1]):
        # 使用 Python 的哈希函数计算散列值
        hash = hashlib.sha256(str(pass_parms_array[i,j]).encode('utf-8')).hexdigest()
        print("hash:",hash,"    type(hash):",type(hash))
        # send data to smart contract
        tx_hash = my_contract.functions.storeData(hash).transact()
        # web3.eth.waitForTransactionReceipt(tx_hash)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print('Data set to:', my_contract.functions.getData().call())
        print("receipt:", receipt)


# test delay
end = time.time()
delay = end - start


print("delay:",delay)
print("=====end=====")

