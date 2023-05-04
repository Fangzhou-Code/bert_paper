#用来传递参数给区块链
import json
import BS_Inference_of_Chinese_Sentence_Relationships
import passKeyInformation
from web3 import Web3


# Connect to Remix via its HTTP provider
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Set the default account to use
web3.eth.default_account = "0xE0c9fBa99dC9480d92B7cd6Bec2d3c8385EC483c"

# Get the current account's balance
account_balance = web3.eth.get_balance(web3.eth.default_account)


# set configuration
abi =[
	{
		"inputs": [],
		"name": "myData",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "_data",
				"type": "uint256"
			}
		],
		"name": "receiveData",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]
contract_address =  "0x04917BA769bb2B7d6880E9A7C23310B488c28c91"
my_contract = web3.eth.contract(address=contract_address, abi=abi)

# 获取参数发往solidty
pass_parms = passKeyInformation.passParms();
# 获取 Parameter 对象对应的张量
pass_parms_data = pass_parms.data

# 将张量转换为 NumPy 数组
pass_parms_array = pass_parms_data.numpy()

# 遍历数组并读取每个元素
for i in range(pass_parms_array.shape[0]):
    for j in range(pass_parms_array.shape[1]):
        # 使用 Python 的哈希函数计算散列值
        hash_value = hash(pass_parms_array[i, j])
		#tx_hash = my_contract.functions.receiveData(hash_value).transact()
        #print("Element at position ({}, {}): {}".format(i, j, hash_value))

# send data to smart contract
for i in  range(5):
	tx_hash = my_contract.functions.receiveData(i).transact()


# web3.eth.waitForTransactionReceipt(tx_hash)
receipt = web3.eth.wait_for_transaction_receipt(tx_hash)


# print
print("=====passKeyInfoToSolidity=====")
print(f"Current account balance: {account_balance}")
print('Data set to', my_contract.functions.myData().call())
print(receipt)
print("=====end=====")