'''
用来给BS和MBS在区块链上注册并分配唯一的id，
调用remix函数RegisterForBS.sol
'''
import hashlib
import datetime
from web3 import Web3, HTTPProvider

def register(puf):

    # 连接到以太坊网络
    w3 = Web3(HTTPProvider('http://localhost:8545'))
    # Set the default account to use
    w3.eth.default_account = "0xE0c9fBa99dC9480d92B7cd6Bec2d3c8385EC483c"

    # Get the current account's balance
    account_balance = w3.eth.get_balance(w3.eth.default_account)
    print(f"Current account balance: {account_balance}")

    # 获取智能合约 ABI 和地址
    abi = [
	{
		"inputs": [],
		"name": "clearIdentifiers",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "inputStr",
				"type": "string"
			}
		],
		"name": "getIdentifier",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]# 智能合约的 ABI
    address = "0x8b2002800Cd2A0a657356D10FC16d18eE5925527" # 智能合约的地址

    # 创建智能合约对象
    contract = w3.eth.contract(address=address, abi=abi)

    # 调用 getUniqueId 函数并传递唯一识别符作为参数
    tx_hash = contract.functions.getIdentifier(puf).transact()

    # 等待事务被区块链网络确认
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # 调用只读的 getUniqueIDFor 函数来获取与唯一识别符相关联的唯一 ID
    unique_id = contract.functions.getIdentifier(puf).call()
    print("=====基站到区块链上进行注册=====")
    print(unique_id) #unique_id 为 int 类型
    if unique_id == 9999:
        tx_hash = contract.functions.clearIdentifiers().transact()
        print("达到最大次数，进行清空")
    print("=====end=====")

def puf_function(input_str):
    # 获取当前时间戳（秒数）
    timestamp = datetime.datetime.now().timestamp()
    # 将时间戳转换为整数
    timestamp_int = int(timestamp)
    # 组合自定义字符串和整数时间戳
    #unique_id_str = input_str + str(timestamp_int)
    unique_id_str = input_str
    print(unique_id_str)
    # 通过对输入进行哈希生成固定长度的输出值
    hashed_input = hashlib.sha256(unique_id_str.encode()).hexdigest()
    # 取哈希值的前16位作为ID
    unique_id = hashed_input[:16]
    return unique_id

if __name__ == '__main__':
    register(puf_function("xxxsaasas"))
    register(puf_function("hello"))
    register(puf_function("xxx"))
