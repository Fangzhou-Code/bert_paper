# bert_paper
This is code for my first paper about federated learning and blockchain.

In paper,i designed a new architecture for federated learning and blockchain.

structure:

    --materials: some unused file.
    --BS_Inference_of_Chinese_Sentence_Relationships.py
    --MBS_Inference_of_Chinese_Sentence_Relationships.py
    --MerkleTree_Based_Data_Compression.py
    --GPUandCPU_allocation.py
    --passKeyInfoToSolidty.py
    --Differential_Privacy.py
    --README.md
    --Test_SNRandLatency.py

algorithm:
      
      --GPUandCPU_allocation.py: resource allocation algorithm
      --MerkleTree_Based_Data_Compression.py： A Data Compression Algorithm Based on Merkle Tree
      --Differential_Privacy.py: Differential privacy


# Architecture Flow.
1. BS_Inference_of_Chinese_Sentence_Relationships.py: simulate the operation model of small base stations

flow: GPUandCPU_allocation.py--->model--->set_params

2. MBS_Inference_of_Chinese_Sentence_Relationships.py: simulate the operation model of 6G macro base stations

flow: GPUandCPU_allocation.py--->get_params--->model

3. passKeyInfoToSolidty:pass parameters to blockchain smart contracts

flow: BS_Inference_of_Chinese_Sentence_Relationships.py-->
MBS_Inference_of_Chinese_Sentence_Relationships.py-->
MerkleTree_Based_Data_Compression.py-->
send data to smart contract-->
web3.eth.waitForTransactionReceipt(tx_hash)


# Test
1. Test_SNRandLatency.py: test the delay of the framework under different signal-to-noise ratios
![avatar](/Users/fangzhou/Documents/my paper/results/snr.png)g


# Run
1. If you want to imitate the interaction process between the base station and the blockchain，
you just need to **run passKeyInfoToSolidty.py**

    It will automatically run BS_Inference_of_Chinese_Sentence_Relationships

    -->MBS_Inference_of_Chinese_Sentence_Relationships

    -->MerkleTree_Based_Data_Compression: compress parms

    -->pass compressed parms to solidty

2. If you want to imitate the interaction process between small base stations and 6G macro base stations
,**run MBS_Inference_of_Chinese_Sentence_Relationships**

    It will automatically run BS_Inference_of_Chinese_Sentence_Relationships

    -->MBS_Inference_of_Chinese_Sentence_Relationships

3. if you want to test the delay of the framework under different signal-to-noise ratios,
**run Test_SNRandDelay.py**

# Notice
you may think fantastic!
However, my code so sucks.

i'm sorry wasting your time to view in this repo.
