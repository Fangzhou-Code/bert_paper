# bert_paper
This is code for my first paper about federated learning and blockchain.

In paper,i designed a new architecture for federated learning and blockchain.

structure:

    --materials: some unused file.
    --BS_Inference_of_Chinese_Sentence_Relationships.py
    --Differential_Privacy.py
    --GPUandCPU_allocation.py
    --MBS_Inference_of_Chinese_Sentence_Relationships.py    
    --MerkleTree_Based_Data_Compression.py
    --passKeyInfoToSolidty.py
    --README.md

# Architecture Flow.
1. BS_Inference_of_Chinese_Sentence_Relationships.py: GPUandCPU_allocation.py--->model--->set_params

2. MBS_Inference_of_Chinese_Sentence_Relationships.py: GPUandCPU_allocation.py--->get_params--->model

3. passKeyInfoToSolidty:BS_Inference_of_Chinese_Sentence_Relationships.py--> MBS_Inference_of_Chinese_Sentence_Relationships.py--> MerkleTree_Based_Data_Compression.py--> send data to smart contract--> web3.eth.waitForTransactionReceipt(tx_hash)



# Run
1. you just need to **run passKeyInfoToSolidty.py**

It will automatically run BS_Inference_of_Chinese_Sentence_Relationships

-->MBS_Inference_of_Chinese_Sentence_Relationships

-->MerkleTree_Based_Data_Compression: compress parms

-->pass compressed parms to solidty

2. run MBS_Inference_of_Chinese_Sentence_Relationships

It will automatically run BS_Inference_of_Chinese_Sentence_Relationships

-->MBS_Inference_of_Chinese_Sentence_Relationships


# Notice
you may think fantastic!
However, my code so sucks.

i'm sorry wasting your time to view in this repo.
