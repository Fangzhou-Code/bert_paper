#用于bs向6gbs发送数据


#print(pass_parms)
#print(type(pass_parms)) class 'torch.nn.parameter.Parameter'
#pass_parms = BS_Inference_of_Chinese_Sentence_Relationships.passParms();
print("=====passKeyInformation=====")
print("=====end=====")
pass_parms = None
class passClass:
    def set_parms(parms):
        pass_parms = parms

    def get_parms(self):
        return pass_parms
