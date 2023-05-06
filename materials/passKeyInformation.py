#用于bs向6gbs发送数据


#print(pass_parms)
#print(type(pass_parms)) class 'torch.nn.parameter.Parameter'
#pass_parms = BS_Inference_of_Chinese_Sentence_Relationships.passParms();


class passClass:
    def __init__(self):
        self.pass_parms = None

    def set_parms(self,parms):
        print("=====passKeyInformation: set params=====")
        self.pass_parms = parms
        print(self.pass_parms)
        print("=====end=====")

    def get_parms(self):
        print("=====passKeyInformation: get params=====")
        print(self.pass_parms)
        print("=====end=====")
        return self.pass_parms
