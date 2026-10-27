from autotimes.models import AutoTimes_Llama
class Exp_Basic:
    def __init__(self,args):
        self.args=args
        self.model_dict={'AutoTimes_Llama':AutoTimes_Llama}
        self.model=self._build_model()

