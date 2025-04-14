import requests
from .generation_template import mode_qg_template, halu_mode_requirements, few_shot_dict
from .generation_utils import read_jsonl, to_jsonl
from tqdm import tqdm

model_dict = {"Qwen": "Vendor-A/Qwen/Qwen2.5-72B-Instruct", "Llama3": "meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen-7B": "Qwen/Qwen2.5-72B-Instruct", "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct", "Qwen-72B":"Qwen/Qwen2.5-72B-Instruct"}
url = "https://api.siliconflow.cn/v1/chat/completions"

halu_modes = ["虚构事实", "属性错误", "实体错误", "关系错误", "时空幻觉", "虚假引用", "过时信息"]

class generator(object):
    def __init__(self, api_key, model="Qwen"):
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.payload = {
            "model": model_dict[model],
            "messages": [
                {
                    "role": "user",
                    "content": ""
                }
            ],
            "stream": False,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1
        }
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer " + api_key
        }

    def read_data(self, data_path=None, data=None):
        if data != None:
            self.data = data
        else:
            self.data = read_jsonl(data_path)
    def response_generate(self, *args, **kwargs):
        # generate pairwise query and response
        raise NotImplementedError("Response generation not yet implemented")
    
    def halu_response_generate(self, *args, **kwargs):
        # generate halu response according to question
        raise NotImplementedError("Halu response generation not yet implemented") 
    
    def halu_label_generate(self, *args, **kwargs):
        # generate halu label according to question, answer, halu answer
        raise NotImplementedError("Halu label generation not yet implemented") 
    
    def generate(self, *args, **kwargs):
        # generate all test data
        raise NotImplementedError("Halu label generation not yet implemented") 