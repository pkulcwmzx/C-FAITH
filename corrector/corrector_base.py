from .corrector_utils import read_jsonl, to_jsonl
import requests

model_dict = {"Qwen": "Vendor-A/Qwen/Qwen2.5-72B-Instruct", "Llama3": "meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen-7B": "Qwen/Qwen2.5-72B-Instruct", "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct", "Qwen-72B":"Qwen/Qwen2.5-72B-Instruct"}
url = "https://api.siliconflow.cn/v1/chat/completions"

halu_modes = ["虚构事实", "属性错误", "实体错误", "关系错误", "时空幻觉", "虚假引用", "过时信息"]

class corrector(object):
    def __init__(self, halu_mode, api_key, model='Qwen'):
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
        self.url = url
        self.halu_mode = halu_mode

    def update_data(self, data_path=None, data=None):
        if data != None:
            self.data = data
        else:
            self.data = read_jsonl(data_path)

    def correct(self, *args, **kwargs):
        # correct current prompt according to the reason
        raise NotImplementedError("Correction not yet implemented!") 