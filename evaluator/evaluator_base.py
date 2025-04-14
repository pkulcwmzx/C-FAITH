import requests
from .evaluator_utils import read_jsonl, to_jsonl
from tqdm import tqdm
from .evaluator_utils import mode_dict, correct_template, halu_template

model_dict = {"Qwen": "Vendor-A/Qwen/Qwen2.5-72B-Instruct", "Llama3": "meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen-7B": "Qwen/Qwen2.5-72B-Instruct", "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct", "Qwen-72B":"Qwen/Qwen2.5-72B-Instruct"}
url = "https://api.siliconflow.cn/v1/chat/completions"

class evaluator(object):
    def __init__(self, api_key, model='Qwen'):
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
            "max_tokens": 1024,
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

    def update_data(self, data_path=None, data=None):
        if data != None:
            self.data = data
        else:
            self.data = read_jsonl(data_path)

    def evaluate(self,  *args, **kwargs):
        # evaluate the generated data
        raise NotImplementedError("Evaluation not yet implemented!") 