import requests
from tqdm import tqdm
from .evaluator_base import evaluator
from .evaluator_utils import mode_dict, read_jsonl, to_jsonl, sum_checker_prompt, halu_sum_prompt

class sum_evaluator(evaluator):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)

    def evaluator_prompt(self, doc, ref_sum, halu_sum, halu_type):
        
        #query = sum_checker_prompt.format(doc, ref_sum, halu_sum)
        query = sum_checker_prompt.format(doc, ref_sum)
        self.payload["messages"][0]["content"] = query
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()

            #print(f"API Response:\n{data}")

            response_string = data["choices"][0]["message"]["content"]
            results = self.parse_response(response_string)
            #return response_string
            if results == "":
                return ""
            if self.parse_response(response_string) != "error":
                return "[参考摘要]" + results
            else:
                return "error"
        except:
            print("Error in get LLM API!")
            return "error"
        
    def halu_prompt(self, doc, ref_sum, halu_sum, halu_type):
        
        #query = sum_checker_prompt.format(doc, ref_sum, halu_sum)
        query = halu_sum_prompt.format(doc, halu_sum)
        self.payload["messages"][0]["content"] = query
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()

            #print(f"API Response:\n{data}")

            response_string = data["choices"][0]["message"]["content"]

            results = self.parse_response(response_string)
            #return response_string
            if results == "":
                return ""
            if self.parse_response(response_string) != "error":
                return "[幻觉摘要]" + results
            else:
                return "error"
        except:
            print("Error in get LLM API!")
            return "error"
    
    def parse_response(self, response_text):
        if response_text.startswith("符合"):
            return ""
        elif response_text.startswith("不符合"):
            return response_text
        else:
            print("Error Format!")
            return "error"

    def check_questions(self, data, output_path=None):
        checked_results = []
        total_count = 0
        correct_account = 0

        if data == None:
            data = self.data

        for data_item in tqdm(data):
            doc = data_item["doc"]
            ref_sum = data_item["ref_sum"]
            halu_sum = data_item["halu_sum"]
            label = data_item["halu_mode"]
            total_count+=1

            check_results = self.evaluator_prompt(doc, ref_sum, halu_sum, label)
            halu_results = self.halu_prompt(doc, ref_sum, halu_sum, label)
            results = check_results + halu_results

            if results != "error" and results != "":
                checked_results.append(results)
            else:
                correct_account+=1

        accuracy=correct_account/total_count if total_count>0 else 0
        print(f"检测器准确度：{accuracy*100:.2f}%")

        if output_path != None:
            to_jsonl(checked_results, output_path)
        
        return checked_results, accuracy

    def evaluate(self, data=None, output_path=None):
        checked_results = []
        total_count = 0
        correct_account = 0

        if data == None:
            data = self.data

        for data_item in tqdm(data):
            doc = data_item["doc"]
            ref_sum = data_item["ref_sum"]
            halu_sum = data_item["halu_sum"]
            label = data_item["halu_mode"]
            total_count+=1

            check_results = self.evaluator_prompt(doc, ref_sum, halu_sum, label)
            halu_results = self.halu_prompt(doc, ref_sum, halu_sum, label)
            results = check_results + halu_results

            if results != "error" and results != "":
                result_dict = {
                    "doc": doc,
                    "ref_sum": ref_sum,
                    "halu_sum": halu_sum,
                    "halu_type": label,
                    "reason": results
                }
                checked_results.append(results)
            else:
                correct_account+=1

        accuracy=correct_account/total_count if total_count>0 else 0
        print(f"检测器准确度：{accuracy*100:.2f}%")

        if output_path != None:
            to_jsonl(checked_results, output_path)
        
        return checked_results, accuracy
    
    def check_data(self, data):
        qualified_data = []
        total_count = 0
        if data == None:
            data = self.data

        for data_item in tqdm(data):
            doc = data_item["doc"]
            ref_sum = data_item["ref_sum"]
            halu_sum = data_item["halu_sum"]
            label = data_item["halu_mode"]
            total_count+=1

            check_results = self.evaluator_prompt(doc, ref_sum, halu_sum, label)
            halu_results = self.halu_prompt(doc, ref_sum, halu_sum, label)
            results = check_results + halu_results
            
            if results != "error" and results != "":
                continue
            else:
                qualified_data.append(data_item)

        return qualified_data