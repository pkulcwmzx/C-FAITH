import requests
import re
from .evaluator_utils import read_jsonl, to_jsonl
from tqdm import tqdm
from .evaluator_base import evaluator
from .evaluator_utils import mode_dict, correct_template, halu_template

model_dict = {"Qwen": "Vendor-A/Qwen/Qwen2-72B-Instruct", "Llama3": "meta-llama/Meta-Llama-3.1-70B-Instruct"}
url = "https://api.siliconflow.cn/v1/chat/completions"

class qa_evaluator(evaluator):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)

    def evaluate_qa(self,data):
        qa_dict={
            "correct_detection":None,
            "reason":None
        }
        
        question = data["question"]
        correct_answer=data["correct_answer"]
        knowledge=data["reason"]
 
        qa_query=correct_template.format(knowledge, question, correct_answer)
        
        self.payload["messages"][0]["content"] = qa_query
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()

            response_string = data["choices"][0]["message"]["content"]
            result_dict,number=self.parse_response(response_string)
            qa_dict["correct_detection"]=result_dict["correct_detection"]
            reason=result_dict.get("reason", "")
            qa_dict["reason"]=reason
            #return response_string
            return qa_dict,number
        
        except:
            print("Error in get LLM API!")
            return None,None
        
    def evaluate_halu_answer(self,data,qa_dict):
        halu_dict={
            "correct_detection":qa_dict["correct_detection"],
            "halu_detection": None,
            "reason":None
        }
        reason=qa_dict["reason"]
        question = data["question"]
        correct_answer=data["correct_answer"]
        halu_answer=data["halu_answer"]
        knowledge=data["reason"]

        halu_query=halu_template.format(knowledge, question, halu_answer)
        

        self.payload["messages"][0]["content"] = halu_query
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()

            response_string = data["choices"][0]["message"]["content"]

            result_dict,number=self.parse_response(response_string)
            halu_dict["halu_detection"]=result_dict["halu_detection"]
            reason_temp=result_dict.get("reason", "")
            reason=reason+reason_temp
            halu_dict["reason"]=reason

            return halu_dict,number
        
        except:
            print("Error in get LLM API!")
            return None,None
        
    def evaluate_halu_type(self,data,halu_dict):
        type_dict={
            "correct_detection":halu_dict["correct_detection"],
            "halu_detection":halu_dict["halu_detection"],
            "mode_detection":None,
            "reason":None
        }
        reason=halu_dict["reason"]
        question = data["question"]
        halu_answer=data["halu_answer"]
        halu_type=data["halu_type"]
        correct_answer=data["correct_answer"]

        type_query=mode_dict[halu_type].format(question,correct_answer,halu_answer,halu_type)
        
        self.payload["messages"][0]["content"] = type_query
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()

            response_string = data["choices"][0]["message"]["content"]

            result_dict,number=self.parse_response(response_string)
            type_dict["mode_detection"]=result_dict["mode_detection"]
            reason_temp = result_dict.get("reason", "")
            reason = reason+reason_temp
            type_dict["reason"] = reason
            return type_dict,number
        except:
            print("Error in get LLM API!")
            return None,None
    
        
    def parse_response(self, response_text):
        reason_dict = {
            "correct_detection": None,
            "halu_detection": None,
            "mode_detection": None,
            "reason": ""
        }
        number=0

        if not response_text:
            print("Empty response text")
            return reason_dict
        
        current_reason = ""

        if re.match(r"\[正确检测\][：:]", response_text):
            reason_dict["correct_detection"] = re.sub(r"\[正确检测\][：:]", "", response_text).strip()
            if reason_dict["correct_detection"].startswith("不符合"):
                current_reason += "[正确回答]" + reason_dict["correct_detection"]
                number+=1

        elif re.match(r"\[幻觉检测\][：:]", response_text):
            reason_dict["halu_detection"] = re.sub(r"\[幻觉检测\][：:]", "", response_text).strip()
            if reason_dict["halu_detection"].startswith("不符合"):
                current_reason += "[幻觉回答]" + reason_dict["halu_detection"]
                number+=1

        elif re.match(r"\[类型检测\][：:]", response_text):
            reason_dict["mode_detection"] = re.sub(r"\[类型检测\][：:]", "", response_text).strip()
            if reason_dict["mode_detection"].startswith("不符合"):
                current_reason += "[幻觉类型]" + reason_dict["mode_detection"]
                number+=1

        reason_dict["reason"] = current_reason.strip()

        if not reason_dict["reason"]:
            reason_dict["reason"] = ""
            return reason_dict,number
        else:
            return reason_dict,number
    
    def evaluate(self,data=None,output_path=None,optimization_round=0):
        
        total_count = 0
        correct_account = 0
        answer_error=0 
        halu_error=0
        type_error=0

        checked_results = []
        total_results = []
        if data == None:
            data = self.data

        for data_item in tqdm(data,desc="evaluating"):
            
            qa_dict,answer_number=self.evaluate_qa(data_item)

            if qa_dict is not None:
                halu_dict,halu_number=self.evaluate_halu_answer(data_item,qa_dict)
            else:
                continue
            if halu_dict is not None:
                type_dict,type_number=self.evaluate_halu_type(data_item,halu_dict)
            else:
                continue
            total_count += 1
            
            type_dict["knowledge"]=data_item["reason"]
            type_dict["question"]=data_item["question"]
            type_dict["correct_answer"]=data_item["correct_answer"]
            type_dict["halu_answer"]=data_item["halu_answer"]
            type_dict["halu_type"]=data_item["halu_type"]
            
            answer_error+=answer_number
            halu_error+=halu_number
            type_error+=type_number
            
            total_results.append(type_dict)
            if type_dict["reason"]:
                checked_results.append(type_dict)
            else:
                correct_account+=1

        accuracy=correct_account/total_count if total_count>0 else 0
        accuracy_answer=(1-answer_error/total_count) if total_count>0 else 0
        accuracy_halu=(1-halu_error/total_count) if total_count>0 else 0
        accuracy_type=(1-type_error/total_count) if total_count>0 else 0

        print(f"检测器准确度：{accuracy*100:.2f}%\n")
        print(f"回答检测准确度：{accuracy_answer*100:.2f}%\n")
        print(f"幻觉检测准确度：{accuracy_halu*100:.2f}%\n")
        print(f"类型检测准确度：{accuracy_type*100:.2f}%\n")
        
        if output_path != None:
            to_jsonl(total_results, output_path)
        else:
            output_path=f"data/intermediate_data_2/check_round_{optimization_round}.jsonl"
            to_jsonl(checked_results, output_path)

        return checked_results,accuracy,accuracy_answer,accuracy_halu,accuracy_type
    
    def check_data(self, data):
        qualified_data = []
        
        if data == None:
            data = self.data

        for data_item in tqdm(data,desc="checking data"):
            
            qa_dict, _=self.evaluate_qa(data_item)
            halu_dict, _=self.evaluate_halu_answer(data_item,qa_dict)
            type_dict, _=self.evaluate_halu_type(data_item,halu_dict)
        
            if type_dict["reason"]:
                continue
            else:
                qualified_data.append(data_item)

        return qualified_data

if __name__ == "__main__":

    data_path = "data/evaluator_raw.jsonl"
    output_path = "data/evaluator_check.jsonl"
    evaluator=qa_evaluator("Qwen")
    evaluator.update_data(data_path)
    _=evaluator.evaluate(output_path)