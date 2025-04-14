import requests
from .corrector_utils import read_jsonl, to_jsonl
from tqdm import tqdm
from .corrector_base import corrector

class sum_corrector(corrector):
    def __init__(self, halu_mode, api_key, model='Qwen'):
        super().__init__(halu_mode, api_key, model)

        self.basic_principle_prompt = ("1. 生成的摘取文档必须来源于背景知识，能够独立构成文档，且逻辑自洽，至少包含200字。\n"
                                "2. 生成的参考摘要必须是摘取文档的摘要，不能与摘取文档中的信息相悖。\n" 
                                "3. 生成的参考摘要必须概括摘取文档的主要内容，且尽量简洁。\n"
                                )
        
        self.halu_sum_principles = ("1. 生成的幻觉摘要必须是文档的一个摘要。\n"
            "2. 生成的幻觉摘要必须与文档中给出的事实信息存在冲突。\n"
            "3. 生成的幻觉摘要必须与参考摘要存在不一致。\n"
            "4. 生成的幻觉摘要只需要包含摘要本身，不能包含解释。\n"
            "5. 生成的幻觉摘要需要尽量具有迷惑性，容易让人误以为是正确的。\n")

    def _parse_adajusted_prompts(self,adjusted_prompts_text):
        prompt_sections = adjusted_prompts_text.split("\n\n")
        
        if len(prompt_sections) < 3:
            raise ValueError("模型返回的修正prompt格式不正确。")
        
        qa_generate_prompt = prompt_sections[0].strip()
        halu_answer_prompt = prompt_sections[1].strip()
        halu_label_prompt = prompt_sections[2].strip()
        
        return {
            "qa_generate_prompt": qa_generate_prompt,
            "halu_answer_prompt": halu_answer_prompt,
            "halu_label_prompt": halu_label_prompt
        }
   
    def correct_(self,evaluator_feedback):
        
        doc = evaluator_feedback.get('doc', "")
        ref_sum = evaluator_feedback.get('ref_sum', "")
        halu_sum = evaluator_feedback.get('halu_sum', "")
        halu_type = evaluator_feedback.get('halu_type', "")
        reason = evaluator_feedback.get('reason', "")
        
        adjusted_prompts={
            "doc":doc,
            "ref_sum":ref_sum,
            "halu_sum":halu_sum,
            "halu_type":halu_type,
            "qa_generate_prompt": None,
            "halu_answer_prompt": None,
            "halu_label_prompt": None,
        }

        correct_answer_reason = self._extract_reason(reason, "参考摘要")
        halu_answer_reason = self._extract_reason(reason, "幻觉摘要")

        if correct_answer_reason:
            adjusted_prompts['sum_generate_prompt'] = self._adjust_sum_generate_prompt(
                doc, ref_sum, halu_sum, halu_type
            )
                 
        if halu_answer_reason:
            adjusted_prompts['halu_sum_prompt'] = self._adjust_halu_answer_prompt(
                doc, ref_sum, halu_sum, halu_type
            )
        
        return adjusted_prompts
    
    def _extract_reason(self,reason,keyword):
        if f"[{keyword}]" in reason:
            start_idx=reason.find(f"[{keyword}]")
            end_idx=reason.find("[",start_idx+1)
            if end_idx==-1:
                end_idx=len(reason)
            return reason[start_idx:end_idx].strip()
        return ""
    
    def _adjust_sum_generate_prompt(self, doc, ref_sum, halu_sum, halu_type):
        sum_generator_inst = "假设你是一个中文摘要数据生成者。请根据给定的背景知识，摘取其中部分作为摘取文档，并为摘取文档生成参考摘要。具体要求如下:\n"
        basic_principle_prompt = self.basic_principle_prompt
    
        original_prompt=sum_generator_inst + basic_principle_prompt
        correction_instruction =(
        "原始prompt:{}\n"
        "当前摘要数据如下：\n"
        "文档：{}\n"
        "参考摘要：{}\n"
        "请根据“评估反馈”的内容仅修改或者增加原始Prompt里的具体要求，生成一个新的prompt。新prompt能够帮助生成更加准确的问答对，让答案具有更高的准确率。新的prompt的格式与原始prompt相同，不包含任何问答对。给出新prompt，请以“假设你是一个中文问题生成者”开头\n\n"
        ).format(original_prompt, doc, ref_sum)
        
        print(correction_instruction)
        self.payload["messages"][0]["content"] = correction_instruction
        response = self._make_request()
        return response

    def _adjust_halu_answer_prompt(self, doc, ref_sum, halu_sum, halu_type):
        halu_sum_inst = "假设你是一个幻觉摘要生成者，请根据输入的文档、和参考摘要生成文档的一个幻觉摘要。具体要求如下:\n"
        halu_sum_principles = self.halu_sum_principles
        original_prompt = halu_sum_inst + halu_sum_principles

        correction_instruction = (
        "原始prompt:{}\n"
        "当前问题和幻觉回答如下:\n"
        "问题：{}\n"
        "幻觉回答：{}\n"
        "评估反馈：{}\n"
        "请根据“评估反馈”的内容仅修改或者增加原始prompt里的具体要求，生成一个新的prompt。新prompt能够帮助生成更好的幻觉回复，既保证幻觉回复中存在幻觉或错误，也让幻觉回复更具有迷惑性。新的prompt的格式与原始Prompt相同，不包含任何问答对。给出新prompt，请以“假设你是一个幻觉回复生成者”开头\n\n"
        ).format(original_prompt, doc, ref_sum, halu_sum)

        print(correction_instruction)
        self.payload["messages"][0]["content"] = correction_instruction
        response = self._make_request()
        return response

    def _make_request(self):

        try:
            response=requests.post(self.url,json=self.payload,headers=self.headers)
            if response.status_code!=200:
                print("Error in API request, status code:", response.status_code)
                print("Response:", response.text)
                return {}
            data=response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("Error in API request:",e)
            return {}
        
    def correct(self, data_path=None):
        feedback_list=self.data
        results=[]
        for feedback in feedback_list:
            adjusted_prompts=self.correct_(feedback)
            results.append(adjusted_prompts)

        if data_path != None:
            to_jsonl(results,data_path)
        
        return results