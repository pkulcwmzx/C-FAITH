import requests
from .corrector_utils import read_jsonl, to_jsonl
from tqdm import tqdm
from .corrector_base import corrector, halu_modes
from generator.generation_template import halu_mode_requirements

class qa_corrector(corrector):
    def __init__(self, halu_mode, api_key, model='Qwen'):
        super().__init__(halu_mode, api_key, model)
        self.basic_principle_prompt = ("1. 生成的答案必须是生成问题的正确回复。\n"
                                "2. 生成的问题必须尽可能难。\n" 
                                "3. 生成的问题必须与背景知识相关。\n"
                                "4. 生成的答案符合事实和逻辑。\n"
                                "5. 生成的依据必须是背景知识中的原句，且能够证明答案的正确性。\n")
        self.halu_answer_principle = ("1. 生成的幻觉回复必须与问题相关，是一个问题的回复。\n"
            "2. 生成的幻觉回复必须与依据不一致。\n"
            "3. 生成的幻觉回复必须与正确答案不一致。\n"
            "4. 生成的幻觉回复只需要包含回复本身，不能包含解释。\n")

        self.halu_answer_prompt = "假设你是一个幻觉回复生成者，请根据输入的问题、正确答案和背景知识生成问题的一个幻觉回复。具体要求如下：\n" + self.halu_answer_principle
        self.halu_type_def = ("虚构事实：幻觉回复犯了事实编造的错误。\n"
        "属性错误：问题以请介绍开头，幻觉回复对相关信息的介绍有误。\n"
        "实体错误：正确答案是一个实体，并且幻觉回复是错误的实体。\n"
        "时空幻觉：问题是对时间或空间提问，正确答案和幻觉回复都是时间或空间概念。\n"
        "关系错误：问题必须要求对不同实体进行比较，而不是对实体分别提问，幻觉回复中也必须包含错误的实体关系。\n"
        "虚假引用：幻觉回复引用名人名言、书籍或是网站等来源的知识佐证回复，但这引用的知识是错误的，与事实不一致。\n"
        "过时信息：问题具有时效性，幻觉回复使用了过去的知识进行回复，与现实相悖。\n")
        self.halu_label_prompt = "假设你是一个幻觉分类器，请根据输入的问题、正确答案和幻觉回复生成幻觉回复的类型。幻觉类型分为虚构事实、属性错误、实体错误、时空幻觉、关系错误。具体的幻觉类型的定义和要求如下：\n" + self.halu_type_def

    def update_principles(self, individual):
        self.basic_principle_prompt ="\n".join(individual[:5])
        self.halu_answer_prompt = "假设你是一个幻觉回复生成者，请根据输入的问题、正确答案和背景知识生成问题的一个幻觉回复。具体要求如下：\n" + "\n".join(individual[6:10])
        self.halu_label_prompt = "假设你是一个幻觉分类器，请根据输入的问题、正确答案和幻觉回复生成幻觉回复的类型。幻觉类型分为虚构事实、属性错误、实体错误、时空幻觉、关系错误。具体的幻觉类型的定义和要求如下：\n" + "\n".join(individual[10:])

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
        
        knowledge = evaluator_feedback.get('knowledge', "")
        question = evaluator_feedback.get('question', "")
        correct_answer = evaluator_feedback.get('correct_answer', "")
        halu_answer = evaluator_feedback.get('halu_answer', "")
        halu_type = evaluator_feedback.get('halu_type', "")
        reason = evaluator_feedback.get('reason', "")
        
        adjusted_prompts={
            "knowledge":knowledge,
            "question":question,
            "correct_answer":correct_answer,
            "halu_answer":halu_answer,
            "halu_type":halu_type,
            "qa_generate_prompt": None,
            "halu_answer_prompt": None,
            "halu_label_prompt": None,
        }

        correct_answer_reason = ""
        halu_answer_reason = ""
        halu_label_reason = ""

        if evaluator_feedback["correct_detection"]:
            if evaluator_feedback["correct_detection"].startswith("不符合"):
                correct_answer_reason = "[正确回答]" + evaluator_feedback["correct_detection"]

        if evaluator_feedback["halu_detection"]:
            if evaluator_feedback["halu_detection"].startswith("不符合"):
                halu_answer_reason = "[幻觉回答]" + evaluator_feedback["halu_detection"]

        if evaluator_feedback["mode_detection"]:
            if evaluator_feedback["mode_detection"].startswith("不符合"):
                halu_label_reason = "[幻觉类型]" + evaluator_feedback["mode_detection"]

        if correct_answer_reason:
            adjusted_prompts['qa_generate_prompt'] = self._adjust_qa_generate_prompt(
                knowledge, question, correct_answer, correct_answer_reason,self.halu_mode
            )
                 
        if halu_answer_reason:
            adjusted_prompts['halu_answer_prompt'] = self._adjust_halu_answer_prompt(
                knowledge, question, correct_answer, halu_answer, halu_answer_reason
            )
    
        if halu_label_reason:
            adjusted_prompts['halu_label_prompt'] = self._adjust_halu_label_prompt(
                question, halu_answer, correct_answer, halu_type, halu_label_reason
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
    
    def _adjust_qa_generate_prompt(self, knowledge, question, correct_answer, feedback,halu_mode):
        generator_prompt = "假设你是一个中文问题生成者。请根据给定的背景知识生成有挑战性和迷惑性的问题和对应的答案，并从背景知识中抽取出依据。生成的问题要求符合实际情况，并且是对背景知识中的相关信息提问。具体要求如下：\n"
        basic_principle_prompt = self.basic_principle_prompt
        halu_mode_prompt = halu_mode_requirements[halu_mode]
    
        original_prompt=generator_prompt + basic_principle_prompt + halu_mode_prompt
        correction_instruction =(
        "原始prompt：{}\n"
        "当前问答对如下：\n"
        "问题：{}\n"
        "正确回答：{}\n"
        "评估反馈：{}\n"

        "请根据“评估反馈”的内容仅修改或者增加原始Prompt里的具体要求，生成一个新的prompt。新prompt能够帮助生成更加准确的问答对，让答案具有更高的准确率。新的prompt的格式与原始prompt相同，不包含任何问答对，且对数据要求数量相同。给出新prompt，请以“假设你是一个中文问题生成者”开头\n\n"

        
        ).format(original_prompt,question,correct_answer,feedback)
        
        self.payload["messages"][0]["content"] = correction_instruction
        response = self._make_request()
        return response

    def _adjust_halu_answer_prompt(self, knowledge, question, correct_answer, halu_answer, feedback):
        original_prompt = self.halu_answer_prompt
        
        correction_instruction = (
        "原始prompt：{}\n"
        "当前问题和幻觉回答如下：\n"
        "问题：{}\n"
        "幻觉回答：{}\n"
        "评估反馈：{}\n"

        "请根据“评估反馈”的内容仅修改或者增加原始prompt里的具体要求，生成一个新的prompt。新prompt能够帮助生成更好的幻觉回复，既保证幻觉回复中存在幻觉或错误，也让幻觉回复更具有迷惑性。新的prompt的格式与原始Prompt相同，不包含任何问答对。给出新prompt，请以“假设你是一个幻觉回复生成者”开头\n\n"
        ).format(original_prompt,question,halu_answer,feedback)

        self.payload["messages"][0]["content"] = correction_instruction
        response = self._make_request()
        return response

    def _adjust_halu_label_prompt(self, question, halu_answer, correct_answer, halu_type, feedback):

        original_prompt = self.halu_label_prompt
        correction_instruction =(
        "原始prompt：{}\n"
        "当前问题、幻觉回答和幻觉类型如下：\n"
        "问题：{}\n"
        "幻觉回答：{}\n"
        "幻觉类型：{}\n"
        "评估反馈：{}\n"

        "请根据“评估反馈”的内容仅修改或者增加原始prompt里的具体要求，生成一个新的prompt。新prompt修改不同幻觉类型的幻觉定义，让幻觉定义更加准确可靠，更易于进行分类。新的prompt的格式与原始prompt相同，不包含任何问答对。给出新prompt，请以“假设你是一个幻觉分类器”开头\n\n"
        ).format(original_prompt,question,halu_answer,halu_type,feedback)

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
        
    def correct(self, data_path=None,optimization_round=0):
        feedback_list=self.data
        results=[]
        for feedback in feedback_list:
            adjusted_prompts=self.correct_(feedback)
            results.append(adjusted_prompts)

        data_path=f"data/intermediate_data_2/correct_round_{optimization_round}.jsonl"

        if data_path != None:
            to_jsonl(results,data_path)
        
        return results
    
if __name__ == '__main__':
    input_path="data/evaluator_output_test_new.jsonl"
    output_path="data/adjusted_prompts_test_new.jsonl"
    halu_mode=halu_modes[2]
    corrector = corrector("Qwen", input_path,halu_mode)
    adjusted_prompts_list = corrector.process_feedback(output_path)