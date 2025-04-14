from .generator_base import generator
from .generation_template import mode_qg_template, halu_mode_requirements, few_shot_dict
from .generation_utils import read_jsonl, to_jsonl
from tqdm import tqdm
import requests
halu_modes = ["虚构事实", "属性错误", "实体错误", "关系错误", "时空幻觉", "虚假引用", "过时信息"]

class qa_generator(generator):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.qa_principles = ("1. 生成的答案必须是生成问题的正确回复。\n"
                                    "2. 生成的问题必须尽可能难。\n" 
                                    "3. 生成的问题必须与背景知识相关。\n"
                                    "4. 生成的答案符合事实和逻辑。\n"
                                    "5. 生成的依据必须是背景知识中的原句，且能够证明答案的正确性。要求生成依据是完整的表述，完全覆盖问答，不能只截取句子片段。\n")

        self.halu_answer_principles = ("1. 生成的幻觉回复必须与问题相关，是一个问题的回复。\n"
                "2. 生成的幻觉回复必须与依据不一致，且尽量具有迷惑性。\n"
                "3. 生成的幻觉回复必须与正确答案不一致。\n"
                "4. 生成的幻觉回复只需要包含回复本身，不能包含解释。\n"
                )

        self.halu_label_def = ("虚构事实：幻觉回复犯了事实编造的错误。\n"
            "属性错误：问题必须以请介绍开头，幻觉回复对相关信息的介绍有误。\n"
            "实体错误：正确答案是一个实体，并且幻觉回复是错误的实体。\n"
            "时空幻觉：问题是对时间或空间提问，正确答案和幻觉回复都是时间或空间概念。\n"
            "关系错误：问题必须要求对不同实体进行比较，而不是对实体分别提问，幻觉回复中也必须包含错误的实体关系。\n"
            "虚假引用：幻觉回复引用名人名言、书籍或是网站等来源的知识佐证回复，但这引用的知识是错误的，与事实不一致。\n"
            "过时信息：问题具有时效性，幻觉回复使用了过去的知识进行回复，与现实相悖。\n"
            )

    def update_principles(self, individual):
        self.qa_principles ="\n".join(individual[:5])
        self.halu_answer_principles = "\n".join(individual[6:10])
        self.halu_label_def = "\n".join(individual[10:])

    def set_templates(self,templates=None):
        if templates is None:
            templates = {}
            return templates

        qa_generate_prompt = templates.get("qa_generate_prompt", "")
        if qa_generate_prompt is not None:
            templates["qa_generate"] = qa_generate_prompt.rstrip()
        else:
            templates["qa_generate"] = ""  
        
        halu_answer_prompt = templates.get("halu_answer_prompt", "")
        if halu_answer_prompt is not None:
            templates["halu_answer"] = halu_answer_prompt.rstrip()
        else:
            templates["halu_answer"] = ""  
        
        halu_label_prompt = templates.get("halu_label_prompt", "")
        if halu_label_prompt is not None:
            templates["halu_label"] = halu_label_prompt.rstrip()
        else:
            templates["halu_label"] = "" 

        return templates

    def qa_generate_prompt(self, knowledge, halu_mode, templates=None, individual=None):
        generator_prompt = "假设你是一个中文问题生成者。请根据给定的背景知识生成有挑战性和迷惑性的问题和对应的答案，并从背景知识中抽取出依据。生成的问题要求符合实际情况，并且是对背景知识中的相关信息提问。具体要求如下：\n"
        
        if individual:
            basic_principle_prompt="\n".join(individual[:5])
            halu_mode_prompt=individual[5]
        else:
            basic_principle_prompt = self.qa_principles
            halu_mode_prompt = halu_mode_requirements[halu_mode]

        few_shots = few_shot_dict[halu_mode]

        if "qa_generate" in templates.keys():
            qa_generate_prompt=templates["qa_generate"] + few_shots
            qa_generate_prompt = qa_generate_prompt.format(knowledge["knowledge"])
        else:
            qa_generate_prompt = generator_prompt + basic_principle_prompt + halu_mode_prompt + few_shots
            qa_generate_prompt = qa_generate_prompt.format(knowledge["knowledge"])

        return qa_generate_prompt
    
    def halu_answer_prompt(self, knowledge, question, answer, templates=None, individual=None):
        halu_answer_inst = "假设你是一个幻觉回复生成者，请根据输入的问题、正确答案和背景知识生成问题的一个幻觉回复。具体要求如下：\n"
        
        if individual:
            halu_answer_principles="\n".join(individual[6:10])
        
        else:
            halu_answer_principles = self.halu_answer_principles

        halu_answer_few_shots = ("给定如下例子：\n"
            "[问题]：中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4，是真的吗？\n" 
            "[答案]：不是真的，中提琴音域比小提琴低完全5度。\n"
            "[依据]：中提琴是一种弓弦乐器, 其音域比小提琴低完全五度。其空弦从粗到细依序是c3-g3-d4-a4。\n"
            "[幻觉回复]：是真的，中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4。\n\n"
            "[问题]：李白最长的一首诗是什么。\n"
            "[答案]：李白最长的一首诗是《经乱离后天恩流夜郎忆旧游书怀赠江夏韦太守良宰》。\n"
            "[依据]：《经乱离后天恩流夜郎忆旧游书怀赠江夏韦太守良宰》是唐代诗人李白创作的自传体长诗，是李白集中最长的一首诗。\n"
            "[幻觉回复]：李白最长的一首诗是《长相思》。\n\n"
            "请为以下问题生成幻觉回复：\n"
            "[问题]：{}\n"
            "[答案]：{}\n"
            "[依据]：{}\n"
            "[幻觉回复]：")
        
        if "halu_answer" in templates.keys():
            halu_prompt = templates["halu_answer"] + halu_answer_few_shots.format(question, answer, knowledge)
        else:
            halu_prompt_template = halu_answer_inst + halu_answer_principles + halu_answer_few_shots
            halu_prompt = halu_prompt_template.format(question, answer, knowledge)

        return halu_prompt
    
    # just for multiple halu answer generation
    def multi_halu_answer_prompt(self, knowledge, question, answer, templates=None, individual=None):
        halu_answer_inst = "假设你是一个幻觉回复生成者，请根据输入的问题、正确答案和背景知识生成问题的幻觉回复。具体要求如下：\n"
        if individual:
            halu_answer_principles="\n".join(individual[6:10])
        
        else:
            halu_answer_principles = self.halu_answer_principles

        halu_answer_few_shots = ("给定如下例子：\n"
            "[问题]：中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4，是真的吗？\n" 
            "[答案]：不是真的，中提琴音域比小提琴低完全5度。\n"
            "[依据]：中提琴是一种弓弦乐器, 其音域比小提琴低完全五度。其空弦从粗到细依序是c3-g3-d4-a4。\n"
            "[幻觉回复]：是真的，中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4。\n\n"
            "[问题]：李白最长的一首诗是什么。\n"
            "[答案]：李白最长的一首诗是《经乱离后天恩流夜郎忆旧游书怀赠江夏韦太守良宰》。\n"
            "[依据]：《经乱离后天恩流夜郎忆旧游书怀赠江夏韦太守良宰》是唐代诗人李白创作的自传体长诗，是李白集中最长的一首诗。\n"
            "[幻觉回复]：李白最长的一首诗是《长相思》。\n\n"
            "生成的幻觉回复请按照以下格式输出，要求不同的幻觉回复之间换行但不要空行，其它地方不要换行：\n"
            "[幻觉回复1]：...\n[幻觉回复2]：...\n[幻觉回复3]：..."
            "请为以下问题生成3个不同的幻觉回复：\n"
            "[问题]：{}\n"
            "[答案]：{}\n"
            "[依据]：{}\n"
            )
        halu_prompt_template = halu_answer_inst + halu_answer_principles + halu_answer_few_shots
        halu_prompt = halu_prompt_template.format(question, answer, knowledge)

        return halu_prompt

    def parse_multi_halu(self, output, knowledge, question, answer, ent):
        def format_check(str_):
            pos1 = str_.find(']')
            pos2 = str_.find('：')
            if pos1 + 1 == pos2:
                return True
            else:
                return False
            
        halu_answers = output.split('\n')
        results = []
        for item in halu_answers:
            if not format_check(item):
                return "error"
            
        for item in halu_answers:
            halu_dict = {}
            if format_check(item):
                halu_answer = '：'.join(item.split('：')[1:])
            halu_dict["question"] = question
            halu_dict["correct_answer"] = answer
            halu_dict["reason"] = knowledge
            halu_dict["halu_answer"] = halu_answer
            halu_dict["ent"] = ent
            results.append(halu_dict)

        return results

    def generate_multi_halu(self, data_item):
        knowledge, question, answer, ent = data_item["reason"], data_item["question"], data_item["correct_answer"], data_item["ent"]
        halu_prompt = self.multi_halu_answer_prompt(knowledge, question, answer)
        self.payload["messages"][0]["content"] = halu_prompt
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()
            response_string = data["choices"][0]["message"]["content"]
            halu_data = self.parse_multi_halu(response_string, knowledge, question, answer, ent)
            if response_string != "error":
                templates = self.set_templates()
                labeled_data = self.halu_label_generate(halu_data, templates=templates)
                if len(labeled_data) < 3:
                    return "error"
                    
                return labeled_data

        except:
            print("Error in get LLM API!")
            return "error"

    def halu_label_prompt(self, question, halu_answer, answer, templates=None, individual=None):
        halu_label_inst = "假设你是一个幻觉分类器，请根据输入的问题、正确答案和幻觉回复生成幻觉回复的类型。幻觉类型分为虚构事实、属性错误、实体错误、时空幻觉、关系错误、虚假引用、过时信息共7类。具体的幻觉类型的定义和要求如下：\n"
        
        if individual:
            halu_label_def="\n".join(individual[10:])

        else:
            halu_label_def = self.halu_label_def

        halu_label_few_shots = ("给定如下例子：\n"
            "[问题]：中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4，是真的吗？\n"
            "[幻觉回复]：是真的，中提琴音域比小提琴高完全5度，其空弦从粗到细依序是c3-g3-d4-a4。\n"
            "[答案]：不是真的，中提琴音域比小提琴低完全5度。\n"
            "[幻觉标签]：虚构事实\n\n"
            "[问题]：李白最长的一首诗是什么。\n"
            "[幻觉回复]：李白最长的一首诗是《长相思》。\n"
            "[答案]：李白最长的一首诗是《经乱离后天恩流夜郎忆旧游书怀赠江夏韦太守良宰》。\n"
            "[幻觉标签]：实体错误\n\n"
            "请为以下问题和幻觉回复生成幻觉标签，要求幻觉标签必须是虚构事实、属性错误、实体错误、时空幻觉、关系错误、虚假引用、过时信息其中之一，请直接输出分类结果：\n"
            "[问题]：{}\n"
            "[幻觉回复]：{}\n"
            "[答案]：{}\n"
            "[幻觉标签]："
        )

        if "halu_label" in templates.keys():
            halu_label_prompt = templates["halu_label"]+halu_label_few_shots.format(question, halu_answer, answer)
        else:
            halu_label_template = halu_label_inst + halu_label_def + halu_label_few_shots
            halu_label_prompt = halu_label_template.format(question, halu_answer, answer)

        return halu_label_prompt
    
    def parse_qa_data(self, output, halu_label, ent):
        def format_check(str_):
            pos1 = str_.find(']')
            pos2 = str_.find('：')
            if pos1 + 1 == pos2:
                return True
            else:
                return False
        qa_pairs = output.split('\n\n')
        results = []
        for item in qa_pairs:
            qa_dict = {}
            tmp = item.split('\n')
            
            if len(tmp) != 3:
                print("Format Error!")
                continue
            question, answer, reason = tmp[0], tmp[1], tmp[2]
            
            if format_check(question) and format_check(answer):
                question = '：'.join(question.split('：')[1:])
                answer = '：'.join(answer.split('：')[1:])
                reason = '：'.join(reason.split('：')[1:])
                qa_dict["question"] = question
                qa_dict["correct_answer"] = answer
                qa_dict["reason"] = reason
                qa_dict["halu_type"] = halu_label
                qa_dict["ent"] = ent
                results.append(qa_dict)
            else:
                print("Format Error!")
                continue
        return results
    
    def parse_halu_answer(self, output, qa_dict):
        output = output.rstrip()
        qa_dict["halu_answer"] = output
        return qa_dict
    def parse_halu_label(self, output, qa_pair):
        halu_label = output[:4]
        if halu_label in halu_modes:
            qa_pair["halu_type"] = halu_label
        else:
            print("Halu label error!")
            print("The error halu label is:\n", halu_label)
            return "error"
        return qa_pair
    
    def response_generate(self, knowledge, halu_mode, templates=None, individual=None):
        # message for further domain classification
        ent = knowledge['ent']
        # qa pairs generaton
        qa_pairs = []
        qa_generator_prompt = self.qa_generate_prompt(knowledge, halu_mode, templates, individual)
        self.payload["messages"][0]["content"] = qa_generator_prompt
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            if response.status_code !=200:
                print("Error in API request, status code:", response.status_code)
                #print("Response:", response.text)
                return []
            data = response.json()
            response_string = data["choices"][0]["message"]["content"]
            response_string = self.parse_qa_data(response_string, halu_mode, ent)
            # check answer here
            qa_pairs.extend(response_string)
        except Exception as e:
            print("Error in get LLM API: ",e)

        return qa_pairs
    
    def halu_response_generate(self, qa_pairs, templates=None, individual=None):
        qa_data = []
        for qa_pair in qa_pairs:
            knowledge, question, answer = qa_pair["reason"], qa_pair["question"], qa_pair["correct_answer"]
            halu_answer_prompt = self.halu_answer_prompt(knowledge, question, answer, templates, individual)
            self.payload["messages"][0]["content"] = halu_answer_prompt
            try:
                response = requests.post(self.url, json=self.payload, headers=self.headers)
                data = response.json()
                response_string = data["choices"][0]["message"]["content"]
                response_string = self.parse_halu_answer(response_string, qa_pair)
                qa_data.append(response_string)
            except:
                print("Error in get LLM API!")
        return qa_data
    
    def halu_label_generate(self, qa_data, templates=None, individual=None):
        final_qa_data = []
        for qa_pair in qa_data:
            question, halu_answer, answer = qa_pair["question"], qa_pair["halu_answer"], qa_pair["correct_answer"]
            halu_label_prompt = self.halu_label_prompt(question, halu_answer, answer, templates, individual)
            self.payload["messages"][0]["content"] = halu_label_prompt
            try:
                response = requests.post(self.url, json=self.payload, headers=self.headers)
                data = response.json()
                response_string = data["choices"][0]["message"]["content"]
                response_string = self.parse_halu_label(response_string, qa_pair)
                if response_string != "error":
                    final_qa_data.append(response_string)
                    
            except:
                print("Error in get LLM API!")
        return final_qa_data
    
    def generate(self, start_idx, end_idx, halu_mode, outfile=None, templates=None, individual=None,optimization_round=0):
        data = self.data[start_idx: end_idx]
        generated_data = []
        templates = self.set_templates(templates=templates)

        if outfile == None:
            outfile=f"data/intermediate_data_2/generate_round_{optimization_round}.jsonl"

        for knowledge in tqdm(data):
            knowledge['knowledge'] = knowledge['knowledge'].rstrip()
            qa_pairs = self.response_generate(knowledge, halu_mode, templates, individual)
            halu_data = self.halu_response_generate(qa_pairs, templates, individual)
            final_data = self.halu_label_generate(halu_data, templates, individual)
            generated_data.extend(final_data)
        
        if outfile != None:
            to_jsonl(generated_data, outfile)
        
        return generated_data
    
    def generate_data(self, data, halu_mode, outfile=None, templates=None, individual=None,optimization_round=0):
        generated_data = []
        templates = self.set_templates(templates=templates)
        
        if outfile == None:
            outfile=f"data/intermediate_data_2/generate_dev_round_{optimization_round}.jsonl"

        for knowledge in tqdm(data):
            knowledge['knowledge'] = knowledge['knowledge'].rstrip()
            qa_pairs = self.response_generate(knowledge, halu_mode, templates, individual)
            halu_data = self.halu_response_generate(qa_pairs, templates, individual)
            final_data = self.halu_label_generate(halu_data, templates, individual)
            generated_data.extend(final_data)
        
        if outfile != None:
            to_jsonl(generated_data, outfile)

        return generated_data

if __name__ == '__main__':
    data_path="data/sample_data.jsonl"
    generator = qa_generator(model="Qwen")
    generator.read_data(data_path=data_path)
    _ = generator.generate(0, 2, halu_modes[-3], "data/test.jsonl")