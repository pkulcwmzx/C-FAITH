from .generator_base import generator
from tqdm import tqdm
import requests
from .generation_utils import to_jsonl, read_jsonl

class sum_generator(generator):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.basic_principle_prompt = ("1. 生成的摘取文档必须来源于背景知识，能够独立构成文档，且逻辑自洽，至少包含200字。\n"
                                "2. 生成的参考摘要必须是摘取文档的摘要，不能与摘取文档中的信息相悖。\n" 
                                "3. 生成的参考摘要必须概括摘取文档的主要内容，且尽量简洁。\n")

        self.halu_sum_principles = ("1. 生成的幻觉摘要必须是文档的一个摘要，对文档进行概括。\n"
            "2. 生成的幻觉摘要必须与文档中给出的事实信息存在冲突。\n"
            "3. 生成的幻觉摘要必须与参考摘要存在不一致。\n"
            "4. 生成的幻觉摘要只需要包含摘要本身，不能包含解释。\n"
            "5. 生成的幻觉摘要需要尽量具有迷惑性，容易让人误以为是正确的。\n")

    def update_principles(self, individual):
        self.basic_principle_prompt = '\n'.join(individual[:3])
        self.halu_sum_principles = '\n'.join(individual[3:])

    def sum_generate_prompt(self, knowledge, halu_mode, individual=None):
        sum_generator_inst = "假设你是一个中文摘要数据生成者。请根据给定的背景知识，摘取其中部分作为摘取文档，并为摘取文档生成参考摘要。具体要求如下:\n"
        
        if individual:
            basic_principle_prompt="\n".join(individual[:3])
        else:
            basic_principle_prompt = self.basic_principle_prompt

        few_shots = ("给定如下例子:\n"
            "[背景知识]:中华人民共和国国防部，中华人民共和国国务院组成部门中华人民共和国国防部是中华人民共和国国务院的一个组成部门，是1954年9月第一届全国人民代表大会第一次会议决议设立的，它是根据《中华人民共和国宪法》设立的一个部门。 [1-2]基本职能是统一管理全国武装力量的建设工作，如人民武装力量的征集、编制、装备、训练、军事科研以及军人衔级、薪给等。中文名中华人民共和国国防部成立时间1954年9月性    质国务院组成部门现任部长董军 [10]目录1主要职责2军委领导3历任部长主要职责播报编辑根据《中华人民共和国宪法》规定，国务院领导和管理国防建设事业。国务院设立国防部，一切需要由政府负责的军事工作，则经国务院作出相应决定，通过国防部或以国防部的名义组织实施。国防部在接受国务院领导的同时也接受中央军事委员会的领导。中华人民共和国国防部是国务院的军事工作部门。它的基本职能是：统一管理全国武装力量的建设工作，如人民武装力量的征集、编制、装备、训练、军事科研以及军人衔级、薪给等。因此，国防部并没有实际军事指挥权。国防部是中华人民共和国国务院的军事部门。中华人民共和国成立后，于1954年9月设立国防部，隶属于国务院，主要负责国防建设方面的具体工作。 [2-3]军委领导播报编辑中国共产党中央军事委员会中华人民共和国中央军事委员会主席习近平主席习近平副主席张又侠、何卫东副主席张又侠、何卫东委员刘振立、苗华、张升民委员刘振立、苗华、张升民参考资料： [6-7] [9]历任部长播报编辑姓名性别任职时间彭德怀男1954年09月-1959年07月林彪男1959年-叶剑英男1975年01月-1978年01月徐向前男1978年02月-1981年02月耿飚男1981年03月-1982年10月张爱萍男1982年11月-1988年03月秦基伟男1988年04月-1993年02月迟浩田男1993年03月-2003年03月曹刚川男2003年03月-2008年3月 [16]梁光烈男2008年03月-2013年03月 [15]常万全男2013年03月-2018年03月 [14]魏凤和男2018年03月-2023年03月 [13]李尚福男2023年03月-2023年10月 [11-12]董军男2023年12月- [10] [3-5] [8]\n"
            "[摘取文档]:中华人民共和国国防部是中华人民共和国国务院的一个组成部门，是1954年9月第一届全国人民代表大会第一次会议决议设立的，它是根据《中华人民共和国宪法》设立的一个部门。基本职能是统一管理全国武装力量的建设工作，如人民武装力量的征集、编制、装备、训练、军事科研以及军人衔级、薪给等。\n"
            "[参考摘要]:中华人民共和国国防部，成立于1954年，是国务院的组成部门，主要负责全国武装力量的建设工作，包括征集、编制、装备、训练、科研以及军人管理等职能。该部门依据《中华人民共和国宪法》设立，但不具备实际的军事指挥权。\n\n"
            "生成的摘取文档和参考摘要请按照以下格式输出，要求摘取文档和参考摘要之间换行但不要空行，不同的文档摘要之间空行，其它地方不要换行:\n"
            "[摘取文档1]:...\n[参考摘要1]:...\n\n[摘取文档2]:...\n[参考摘要2]:...\n\n"
            "请为以下背景知识生成3组摘取文档和参考摘要:\n" + "[背景知识]:{}\n\n"
        )

        sum_generate_prompt_template = sum_generator_inst + basic_principle_prompt + few_shots
        sum_generate_prompt = sum_generate_prompt_template.format(knowledge)
        return sum_generate_prompt
    
    def halu_sum_prompt(self, doc, sum, individual=None):
        halu_sum_inst = "假设你是一个幻觉摘要生成者，请根据输入的文档、和参考摘要生成文档的一个幻觉摘要。具体要求如下:\n"
       
        if individual:
            halu_sum_principles="\n".join(individual[3:])
        else:
            halu_sum_principles = self.halu_sum_principles

        halu_sum_few_shots = ("给定如下例子:\n"
            "[文档]:1926年12月，周恩来任中共中央军事委员会书记。1927年5月25日，中共中央政治局在武汉召开常委会议，决定以中共湖北省委军委为基础，成立中共中央军事部（又称军人部），任命周恩来为军事部部长，同时担任军事委员会书记。军委委员有周恩来、王一飞、聂荣臻、顾顺章、颜昌颐、贺昌、罗亦农、邓中夏等，主要是进行组织、联络和政治工作。周恩来赴南昌领导起义期间，中共中央军事部部务工作由王一飞负责。8月9日，中共临时中央政治局会议决定，仍由周恩来负责中共中央军事部，王一飞继续代理部长。\n" 
            "[参考摘要]:1926年12月，周恩来任中共中央军事委员会书记。1927年5月，中共中央政治局决定成立中共中央军事部，周恩来任军事部部长兼书记，主要负责组织、联络和政治工作。周恩来赴南昌领导起义期间，王一飞代理部长职务。\n"
            "[幻觉摘要]:1926年12月，周恩来任中共中央军事委员会书记。1927年5月，中共中央政治局决定成立中共中央军事部，彭德怀任军事部部长兼书记，主要负责组织、联络和政治工作。周恩来赴南昌领导起义期间，毛泽东代理部长职务。\n\n"
            "[文档]:联勤保障部队直属于中央军事委员会，以武汉联勤保障基地为建制领导，下属无锡联勤保障中心、桂林联勤保障中心、西宁联勤保障中心、沈阳联勤保障中心、郑州联勤保障中心，以及解放军总医院、解放军疾病预防控制中心等。中央军委联勤保障部队主要由“一基地、五中心”组成，主要是整合了原总后勤部系统的有关保障力量。武汉联勤保障基地脱胎于原总后勤部武汉后方基地，而其他五大联勤保障中心分布在东部、南部、西部、北部、中部五大战区内，合理布局，有效支持各战区的作战需求。通过这一体系的改革，联勤保障部队实现了快速、有效的后勤保障能力，确保了作战部队的持续作战能力。\n"
            "[参考摘要]:中国人民解放军联勤保障部队直属中央军委，具有全国性指挥体系，并以武汉为总部，设有五个联勤保障中心分布在各大战区。通过整合原有后勤力量并优化资源配置，联勤保障部队在提供多方位后勤支持的同时，有效提升了后勤保障的时效性与针对性，确保了作战部队的快速响应与持续作战能力。\n"
            "[幻觉摘要]:中国人民解放军联勤保障部队直属中央军委，具有全国性指挥体系，并以北京为总部，设有八个联勤保障中心分布在各大战区。通过整合原有后勤力量并优化资源配置，联勤保障部队能够有效提升后勤保障的时效性与针对性，确保作战部队的快速响应与持续作战能力。\n\n"
            "请为以下文档生成幻觉摘要:\n"
            "[文档]:{}\n"
            "[参考摘要]:{}\n"
            "[幻觉摘要]:")
        
        halu_sum_template = halu_sum_inst + halu_sum_principles + halu_sum_few_shots
        halu_sum_prompt = halu_sum_template.format(doc, sum)
        return halu_sum_prompt
    
    def parse_sum_data(self, output, halu_mode, ent):
        def format_check(str_):
            pos1 = str_.find(']')
            pos2 = str_.find(':')
            if pos1 + 1 == pos2:
                return True
            else:
                return False
        doc_sum = output.split('\n\n')
        results = []
        for item in doc_sum:
            sum_dict = {}
            tmp = item.split('\n')
            
            if len(tmp) != 2:
                print("Format Error!")
                continue
            doc, sum = tmp[0], tmp[1]
            
            if format_check(doc) and format_check(sum):
                doc = ':'.join(doc.split(':')[1:])
                sum = ':'.join(sum.split(':')[1:])
                sum_dict["doc"] = doc
                sum_dict["ref_sum"] = sum
                sum_dict["halu_mode"] = halu_mode
                sum_dict["ent"] = ent
                results.append(sum_dict)
            else:
                print("Format Error!")
                continue
        return results

    def parse_halu_sum(self, output, sum_dict):
        output = output.rstrip()
        sum_dict["halu_sum"] = output
        return sum_dict
    
    def response_generate(self, knowledge, halu_mode="用户上下文不一致", individual=None):
        # message for further domain classification
        ent = knowledge['ent']
        # doc_sum pairs generaton
        doc_sum = []
        sum_propmt = self.sum_generate_prompt(knowledge, halu_mode, individual)
        self.payload["messages"][0]["content"] = sum_propmt
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            if response.status_code !=200:
                print("Error in API request, status code:", response.status_code)
                print("Response:", response.text)
                return []
            data = response.json()
            response_string = data["choices"][0]["message"]["content"]
            response_string = self.parse_sum_data(response_string, halu_mode, ent)
            # check answer here
            doc_sum.extend(response_string)
        except Exception as e:
            print("Error in get LLM API: ", e)

        return doc_sum
    
    def halu_response_generate(self, doc_sum, individual=None):
        sum_data = []
        for sum_pair in doc_sum:
            doc, ref_sum = sum_pair["doc"], sum_pair["ref_sum"]
            halu_sum_prompt = self.halu_sum_prompt(doc, ref_sum, individual)
            self.payload["messages"][0]["content"] = halu_sum_prompt
            try:
                response = requests.post(self.url, json=self.payload, headers=self.headers)
                data = response.json()
                response_string = data["choices"][0]["message"]["content"]
                response_string = self.parse_halu_sum(response_string, sum_pair)
                sum_data.append(response_string)
            except:
                print("Error in get LLM API!")
        return sum_data
    
    def halu_label_generate(self, *args, **kwargs):
        pass

    def generate(self, start_idx, end_idx, halu_mode, outfile=None, individual=None):
        data = self.data[start_idx: end_idx]
        generated_data = []
        for knowledge in tqdm(data):
            knowledge['knowledge'] = knowledge['knowledge'].rstrip()
            sum_pairs = self.response_generate(knowledge, halu_mode, individual)
            halu_data = self.halu_response_generate(sum_pairs, individual)
            generated_data.extend(halu_data)
        
        if outfile != None:
            to_jsonl(generated_data, outfile)

        return generated_data