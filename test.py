from generator.qa_generator import qa_generator
from generator.qa_generator import halu_modes
from evaluator.qa_evaluator import qa_evaluator
from corrector.qa_corrector import qa_corrector

def test_generator():
    data_path="data/safety_sample_data.jsonl"
    generator = qa_generator(model="Qwen")
    generator.read_data(data_path=data_path)
    _ = generator.generate(0, 2, halu_modes[-3], "data/test.jsonl")

def test_evaluator():
    data_path = "data/evaluator_raw.jsonl"
    output_path = "data/evaluator_output.jsonl"
    evaluator = qa_evaluator("Qwen")
    evaluator.update_data(data_path=data_path)
    _ = evaluator.evaluate(output_path)

def test_corrector():
    data_path="data/evaluator_output_test.jsonl"
    output_path="data/adjusted_prompts_test_new.jsonl"
    halu_mode=halu_modes[2]
    corrector = qa_corrector(halu_mode, model='Qwen')
    corrector.update_data(data_path=data_path)
    _ = corrector.correct(output_path)

def test_generator_round():
    generator = qa_generator(model="Qwen")
    data_path = "data/round1_correct.jsonl"
    generator.read_data(data=data_path)
    generator.generate(0, 30, "data/roun2_gen.jsonl")

if __name__ == '__main__':
    test_generator_round()
    
    