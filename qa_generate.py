from generator.qa_generator import qa_generator
from generator.qa_generator import halu_modes
from generator.sum_generator import sum_generator
from evaluator.qa_evaluator import qa_evaluator
from corrector.qa_corrector import qa_corrector
import argparse
from utils import read_jsonl, to_jsonl
from tqdm import tqdm

def main(args):
    generate_agent = qa_generator(api_key=args.api_key, model=args.model)
    evaluate_agent = qa_evaluator(api_key=args.api_key, model=args.model)
    generate_agent.read_data(data_path=args.train_file)
    data_len = len(generate_agent.data)
    data = []
    for i in tqdm(range(data_len-1)):
        gen_data = generate_agent.generate(start_idx=i, end_idx=i+1, halu_mode=halu_modes[args.halu_mode])
        qualified_data = evaluate_agent.check_data(gen_data)
        data += qualified_data
        to_jsonl(data, args.output_file)

def parse_args(joint = False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='Qwen-72B', help="model for generation")
    parser.add_argument('--api_key', type=str, default='sk-', help="api for model")
    parser.add_argument('--halu_mode', type=int, default=1, help="halu mode for data generation")
    parser.add_argument('--train_file', type=str, default="./data/domain_data.jsonl", help="train path for data")
    parser.add_argument('--output_file', type=str, default="./data/halu_data0.jsonl", help="output path for data")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)