import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--try_times", type=int, default=20,
        help="try times for meaningful reasoning process"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help=""
    )
    parser.add_argument(
        '--dataset', default='NLI',
        help="dataset",
        choices=["ASDiv", "2WikimhQA", "gsm8k", "hotpotQA", "logiQA", "AddSub", "SingleEq", "CommonsenseQA", "StrategyQA", "CAD", "TriviaQA", "Math", "NLI", "svamp"]
    )
    parser.add_argument(
        "--datapath", default=None, type=str, help='file path'
    )
    parser.add_argument(
        "--api_key", default="", type = str, help='gpt api_key'
    )
    parser.add_argument(
        "--model_engine", default='llama2-7b', help="model engine",
        choices=["llama3-1_8B", "llama2-13b"]
    )
    parser.add_argument(
        "--uq_engine", default='probas-mean', help="uncertainty quantification engine",
        choices=["probas-mean", "probas-min", "token-sar", "p-true", "self-probing"]
    )
    parser.add_argument(
        "--model_path", default='llama3-1_8B', help="your local model path",
        choices=["llama3-1_8B", "llama2-13b"]
    )
    parser.add_argument(
        "--output_path", default='output/llama-3.1-8B/', help="your local output path"
    )
    parser.add_argument(
        "--test_start", default='0', help='string, number'
    )
    parser.add_argument(
        "--test_end", default='full', help='string, number'
    )
    parsed_args = parser.parse_args()
    return parsed_args


args = parse_arguments()
