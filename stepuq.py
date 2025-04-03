import re
import os
import json
from config import args
from tqdm import tqdm
from torchmetrics import AUROC

from utils import get_tokenwise_importance, extract_p, extract_p_t_importance, extract_keywords, \
                    extract_keykeywords, extract_keystep, weighted_sum, extract_probing_confidence

from src.model.llama2_predict import predict, model_init
from src.model.llama2_predict import predict, model_init, generate_model_answer

from sentence_transformers.cross_encoder import CrossEncoder

# AP strategies: Probas-mean, Probas-min, Token-SAR
def compute_step_uncertainty():
    with open(f"{args.output_path}/output_v1.json", 'r', encoding='utf-8') as f:
        json_data = []
        for line in f.readlines():
            dic = json.loads(line)
            json_data.append(dic)

    output_dir = f"{args.output_path}/confidences/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    with open(f"{args.output_path}/confidences/output_v1_{args.uq_engine}.json", "a", encoding="utf-8") as f:
        _, tokenizer, _ = model_init(args)
        measure_model = CrossEncoder(model_name="cross-encoder/stsb-roberta-large", num_labels=1)
        for idx, line in enumerate(tqdm(json_data, total=len(json_data))):
            question = line['question']
            correct_answer = line['correct answer']
            llm_answer = line['llm answer']
            if llm_answer == "":
                continue

            keyword_token_probability = line['keyword token probability']
            if keyword_token_probability == {}:
                continue
            contribution_scores = line['keyword contribution']
            if contribution_scores == {}:
                continue

            if args.uq_engine in ["probas-mean", "probas-min"]:
                probabilities, contribution_dict = extract_p(keyword_token_probability, contribution_scores)
            elif args.uq_engine in ["token-sar"]:
                probabilities, contribution_dict = extract_p_t_importance(question, keyword_token_probability, tokenizer, measure_model, contribution_scores)

            probabilities = {key: weighted_sum(value) for key, value in probabilities.items()}
            contributions = {key: sum(value)/len(value) for key, value in contribution_dict.items()}
            
            # CoT-UQ
            total_sum = sum(probabilities[key] * contributions[key] for key in probabilities)
            total_weight = sum(contributions[key] for key in contributions)
            if total_weight == 0:
                p_list = [v for v in probabilities.values()]
                confidence = sum(p_list) / len(p_list)
            else:
                confidence = total_sum / total_weight

            formatted_data = {
                # "id": id,
                "question": question,
                "correct answer": correct_answer,
                "llm answer": llm_answer,
                "confidence": confidence,
                "llm answer token probability": line['llm answer token probability']
            }
            f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")

# Self-Probing
def self_probing_uncertainty():
    with open(f"{args.output_path}/output_v1.json", 'r', encoding='utf-8') as f:
        json_data = []
        for line in f.readlines():
            dic = json.loads(line)
            json_data.append(dic)

    model, tokenizer, device = model_init(args)

    output_dir = f"{args.output_path}/confidences/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    for idx, line in enumerate(tqdm(json_data[:], total=len(json_data[:]))):
        with open(f"{args.output_path}/confidences/output_v1_{args.uq_engine}.json", "a", encoding="utf-8") as f:
            # give labels for open-ended llm answer
            # id = line['id']
            question = line['question']
            correct_answer = line['correct answer']
            llm_answer = line['llm answer']
            if llm_answer == "":
                continue

            keyword_token_probability = line['keyword token probability']
            if keyword_token_probability == {}:
                continue

            contribution_scores = line['keyword contribution']
            if contribution_scores == {}:
                continue

            # keywords = extract_keywords(keyword_token_probability, contribution_scores)

            # keywords = extract_keykeywords(contribution_scores)

            # step_wise_keywords = line['step-wise keywords']
            # keyword_token_probability = line['keyword token probability']
            # if keyword_token_probability == {}:
            #     continue
            # """
            # Question: xxxx
            # Possible answer: xxx
            
            # Q: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {e.g. (B)} to be correct, not the one you think correct, please only include the numerical number]%```
            # """
            
            # prompt = f"Question: {question_text}\nPossible Answer:{answer}\nQ: How likely is the above answer to be correct? Analyze the possible answer, provide your reasoning concisely and give your confidence in this answer using the following format:\n```Confidence: [the probability of answer {answer} to be correct, not the one you think correct, please only include the numerical number]%```"
            prompt = f"Question: {question}\nPossible Answer: {llm_answer}\nQ: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {llm_answer} to be correct, not the one you think correct, please only include the numerical number]%```\nConfidence: "
            
            # our_prompt_keywords = f"Question: {question}\nPossible Answer: {llm_answer}\nKeywords during reasoning to the possible answer: {keywords}\nQ: Considering these keywords as additional information, how likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {llm_answer} to be correct, not the one you think correct, please only include the numerical number]%```\nConfidence: "
            
            llm_response = line['llm response']
            if llm_response == "":
                continue

            # our_prompt_steps = f"Question: {question}\nPossible Answer: {llm_answer}\nA step-by-step reasoning to the possible answer: {llm_response}\nQ: Considering these reasoning steps as additional information, how likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {llm_answer} to be correct, not the one you think correct, please only include the numerical number]%```\nConfidence: "

            contribution_scores = line['keyword contribution']
            key_step = extract_keystep(llm_response, contribution_scores)

            our_prompt_keystep = f"Question: {question}\nPossible Answer: {llm_answer}\nThe most critical step in reasoning to the possible answer: {key_step}\nQ: Considering this critical reasoning step as additional information, how likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {llm_answer} to be correct, not the one you think correct, please only include the numerical number]%```\nConfidence: "

            try_time = 0
            while try_time < args.try_times:
                # response = predict(args, prompt, model, tokenizer)
                # response = predict(args, our_prompt_keywords, model, tokenizer)
                # response = predict(args, our_prompt_steps, model, tokenizer)
                response = predict(args, our_prompt_keystep, model, tokenizer)

                confidence = extract_probing_confidence(response)

                if confidence == None:
                    print(f"Cannot extract confidence, Please check the response: {response}")
                    print(f"Current try is {try_time + 1}")
                    try_time += 1
                    continue
  
                formatted_data = {
                    # "id": id,
                    "question": question,
                    "correct answer": correct_answer,
                    "llm answer": llm_answer,
                    "confidence": confidence,
                    "probing response": response,
                }
                f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                break
        if try_time >= args.try_times:
            print(f'#####Cannot extract confidence from:#####\n{response}\nRecord and Skip')
            error_dir = f"{args.output_path}/confidences/probing_errors"
            if not os.path.exists(error_dir):
                os.makedirs(error_dir) 
            with open(f"{args.output_path}/confidences/probing_errors/output_v1_conf_ours_self_probing_keystep.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

# P(True)
def p_true_uncertainty():

    with open(f"{args.output_path}/output_v1.json", 'r', encoding='utf-8') as f:
        json_data = []
        for line in f.readlines():
            dic = json.loads(line)
            json_data.append(dic)

    model, tokenizer, device = model_init(args)

    token_a = tokenizer.encode('A', add_special_tokens=False)[0]
    token_b = tokenizer.encode('B', add_special_tokens=False)[0]

    output_dir = f"{args.output_path}/confidences/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    with open(f"{args.output_path}/confidences/output_v1_{args.uq_engine}.json", "a", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(json_data[:], total=len(json_data[:]))):
            # give labels for open-ended llm answer
            # id = line['id']
            question = line['question']
            correct_answer = line['correct answer']
            llm_answer = line['llm answer']
            if llm_answer == "":
                continue

            prompt = \
                f"""The problem is: {question}
            A student submitted: {llm_answer}
            Is the student's answer:
            (A) True
            (B) False
            The student's answer is: """

            keyword_token_probability = line['keyword token probability']
            if keyword_token_probability == {}:
                continue

            contribution_scores = line['keyword contribution']
            if contribution_scores == {}:
                continue
            # keywords = extract_keywords(keyword_token_probability, contribution_scores)

            keywords = extract_keykeywords(contribution_scores)

            # our_prompt_keywords = \
            #     f"""The problem is: {question}
            # A student submitted: {llm_answer}
            # The student explained the answer, which included the following keywords []
            # Refering to these keywords, is the student's answer:
            # (A) True
            # (B) False
            # The student's answer is: """

            our_prompt_keywords = \
                f"""The problem is: {question}
            A student submitted: {llm_answer}
            The student explained the answer, which included the following keywords {keywords}
            Considering these keywords as additional information, is the student's answer:
            (A) True
            (B) False
            The student's answer is: """

            llm_response = line['llm response']
            if llm_response == "":
                continue

            # our_prompt_steps = \
            #     f"""The problem is: {question}
            # A student submitted: {llm_answer}
            # The student explained the answer, which included a step-by-step reasoning: {llm_response}
            # Considering these reasoning steps as additional information, is the student's answer:
            # (A) True
            # (B) False
            # The student's answer is: """

            # contribution_scores = line['keyword contribution']
            # key_step = extract_keystep(llm_response, contribution_scores)

            # our_prompt_keystep = \
            #     f"""The problem is: {question}
            # A student submitted: {llm_answer}
            # The student explained the answer, where the most critical step is: {key_step}
            # Considering this critical reasoning step as additional information, is the student's answer:
            # (A) True
            # (B) False
            # The student's answer is: """

            _, _, scores, _ = generate_model_answer(args, our_prompt_keywords, model, tokenizer, device,
                                                                   max_new_tokens=1,
                                                                   output_scores=True,
                                                                   temperature = args.temperature)

            p_true = scores[0, [token_a, token_b]].softmax(dim=0)[0].numpy()

            formatted_data = {
                # "id": id,
                "question": question,
                "correct answer": correct_answer,
                "llm answer": llm_answer,
                "confidence": float(p_true),
            }
            f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    if args.uq_engine in ['probas-mean', 'probas-min', 'token-sar']:
        compute_step_uncertainty()
    elif args.uq_engine == 'p-true':
        p_true_uncertainty()
    elif args.uq_engine == 'self-probing':
        self_probing_uncertainty()
    else:
        raise ValueError(f"Invalid UQ engine: {args.uq_engine}")

