import json
import os
import logging
import re
import math

import torch

Dataset_Folder = 'dataset'

def print_exp(args, return_flag=0):
    info = ''
    for k, v in vars(args).items():
        info += '{}:{}\n'.format(k, v)
    print('---------------experiment args---------------')
    print(info)
    print('---------------------------------------------')
    if return_flag == 0:
        return
    elif return_flag == 1:
        return info
    else:
        pass

def load_data(args):
    decoder = json.JSONDecoder()
    questions = []
    answers = []
    ids = []
    types = []
    datapath = args.datapath if args.datapath else '{}/{}/{}.json'.format(Dataset_Folder, args.dataset, args.dataset)
    # read dataset file
    if args.dataset.lower() in ['2wikimhqa', 'gsm8k', 'hotpotqa', 'svamp', 'asdiv']:
        with open(datapath) as f:
            if args.dataset.lower() in ['hotpotqa','2wikimhqa']:
                json_data = []
                for line in f.readlines():
                    dic = json.loads(line)
                    json_data.append(dic)
            else:
                json_data = json.load(f)

            for idx, line in enumerate(json_data):
                if args.dataset.lower() in ['svamp', 'asdiv']:
                    if line['Body'][-1] != '.':
                        q = line['Body'].strip() + ". " + line["Question"].strip()
                    else:
                        q = line['Body'].strip() + " " + line["Question"].strip()
                    a = float(line["Answer"])
                    id = line["ID"]
                elif args.dataset in ['hotpotQA']:
                    q = line['question']
                    a = line['answer']
                    id = line['id']
                    t = line['type']
                    types.append(t)
                elif args.dataset in ['2WikimhQA']:
                    if line['type'] == "inference":
                        q = line['question']
                        a = line['answer']
                        id = line['_id']
                        t = line['type']
                        types.append(t)
                    else: continue
                elif args.dataset.lower() in ['gsm8k']:
                    q = line['question']
                    a = float(line['answer'])
                    id = 'temp_{}'.format(idx)
                else:
                    raise ValueError('not support dataset: {}'.format(args.dataset))
                questions.append(q)
                answers.append(a)
                ids.append(id)
            print("The Number of Different Questions: ", len(questions))
    else:
        raise ValueError('not support dataset: {}'.format(args.dataset))

    if args.test_end == 'full':
        if args.dataset.lower() in ['hotpotqa', '2wikimhqa']:
            return questions[int(args.test_start):], answers[int(args.test_start):], ids[int(args.test_start):], types[int(args.test_start):]
        else:
            return questions[int(args.test_start):], answers[int(args.test_start):], ids[int(args.test_start):]
    else:
        return questions[int(args.test_start):int(args.test_end)], answers[
                                                                   int(args.test_start):int(args.test_end)], ids[
                                                                                                             int(args.test_start):int(
                                                                                                                 args.test_end)]

def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()

#############################################
########### Inference Refining ###########
############################################# 

def is_effectively_empty(obj):
    
    if obj is None:
        return True

    if isinstance(obj, (int, float)) and obj == 0:
        return True

    if obj == "":
        return True

    if isinstance(obj, list):
        return all(is_effectively_empty(item) for item in obj)
    
    if isinstance(obj, dict):
        if len(obj) == 0: 
            return True
        return all(is_effectively_empty(value) for value in obj.values())
    return False

def clean_words(word):
  return word.replace(" ", "").replace(".", "").replace("\"", "").replace("\n", "").replace("_", "").replace("Ġ", "").lower()

def find_subsequence_position(sub_sequence, long_sequence):
    len_long = long_sequence.size(0)
    len_sub = len(sub_sequence) 

    sub_sequence_tensor = torch.tensor(sub_sequence, device=long_sequence.device)
    
    for i in range(len_long - len_sub + 1):
        if torch.equal(long_sequence[i:i + len_sub], sub_sequence_tensor):
            return i 
    return -1  

def step_exacts_2_list(response):
    # Split response into lines and filter out empty lines
    lines = response.splitlines()
    lines = [line for line in lines if line.strip()]

    keywords_by_step = []
    contributions_by_step = []
    valid_response_text = []

    for line in lines:
        # Match lines starting with "Step X:"
        match = re.search(r"Step \d+: (.+)", line)
        if match:
            if "(/" not in line or "/)" not in line:
                continue  # Skip invalid lines

            # Extract keywords with contributions
            keywords_w_contribution = match.group(1).split("; ")

            # Check for valid format and skip invalid lines
            if any("(/" not in key_w_c or "/)" not in key_w_c for key_w_c in keywords_w_contribution):
                continue

            try:
                # Extract keywords and contributions
                keywords = [key_w_c.split("(/")[0].strip() for key_w_c in keywords_w_contribution]
                contributions = [int(key_w_c.split("(/")[1].split("/)")[0].strip()) for key_w_c in keywords_w_contribution]
            except ValueError:
                return False  # Return False if contributions cannot be converted to int

            for i in contributions:
                if i > 10:
                    return False

            keywords_by_step.append(keywords)
            contributions_by_step.append(contributions)
            valid_response_text.append(line)  # Add valid lines from the original response

    # If no valid lines are found, return False
    if not valid_response_text:
        return False

    return "\n".join(valid_response_text), keywords_by_step, contributions_by_step

def parse_response_to_dict(response):
    steps = {}  
    final_answer = None

    # Match Final Answer
    match = re.search(r"Final Answer:\s*(.+?)\s*(?=(\n|$))", response, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        response_before_final_answer = response[:match.end()].strip()
    else:
        return None, None, None

    # Match Steps
    matches = list(re.finditer(r'(Step \d+):', response_before_final_answer))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_before_final_answer)
        segment = response[start:end].strip()
        steps[match.group(1)] = segment

    return_response = response_before_final_answer
    return final_answer, steps, return_response

def find_token_indices(tokens, word):
    word_len = len(word.replace(" ", ""))
    
    for start_index in range(len(tokens)):
        combined_text = ""
        end_index = start_index       
        while end_index < len(tokens) and len(combined_text) < word_len:
            combined_text += tokens[end_index]
            if clean_words(combined_text) == clean_words(word):
                return start_index, end_index
            end_index += 1
    
    return -1, -1 

def is_word_in_sentence(sentence, word):
    pattern = re.escape(word)
    match = re.search(pattern, sentence, re.IGNORECASE)
    return True if match else False

def match_final_answer_token_ids(args, original_tokens, response_tokens, original_token_ids):
    if "llama2" in args.model_engine:
        final_answer_tokens = ['Final', '▁Answer', ':']
    elif "llama3" in args.model_engine:
        final_answer_tokens = ['Final', 'ĠAnswer', ':']
    
    end_index = None
    end_index_original = None

    for i in range(len(response_tokens) - len(final_answer_tokens) + 1):
        if response_tokens[i:i + len(final_answer_tokens)] == final_answer_tokens:
            start_index = i
            end_index = i + len(final_answer_tokens)
            break
    
    if end_index == None or end_index + 1 == len(response_tokens):
        return None, None

    for i in range(len(original_tokens) - len(final_answer_tokens) + 1):
        if original_tokens[i:i + len(final_answer_tokens)] == final_answer_tokens:
            end_index_original = i + len(final_answer_tokens)
            break

    if end_index_original == None:
        return None, None

    if response_tokens[end_index] in ["▁", "Ġ"]:
        end_index += 1
        end_index_original += 1

    target_tokens = response_tokens[end_index: ]

    final_answer_token_ids = original_token_ids[end_index_original : end_index_original + len(target_tokens)]

    return end_index_original, final_answer_token_ids.data.cpu().numpy()

def setup_log(args):    
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.output_path, "output_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    # log.debug(f"#########{args.name}############")
    return log


#############################################
########### Reasoning Enhanced UQ ###########
############################################# 

def get_tokenwise_importance(question, generated_text, tokenizer, measure_model):
    token_importance_list = []
    tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))

    token_importance = []
    # measure cosine similarity by removing each token and compare the similarity
    for token in tokenized:
        similarity_to_original = measure_model.predict([question + generated_text,
                                                        question + generated_text.replace(
                                                            tokenizer.decode(token, skip_special_tokens=True),
                                                            '')])
        token_importance.append(1 - torch.tensor(similarity_to_original))

    token_importance = torch.tensor(token_importance).reshape(-1)
    return token_importance

def extract_p(keyword_token_probability, contribution_scores = None):
    if contribution_scores == None:
        return_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                # if key.isdigit(): 
                #     value_to_add = values[0] 
                # else:
                #     value_to_add = values[0] 
                # value_to_add = sum(values)/len(values)
                value_to_add = min(values)
                # value_to_add = max(values)
                if key in return_dict:
                    return_dict[key].append(value_to_add)
                else:
                    return_dict[key] = [value_to_add]
        return return_dict
    else:
        return_keyword_dict = {}
        return_contribution_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                # if key.isdigit(): 
                #     value_to_add = values[-1] 
                # else:
                #     value_to_add = values[0] 
                # value_to_add = sum(values)/len(values)
                value_to_add = min(values)
                # value_to_add = max(values)
                if key in return_keyword_dict:
                    return_keyword_dict[key].append(value_to_add)
                    return_contribution_dict[key].append(contribution_scores[step][key])
                else:
                    return_keyword_dict[key] = [value_to_add]
                    return_contribution_dict[key] = [contribution_scores[step][key]]
        return return_keyword_dict, return_contribution_dict

def extract_p_t_importance(question, keyword_token_probability, tokenizer, measure_model, contribution_scores = None):
    if contribution_scores == None:
        return_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                token_importance = get_tokenwise_importance(question, key, tokenizer, measure_model).data.cpu().numpy()
                if len(token_importance) == len(values):
                    weighted_score = ((token_importance / sum(token_importance)) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) > 0:
                    start = len(token_importance) - len(values)
                    # end = len(values) - len(token_importance)
                    weighted_score = ((token_importance[start:] / sum(token_importance[start:])) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) < 0:
                    start = len(values) - len(token_importance)
                    end = len(token_importance) - len(values)
                    weighted_score = ((token_importance[:] / sum(token_importance[:])) * values[start:])
                    value_to_add = sum(weighted_score)
                else:
                    value_to_add = sum(values) / len(values)
                if key in return_dict:
                    return_dict[key].append(value_to_add)
                else:
                    return_dict[key] = [value_to_add]
        return return_dict
    else:
        return_keyword_dict = {}
        return_contribution_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                token_importance = get_tokenwise_importance(question, key, tokenizer, measure_model).data.cpu().numpy()
                if len(token_importance) == len(values):
                    weighted_score = ((token_importance / sum(token_importance)) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) > 0:
                    start = len(token_importance) - len(values)
                    end = len(values) - len(token_importance)
                    weighted_score = ((token_importance[start:] / sum(token_importance[start:])) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) < 0:
                    start = len(values) - len(token_importance)
                    end = len(token_importance) - len(values)
                    weighted_score = ((token_importance[:] / sum(token_importance[:])) * values[start:])
                    value_to_add = sum(weighted_score)
                else:
                    value_to_add = sum(values) / len(values)
                if key in return_keyword_dict:
                    return_keyword_dict[key].append(value_to_add)
                    return_contribution_dict[key].append(contribution_scores[step][key])
                else:
                    return_keyword_dict[key] = [value_to_add]
                    return_contribution_dict[key] = [contribution_scores[step][key]]
        return return_keyword_dict, return_contribution_dict

def extract_keywords(keyword_token_probability, contribution_scores = None):
    return_list = []
    for step, inner_dict in keyword_token_probability.items():
        for key, values in inner_dict.items():
            if key in return_list:
                continue
            # elif contribution_scores[step][key] < 7 or contribution_scores[step][key] == None:
            #     continue
            else:
                return_list.append(key)

    return return_list

def extract_keykeywords(contribution_scores = None):
    return_list = []
    added_keys = set()
    key_list = []
    con_list = []
    for step, inner_dict in contribution_scores.items():
        for key, value in inner_dict.items():
            if key in return_list:
                continue

            key_list.append(key)
            con_list.append(value)
            
            if value >= 4 and key not in added_keys:
                return_list.append(key)
                added_keys.add(key)
    
    if len(return_list) < 3:
        pairs = list(zip(key_list, con_list)) 
        pairs.sort(key=lambda x: x[1], reverse=True)
        if len(pairs) < 3:
            top_3_keys = [pair[0] for pair in pairs]
        else:
            top_3_keys = [pair[0] for pair in pairs[:3]]
        return_list = top_3_keys

    return return_list

def extract_keystep(llm_response, contribution_scores = None):
    if contribution_scores == None:
        return ""
    else:
        step_avg_cons = []
        for step, inner_dict in contribution_scores.items():
            con_list = []
            for key, value in inner_dict.items():
                if value == None:
                    print("##### Contribution Score has None Value #####")
                    continue
                con_list.append(value)
            if len(con_list) == 0:
                avg_con = 0
            else:
                avg_con = sum(con_list) / len(con_list)
            step_avg_cons.append(avg_con)
        max_con = max(step_avg_cons) 
        step_idx = len(step_avg_cons) - 1 - step_avg_cons[::-1].index(max_con)

        return_text = llm_response.split("\n")[step_idx].strip()

        return_text = return_text.split(f"Step {str(step_idx + 1)}: ")[-1]
        
        return return_text.strip()

def weighted_sum(values):
    if len(values) == 1:
        return values[0] 
    weights = [math.exp(-c) for c in values]  
    sum_weights = sum(weights)  
    normalized_weights = [w / sum_weights for w in weights] 
    result = sum(w * c for w, c in zip(normalized_weights, values)) 
    return result 

def extract_probing_confidence(response):
    match = re.search(r"(\d+(\.\d+)?)%", response)
    if match:
        return float(match.group(1)) / 100  

    first_line = response.strip().split("\n")[0]

    try:
        confidence = float(first_line)
        if confidence > 1:
            confidence /= 100
        return confidence
    except ValueError:
        pass  

    match = re.search(r"(\d+(\.\d+)?)", first_line)
    if match:
        confidence = float(match.group(1))
        if confidence > 1:
            confidence /= 100
        return confidence

    return None  

