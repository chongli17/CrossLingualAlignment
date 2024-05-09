import json
import random
import os
from tqdm import tqdm
import argparse

from lang_dict import _LANG_NAME

# _LANG_NAME={
#     "en": "English",
#     "zh": "Chinese",
#     "th": "Thai",
#     "tr": "Turkish",
#     "sw": "Swahili"
# }


def get_random_item(in_list:list):
    idx = random.randint(0,len(in_list)-1)
    return in_list[idx]

def read_tsv_data(tsv_path:str):
    samples = []
    with open(tsv_path,"r", encoding="utf-8") as f:
        for line in f.readlines():
            samples.append(line.strip().split("\t"))
    return samples

def add2list_p(keys, data_list, reverse_prob):
    assert(len(keys)==2 and keys[0]["id"] == keys[1]["id"])
    k1 = keys[0]
    k2 = keys[1]

    c_d = {0:0, 1:0}

    for k in k1.keys():
        if k != "instruction" or k1[k]==None or len(k1[k]) == 0 or k2[k]==None or len(k2[k])==0 or (k1[k] == k2[k]):
            continue
        
        if random.random()<reverse_prob:
            data_list.append([k2[k], k1[k]])
            c_d[1] += 1
        else:
            data_list.append([k1[k], k2[k]])
            c_d[0] += 1
            
    return data_list, c_d

def read_json(path:str):
    with open(path, "r") as f:
        samples = json.load(f)
    return samples

def build_bilingual(
    languages=["en","zh"], #The first language is the pivotal language.
    cross_probability=0.5, 
    repeat_times=3,
    bx_dir:str="./Bactrian-X/data",
    out_dir:str="./data",
):
    assert len(languages)==2, f"Number of languages({len(languages)}) input must be equal to 2."

    out_file = f"AFP.{'_'.join(languages)}.X{cross_probability}.R{repeat_times}.json"
    out_path = os.path.join(out_dir, out_file)

    lang_set = set(languages)
    all_samples = []

    data_dict = {}
    sample_num = -1
    for lang in languages:
        l_path = os.path.join(bx_dir, f"{lang}.json")
        l_data = read_json(l_path)
        l_data = sorted(l_data, key= lambda x: x['id'])
        sample_num = len(l_data)
        data_dict[lang] = l_data
    
    # load translation data from bactrian-x
    all_trans = []

    for si in range(sample_num):
        k0 = data_dict[languages[0]][si]
        k1 = data_dict[languages[1]][si]

        all_trans, count_dict = add2list_p(keys=[k0, k1], data_list=all_trans, reverse_prob=0)

    random.shuffle(all_trans)
    curr_i = 0

    # Cross-lingual Instruction Following
    CIF_FORMAT="{instruction} {input} Answer in {tgt_lang}: {output}"
    CIF_NOIN="{instruction} Answer in {tgt_lang}: {output}"

    cif_num = int(repeat_times*sample_num) 
    repeat_times = int(repeat_times) + 1 if type(repeat_times) != int else repeat_times
    sample_ids = [i%sample_num for i in range(sample_num*repeat_times)]
    random.shuffle(sample_ids)

    for s_i in sample_ids:
        src_lang = get_random_item(languages)
        if random.random() < cross_probability:
            lang_set.remove(src_lang)
            tgt_lang = get_random_item(list(lang_set))
            lang_set.add(src_lang)
        else:
            tgt_lang = src_lang

        # tgt_lang = get_random_item(languages)

        s_dict = data_dict[src_lang][s_i].copy()
        s_dict["output"] = data_dict[tgt_lang][s_i]["output"]
        s_dict["tgt_lang"] = _LANG_NAME[tgt_lang]

        curr_format = CIF_FORMAT if s_dict["input"] is not None and len(s_dict["input"]) >0 else CIF_NOIN
        clm_text = curr_format.format_map(s_dict)

        curr_trans = all_trans[curr_i]
        curr_i += 1
        if curr_i >= len(all_trans):
            random.shuffle(all_trans)
            curr_i = 0
        
        w_s = {
            "sent0": curr_trans[0],
            "sent1": curr_trans[1],
            "clm_text": clm_text,
            "clm_prompt_len": len(clm_text)-len(s_dict["output"]),
        }

        all_samples.append(w_s)
        if len(all_samples) > cif_num:
            break
    
    random.shuffle(all_samples)
    with open(out_path, "w") as f:
        for w_s in all_samples:
            f.write(json.dumps(w_s, ensure_ascii=False)+"\n")

def build_crosslingual(
    languages=["en", "zh", "ar"], #The first language is the pivotal language.
    cross_probability=0.5, 
    change_probability:float=0.5, 
    repeat_times=3,
    bx_dir:str="./Bactrian-X/data",
    out_dir:str="./data",
):
    assert len(languages)>2, f"Number of languages({len(languages)}) input must be more than 2."

    out_file = f"AFP.{'_'.join(languages)}.X{cross_probability}.C{change_probability}.R{repeat_times}.json"
    out_path = os.path.join(out_dir, out_file)

    lang_set = set(languages)
    all_trans = {}
    cif_count = {}

    data_dict = {}
    sample_num = -1
    for lang in languages:
        l_path = os.path.join(bx_dir, f"{lang}.json")
        l_data = read_json(l_path)
        l_data = sorted(l_data, key= lambda x: x['id'])
        sample_num = len(l_data)
        data_dict[lang] = l_data

    # Translation pairs from bx
    for li in range(len(languages)-1):
        curr_trans = []

        for si in range(sample_num):
            k0 = data_dict[languages[0]][si]
            k1 = data_dict[languages[li+1]][si]

            curr_trans, count_dict = add2list_p(keys=[k0, k1], data_list=curr_trans, reverse_prob=0)

        all_trans[f"{languages[0]}->{languages[li+1]}"] = curr_trans
    
    # Add cross-lingual instruction following data
    choices = [i+1 for i in range(len(languages)-1)]

    CIF_FORMAT="{instruction} {input} Answer in {tgt_lang}: {output}"
    CIF_NOIN="{instruction} Answer in {tgt_lang}: {output}"

    cif_num = int(repeat_times*sample_num) 
    repeat_times = int(repeat_times) + 1 if type(repeat_times) != int else repeat_times
    sample_ids = [i%sample_num for i in range(sample_num*repeat_times)]
    random.shuffle(sample_ids)

    all_samples = []
    for s_i in sample_ids:
        src_lang = languages[0]
        tgt_lang = languages[get_random_item(choices)]
        trans_key = f"{src_lang}->{tgt_lang}"

        # keep the target language is the same with the source language
        if random.random() >= cross_probability:
            src_lang = get_random_item(languages)
            tgt_lang = src_lang
        # change the position of target language and pivotal language
        elif random.random() < change_probability:
            src_lang = tgt_lang
            tgt_lang = languages[0]
            
        cif_count[f"{src_lang}->{tgt_lang}"] = cif_count[f"{src_lang}->{tgt_lang}"] + 1 if f"{src_lang}->{tgt_lang}" in cif_count else 1

        s_dict = data_dict[src_lang][s_i].copy()
        s_dict["output"] = data_dict[tgt_lang][s_i]["output"]
        s_dict["tgt_lang"] = _LANG_NAME[tgt_lang]

        curr_format = CIF_FORMAT if s_dict["input"] is not None and len(s_dict["input"]) >0 else CIF_NOIN
        clm_text = curr_format.format_map(s_dict)

        curr_trans = get_random_item(in_list=all_trans[trans_key])

        w_s = {
            "sent0": curr_trans[0],
            "sent1": curr_trans[1],
            "clm_text": clm_text,
            "clm_prompt_len": len(clm_text)-len(s_dict["output"]),
        }

        all_samples.append(w_s)
        if len(all_samples) > cif_num:
            break

    random.shuffle(all_samples)
    with open(out_path, "w") as f:
        for w_s in all_samples:
            f.write(json.dumps(w_s, ensure_ascii=False)+"\n")

    for k,v in cif_count.items():
        print(f"{k}:{v}")

def build_translate(
    file_path="./data/en2zh.tsv",
    out_dir:str="./data",
    change_probability=0.5,
    repeat_times=1.5,
    src_lang="en",
    tgt_lang="zh",
):
    all_samples = []

    out_path = os.path.join(out_dir, f"AFP.{src_lang}-{tgt_lang}-trans.C{change_probability}.R{repeat_times}.json")

    trans_data = read_tsv_data(file_path)

    sample_num = len(trans_data)
    
    MMT_FORMATS=[
        "{input} Translate into {tgt_lang}: {output}", 
        "{input} = {output}"
    ]
    
    cla_num = int(repeat_times*sample_num) 
    repeat_times = int(repeat_times) + 1 if type(repeat_times) != int else repeat_times
    sample_ids = [i%sample_num for i in range(sample_num*repeat_times)]
    random.shuffle(sample_ids)

    for s_i in sample_ids:
        curr_trans = trans_data[s_i]
        curr_src = src_lang
        curr_tgt = tgt_lang

        if random.random() <change_probability:
            curr_src = tgt_lang
            curr_tgt = src_lang
            curr_trans = [curr_trans[1], curr_trans[0]]

        clm_text = get_random_item(MMT_FORMATS).format_map({"input":curr_trans[0], "tgt_lang": _LANG_NAME[curr_tgt], "output": curr_trans[1]})
        
        w_s = {
            "sent0": curr_trans[0],
            "sent1": curr_trans[1],
            "clm_text": clm_text,
            "clm_prompt_len": len(clm_text)-len(curr_trans[1]),
        }

        all_samples.append(w_s)
        if len(all_samples) > cla_num:
            break

    random.shuffle(all_samples)
    with open(out_path, "w") as f:
        for w_s in all_samples:
            f.write(json.dumps(w_s, ensure_ascii=False)+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to build training samples for AFP.')

    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-m', '--mode', type=str, default="bilingual", choices=["bilingual", "crosslingual", "translation"])
    parser.add_argument('-l', '--languages', type=str, default=["en", "zh"], nargs="*", help="The list of language names to align, where the first one is the pivotal language.")
    parser.add_argument('-x', '--cross-probability', type=float, default=0.5)
    parser.add_argument('-c', '--change-probability', type=float, default=0.5)
    parser.add_argument('-r', '--repeat', type=float, default=1)
    parser.add_argument('-t', '--translation-file', type=str, default="CrossLingualAlignment/data/opus.en2zh.tsv")
    parser.add_argument('-b', '--bactrian-dir', type=str, default="CrossLingualAlignment/Bactrian-X/data")
    parser.add_argument('-o', '--output-dir', type=str, default="CrossLingualAlignment/data/")

    args = parser.parse_args()

    random.seed(args.seed)

    if args.mode == "bilingual":
        build_bilingual(
            languages=args.languages,
            cross_probability=args.cross_probability, 
            repeat_times=args.repeat,
            bx_dir=args.bactrian_dir,
            out_dir=args.output_dir,
        )
    elif args.mode == "crosslingual":
        build_crosslingual(
            languages=args.languages,
            cross_probability=args.cross_probability, 
            change_probability=args.change_probability,
            repeat_times=args.repeat,
            bx_dir=args.bactrian_dir,
            out_dir=args.output_dir,
        )
    else:
        assert len(args.languages) == 2, "The number of languages for translation file must be equal to two."
        build_translate(
            file_path=args.translation_file,
            out_dir=args.output_dir,
            change_probability=args.change_probability,
            repeat_times=args.repeat,
            src_lang=args.languages[0],
            tgt_lang=args.languages[1]
        )