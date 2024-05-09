import random

# EXAMPLE_FORMAT="Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, A. entailment B. neutral or C. contradiction?\nAnswer: {answer}"
# EXAMPLE_FORMAT="Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, entailment, neutral, or contradiction?\nAnswer: {label}"
# EXAMPLE_FORMAT="Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, A. entailment, B. neutral or C. contradiction?\nAnswer: {answer}. {label}"

# default setting
_EXAMPLE_FORMAT={
    "en": "{sentence1}{sentence2}{sentence3}{sentence4}{label}",
    "zh": "{sentence1}{sentence2}{sentence3}{sentence4}{label}",
    "ar": "{sentence1}{sentence2}{sentence3}{sentence4}{label}"
}

# _CHOICES=["A", "B", "C"]
# _CHOICES=["entailment", "neutral", "contradiction"]

# _CHOICES={
#     "en": ["entailment", "neutral", "contradiction"],
#     "zh": ["蕴涵", "中立", "矛盾"],
#     "ar": ["الحاجة", "محايدة.", "تناقض"]
# }

_CHOICES={
    "en": ["1", "2"],
    "zh": ["1", "2"],
    "ar": ["1", "2"]
}

_GOLD_LABELS=["1", "2"]

_LABEL2ANS={
    "1": "1",
    "2": "2",
}

_CONNECTOR="</s>"

# _FEW_SHOT_FORMAT="{demos}Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, A. entailment, B. neutral, or C. contradiction?\nAnswer: "
# _FEW_SHOT_FORMAT="{demos}Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, entailment, neutral, or contradiction?\nAnswer: "
# _FEW_SHOT_FORMAT="{demos}Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between premise and hypothesis, A. entailment, B. neutral, or C. contradiction?\nAnswer: "

_FEW_SHOT_FORMAT={
    "en": "{demos}{sentence1}{sentence2}{sentence3}{sentence4}{choice}",
    "zh": "{demos}{sentence1}{sentence2}{sentence3}{sentence4}{choice}",
    "ar": "{demos}{sentence1}{sentence2}{sentence3}{sentence4}{choice}"
}

def get_random_items_no_replace(in_list:list,num:int=1000):
    assert num <= len(in_list)
    random.shuffle(in_list)
    return in_list[:num]

def read_prompt(file_path:str):
    file_handle = open(file_path)
    prompts = []

    while True:
        line = file_handle.readline()
        if not line:
            break
        line = line.strip()
        prompts.append(line)
    
    return prompts

class XSTORYCLOZE_FEW_SHOTS:
    prompt_language:str="en"

    EXAMPLE_FORMAT:str=_EXAMPLE_FORMAT["en"]
    CHOICES:list=_CHOICES["en"]
    FEW_SHOT_FORMAT:str=_FEW_SHOT_FORMAT["en"]
    CONNECTOR:str=_CONNECTOR

    seed:int=0
    num_shots:int=3
    choices:list=[]
    samples:list=[]
    demos_str:str=""
    _idx:int=0

    def read_tsv(self, filepath:str):
        samples = []
        with open(filepath, encoding="utf-8") as f:
            for key,line in enumerate(f):
                if key == 0:
                    continue
                line = line.strip().split("\t")
                sample = {
                        "id": line[0],
                        "sentence1": line[1],
                        "sentence2": line[2],
                        "sentence3": line[3],
                        "sentence4": line[4],
                        "answer1": line[5],
                        "answer2": line[6],
                        "label": line[7],
                        "answer": _LABEL2ANS[line[7]]
                    }
                samples.append(sample)
        return samples

    def __init__(
            self, 
            prompt_sample_path:str, 
            test_sample_path:str, 
            num_shots:int=3, 
            seed:int=0, 
            prompt_language:str="en",
            prompt_format:dict=None,
        ) -> None:
        
        self.num_shots = num_shots
        self.seed = seed
        self.prompt_language = prompt_language

        # print(prompt_format)

        if prompt_format is not None:
            self.EXAMPLE_FORMAT = prompt_format["EXAMPLE_FORMAT"]
            self.CHOICES = prompt_format["CHOICES"]
            self.FEW_SHOT_FORMAT = prompt_format["FEW_SHOT_FORMAT"]
            self.CONNECTOR = prompt_format["CONNECTOR"]
        else:
            self.EXAMPLE_FORMAT = _EXAMPLE_FORMAT[prompt_language]
            self.CHOICES = _CHOICES[prompt_language]
            self.FEW_SHOT_FORMAT = _FEW_SHOT_FORMAT[prompt_language]


        random.seed(seed)

        demos_str = ""
        if num_shots > 0:
            prompt_samples = get_random_items_no_replace(in_list=self.read_tsv(prompt_sample_path), num=num_shots)
            
            for prompt_sample in prompt_samples:
                # prompt_sample["label"] = self.CHOICES[_GOLD_LABELS.index(prompt_sample["label"])]
                # label of xtorycloze is converted into answer1 or answer2
                prompt_sample["label"] = prompt_sample[f"answer{prompt_sample['answer']}"]
                prompt_str = self.EXAMPLE_FORMAT.format_map(prompt_sample)
                demos_str += f"{prompt_str}{self.CONNECTOR}"
        
        self.demos_str = demos_str
        self.samples = []
        test_samples = self.read_tsv(test_sample_path)
        for test_sample in test_samples:
            test_sample["demos"] = demos_str
            # test_sample["label"] = self.CHOICES[_GOLD_LABELS.index(test_sample["label"])]
            # label of xtorycloze is converted into answer1 or answer2
            test_sample["label"] = test_sample[f"answer{test_sample['answer']}"]
            
            # We don't input choice here
            test_prompt = self.FEW_SHOT_FORMAT.replace("{choice}","").format_map(test_sample)

            self.samples.append({
                # For build prompt outside
                "sample": test_sample,
                "prompt": test_prompt,
                # 1, 2
                "answer": test_sample["answer"],
                # The sentence in answer1 or answer2
                "label": test_sample["label"],
            })
        
        for i, print_sample in enumerate(get_random_items_no_replace(self.samples, num=3)):
            print(f'Random Sample-{i+1}')
            print(f'Prompt: \n{print_sample["prompt"]}')
            print(f'Label: \n{print_sample["label"]}')
            print(f'Answer: \n{print_sample["answer"]}')

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> dict:
        return self.samples[index]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < len(self.samples):
            x = self.samples[self._idx]
            self._idx += 1
            return x
        else:
            raise StopIteration

    def get_choice_idx(self, label:str):
        if label not in self.CHOICES:
            return _GOLD_LABELS.index(label)
        return self.CHOICES.index(label)
    
    def choices(self, sample:dict):
        return [sample["sample"]["answer1"], sample["sample"]["answer2"]]
    
    def choice_idx(self, sample:dict):
        return self.get_choice_idx(sample["answer"])