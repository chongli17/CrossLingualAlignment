import random
import json


# default setting
_EXAMPLE_FORMAT={
    "en": "{sentence_part1}{label}{sentence_part2}",
    "zh": "{sentence_part1}{label}{sentence_part2}",
    "ar": "{sentence_part1}{label}{sentence_part2}",
}

_CHOICES={
    "en": [1, 2],
    "zh": [1, 2],
    "ar": [1, 2]
}

_GOLD_LABELS=[1, 2]

_LABEL2ANS={
    "1": 1,
    "2": 2,
    1:   1,
    2:   2,
}

_CONNECTOR="</s>"

_FEW_SHOT_FORMAT={
    "en": "{demos}{sentence_part1}{choice}{sentence_part2}",
    "zh": "{demos}{sentence_part1}{choice}{sentence_part2}",
    "ar": "{demos}{sentence_part1}{choice}{sentence_part2}",
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

class XWINOGRAD_FEW_SHOTS:
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

    def read_json(self, filepath:str):
        samples = []
        with open(filepath, encoding="utf-8") as f:
            for key,line in enumerate(f):
                sample = json.loads(line.strip())
                # label of xwinograd is converted into option1 or option2
                sample["label"] = sample[f'option{sample["answer"]}']
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
            prompt_samples = get_random_items_no_replace(in_list=self.read_json(prompt_sample_path), num=num_shots)
            
            for prompt_sample in prompt_samples:
                # label of xwinograd is already converted into option1 or option2
                prompt_str = self.EXAMPLE_FORMAT.format_map(prompt_sample)
                demos_str += f"{prompt_str}{self.CONNECTOR}"
        
        self.demos_str = demos_str
        self.samples = []
        test_samples = self.read_json(test_sample_path)
        for test_sample in test_samples:
            test_sample["demos"] = demos_str
            # label of xwinograd is already converted into option1 or option2
            
            # We don't input choice here
            test_prompt = self.FEW_SHOT_FORMAT.replace("{choice}","").format_map(test_sample)

            self.samples.append({
                # For build prompt outside
                "sample": test_sample,
                "prompt": test_prompt,
                # 1, 2
                "answer": test_sample["answer"],
                # The sentence in option1 or option2
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
        return [sample["sample"]["option1"], sample["sample"]["option2"]]
    
    def choice_idx(self, sample:dict):
        return self.get_choice_idx(sample["answer"])