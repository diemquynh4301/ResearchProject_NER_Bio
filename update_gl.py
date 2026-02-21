from dataclasses import dataclass, field
from pathlib import Path
import random

from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from tqdm import trange

from src.model import load_tokenizer, load_model, generate, extract_block
from src.utils import read_jsonl, write_jsonl

@dataclass
class ExperimentConfig():
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    output_dir: str = field(
        metadata={"help": "Path of the directory used to save outputs (guideline_history.jsonl)"}
    )
    prompt_path: str = field(
        default="data/prompts/default_update.txt",
        metadata={"help": "Path to prompt file. Should be a txt file."}
    )
    base_guideline_path: str = field(
        default="data/guidelines/default_genia.txt",
        metadata={"help": "Path to base guideline file. Should be a txt file."}
    )
    examples_path: str = field(
        default="data/genia/train.jsonl",
        metadata={"help": "Path to examples file. Should be a jsonl file."}
    )
    n_updates: int = field(
        default=5,
        metadata={"help": "Number of guideline update to perform."}
    )
    n_examples: int = field(
        default=5,
        metadata={"help": "Number of examples used for each update."}
    )
    random_seed: int = field(
        default=42, 
        metadata={"help": "Random seed used to shuffle examples."}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "See https://huggingface.co/docs/transformers/v5.2.0/en/main_classes/quantization#transformers.BitsAndBytesConfig"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "See https://huggingface.co/docs/transformers/v5.2.0/en/main_classes/quantization#transformers.BitsAndBytesConfig"}
    )


parser = HfArgumentParser((ExperimentConfig))
xp_config = parser.parse_args_into_dataclasses()[0]

prompt = Path(xp_config.prompt_path).read_text()
current_guideline = Path(xp_config.base_guideline_path).read_text()
examples = read_jsonl(xp_config.examples_path)

output_dir = Path(xp_config.output_dir)
    
if not output_dir.is_dir():
    output_dir.mkdir()

quantization_config = BitsAndBytesConfig(
    load_in_8bit=xp_config.load_in_8bit, load_in_4bit=xp_config.load_in_4bit
)

tokenizer = load_tokenizer(xp_config.model_name_or_path)
model = load_model(xp_config.model_name_or_path, quantization_config)

generation_config = GenerationConfig(
    max_new_tokens = 2048,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

random.seed(xp_config.random_seed)
random.shuffle(examples)

guideline_history = [{"guideline":current_guideline, "examples":[]}]

for i in trange(xp_config.n_updates):
    
    example_start = i*xp_config.n_examples
    current_examples = examples[example_start:example_start+xp_config.n_examples]
    turns = [
        {"role":"user", "content":prompt.format(guideline=current_guideline, examples=current_examples)}
    ]
    
    answer = generate(turns, tokenizer, model, generation_config)

    updated_guideline = extract_block(answer)
    if updated_guideline:
        current_guideline = updated_guideline
    
    guideline_history.append({"guideline":current_guideline, "examples":current_examples})
    write_jsonl(f"{output_dir}/guideline_history.jsonl", guideline_history)
