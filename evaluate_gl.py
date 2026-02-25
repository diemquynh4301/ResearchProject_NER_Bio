
import json

from src.evaluate import try_ast_eval, gather_entities,compute_f1_metrics
from src.model import load_tokenizer, load_model, generate, extract_block
from dataclasses import dataclass, field
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from tqdm import trange
from pathlib import Path
from src.utils import read_jsonl, write_jsonl
from typing import List



@dataclass
class EvalPrompt():
    model_name_or_path: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "HF model id"}
    )
    output_dir: str = field(
        default="results/eval",
        metadata={"help": "Output directory"}
    )
    prompt_path: str = field(
        default="data/prompts/default_evaluate.txt",
        metadata={"help": "Path to prompt file. Should be a txt file."}
    )
    data_path: str = field(
        default="data/genia/test.jsonl",
        metadata={"help": "Test jsonl"}
    )
    guideline_history_path: str = field(
        default="results/toy/guideline_history.jsonl",
        metadata={"help": "Guideline history"}
    )
    guideline_idx: int = field(default=-1)
    n_prompt: int = field(
        default=2,
        metadata={"help": "Number of prompt that is used for evaluation."}
    )
    n_testdata: List[int] = field(
    default_factory=lambda: [1, 2],
    metadata={"help": "Indices of test data to evaluate."}
    )

    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)


# -------------------------
# Parse arguments
# -------------------------
parser = HfArgumentParser((EvalPrompt,))
config = parser.parse_args_into_dataclasses()[0]
output_dir = Path(config.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# -------------------------
# Load prompt
# -------------------------
prompt = Path(config.prompt_path).read_text()

# -------------------------
# Load guideline history
# -------------------------
guideline_history = read_jsonl(config.guideline_history_path)

current_guideline = guideline_history[config.guideline_idx]["guideline"]

# -------------------------
# Load test examples
# -------------------------
data = read_jsonl(config.data_path)


if config.n_testdata is not None:
    data = [
        data[i] for i in config.n_testdata if 0 <= i < len(data)
    ]

# -------------------------
# Load model + tokenizer
# -------------------------
quantization_config = BitsAndBytesConfig(
    load_in_8bit=config.load_in_8bit,
    load_in_4bit=config.load_in_4bit
)

tokenizer = load_tokenizer(config.model_name_or_path)
model = load_model(config.model_name_or_path, quantization_config)

generation_config = GenerationConfig(
    max_new_tokens=2048,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# -------------------------
# Evaluation loop
# -------------------------
for i in trange(len(data)):

    example = data[i]

    turns = [{
        "role": "user",
        "content": prompt.format(
            guideline=current_guideline,
            example=example["text"]
        )
    }]

    raw_answer = generate(turns, tokenizer, model, generation_config)

    # store raw answer
    example["raw_answer"] = raw_answer

    # Extract markdown block
    extracted = extract_block(raw_answer, block_type="md")
    print(extracted)
    example["extracted_block"] = extracted


    # ---- parse extracted block ----
    data["pred_entities"] = try_ast_eval(extracted)

# ==================================================
# Compute metrics
# ==================================================

gold_set = gather_entities(data, "entities")
pred_set = gather_entities(data, "pred_entities")

metrics = compute_f1_metrics(pred_set, gold_set)

print("\n===== FINAL METRICS =====")
print(json.dumps(metrics, indent=2))


# ==================================================
# Save everything
# ==================================================

# write_jsonl(output_dir / "eval_results.jsonl", examples)

# with open(output_dir / "metrics.json", "w") as f:
#     json.dump(metrics, f, indent=2)