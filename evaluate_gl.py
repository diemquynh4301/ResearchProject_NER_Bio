import json
from src.evaluate import try_ast_eval, gather_entities, compute_f1_metrics
from src.model import load_tokenizer, load_model, generate, extract_block
from dataclasses import dataclass, field
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from tqdm import trange
from pathlib import Path
from src.utils import read_jsonl, write_jsonl
from typing import List
from collections import Counter

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
    guideline_idx: int = field(
        default=-1
    )
    n_prompt: int = field(
        default=2,
        metadata={"help": "Number of prompt that is used for evaluation."}
    )
    n_testdata: List[int] = field(
        default_factory=lambda: [0, -1],
        metadata={"help": "Indices of test data to evaluate."}
    )
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)

# -------------------------
# 1. Setup and Arguments
# -------------------------
parser = HfArgumentParser((EvalPrompt,))
config = parser.parse_args_into_dataclasses()[0]
output_dir = Path(config.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

prompt = Path(config.prompt_path).read_text()
guideline_history = read_jsonl(config.guideline_history_path)
current_guideline = guideline_history[config.guideline_idx]["guideline"]

data = read_jsonl(config.data_path)
if config.n_testdata is not None:
    start_idx, end_idx = config.n_testdata
    if end_idx < 0:
        end_idx = len(data)
    data = data[start_idx:end_idx]

# -------------------------
# 2. Model Initialization
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
# 3. Main Evaluation Loop
# -------------------------
print(f"Starting evaluation on {len(data)} samples...")
for i in trange(len(data)):
    example = data[i]
    turns = [{
        "role": "user",
        "content": prompt.format(
            guideline=current_guideline,
            text=example["text"]
        )
    }]

    raw_answer = generate(turns, tokenizer, model, generation_config)
    example["raw_answer"] = raw_answer

    extracted = extract_block(raw_answer, "json")
    example["extracted_block"] = extracted

    # Parse and handle potential None returns from try_ast_eval
    pred_list = try_ast_eval(extracted)
    example["pred_entities"] = pred_list if pred_list is not None else []

# -------------------------
# 4. Metric Computation
# -------------------------
gold_set = gather_entities(data, "entities")
pred_set = gather_entities(data, "pred_entities")

# Overall metrics
overall_metrics = compute_f1_metrics(pred_set, gold_set)

# Per-entity type metrics
entity_types = set([e[1] for e in gold_set]) 
per_entity_metrics = {}

for etype in entity_types:
    specific_gold = {e for e in gold_set if e[1] == etype}
    specific_pred = {e for e in pred_set if e[1] == etype}
    
    if len(specific_gold) > 0 or len(specific_pred) > 0:
        per_entity_metrics[etype] = compute_f1_metrics(specific_pred, specific_gold)

final_metrics = {
    "overall": overall_metrics,
    "per_entity": per_entity_metrics
}

# -------------------------
# 5. Error Analysis Logic
# -------------------------
error_analysis = []
all_false_positives = []

for example in data:
    # Set of tuples (text, type) for accurate comparison
    gold_entities = set((e['text'], e['type']) for e in example.get("entities", []))
    pred_entities = set((e['text'], e['type']) for e in example.get("pred_entities", []) if e)

    fp = pred_entities - gold_entities
    fn = gold_entities - pred_entities

    if fp or fn:
        # Detect Type Mismatches (Same text, different label)
        type_mismatches = []
        for f_text, f_type in fp:
            for n_text, n_type in fn:
                if f_text == n_text:
                    type_mismatches.append({"text": f_text, "pred": f_type, "gold": n_type})

        error_entry = {
            "text": example["text"],
            "false_positives": [list(e) for e in fp],
            "false_negatives": [list(e) for e in fn],
            "type_mismatches": type_mismatches,
            "error_count": len(fp) + len(fn)
        }
        error_analysis.append(error_entry)
        all_false_positives.extend([tuple(e) for e in fp])

# -------------------------
# 6. Final Outputs and Saving
# -------------------------
print("\n===== FINAL METRICS =====")
print(json.dumps(final_metrics, indent=2))

common_errors = Counter(all_false_positives).most_common(10)
print("\n===== TOP 10 POTENTIAL MISANNOTATIONS (FP) =====")
for err, count in common_errors:
    print(f"{err}: {count} times")

# Save files
write_jsonl(output_dir / "eval_results.jsonl", data)
write_jsonl(output_dir / "error_analysis.jsonl", error_analysis)

with open(output_dir / "metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print(f"\nEvaluation complete. Results saved to {output_dir}")