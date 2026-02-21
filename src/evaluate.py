import warnings
import ast

def try_ast_eval(text):
    try:
        items = ast.literal_eval(text)
        return [i for i in items if "text" in i and "type" in i]
    except Exception as e:
        warnings.warn(f"ast literal eval fail:\n\n{e}\n\n{text}")
        return []


def gather_entities(examples, entity_key):
    outputs = []
    for i, x in enumerate(examples):
        outputs.extend([f"{i}__{e['text']}__{e['type']}" for e in x[entity_key]])
    return outputs
    

def compute_f1_metrics(candidates, references):
    
    if len(candidates) == 0 and len(references) == 0:
        return {"tp":0, "fp":0, "fn":0, "p":1, "r":1, "f1":1}
    elif len(candidates) == 0:
        return {"tp":0, "fp":0, "fn":len(references), "p":0, "r":0, "f1":0}
    elif len(references) == 0:
        return {"tp":0, "fp":len(candidates), "fn":0, "p":0, "r":0, "f1":0}
    
    tp = len(references.intersection(candidates))
    fp = len(candidates - references)
    fn = len(references - candidates)
    
    if tp == 0:
        return { "tp":tp, "fp":fp, "fn":fn, "p":0, "r":0, "f1":0 }
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))
    
    return { "tp":tp, "fp":fp, "fn":fn, "p":p, "r":r, "f1":f1 }

