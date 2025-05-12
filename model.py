import torch

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate(text, model, tokenizer, k=5):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        top_k=k,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id
    )
    
    completions = []
    for output in outputs:
        completion = tokenizer.decode(output, skip_special_tokens=True)
        completions.append(completion)
    
    return completions

def logits(text, model, tokenizer, k=5):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        
    # Get logits for the next token
    next_token_logits = outputs.logits[:, -1, :]
    
    # Get top k logits and their corresponding token ids
    top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
    
    # Convert to lists
    logits_list = top_k_logits[0].tolist()
    
    return logits_list


class Model:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model, self.tokenizer = load_model(model_name)

    def generate(self, text, k=5):
        return generate(text, self.model, self.tokenizer, k)
    
    def logits(self, text, k=5):
        return logits(text, self.model, self.tokenizer, k)