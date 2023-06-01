import torch
from transformers.models.gpt2 import GPT2TokenizerFast
import torch.nn.functional as F

import gpt2


def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = torch.topk(logits, k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
    mask = logits < min_values
    logits = logits.masked_fill(mask, -1e10)
    return logits


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float('inf'))
    return logits


def generate(text: str,
             config: gpt2.GPT2Config,
             tokenizer: GPT2TokenizerFast,
             checkpoint=None,
             model=None,
             max_length=20,
             temperature=0.1,
             top_k=None,
             top_p=None,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    input_ids = tokenizer(text, return_tensors="pt")['input_ids'].to(device)

    if model is None:
        model = gpt2.GPT2CasualLM(config, bidirectional=False).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("checkpoint loaded")

    model.eval()
    past = None
    for _ in range(max_length):
        logits, past = model(input_ids, past=past)
        logits = logits[:, -1] / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        if top_p is not None:
            logits = top_p_logits(logits, top_p)

        probabilities = F.softmax(logits, -1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token == tokenizer.eos_token_id:
            break
    output_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    return {'input_ids': input_ids, 'generated_text': output_text}


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer

    model_ckpt = "rootacess/FlashCoder"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.add_tokens('<pad>')
    tokenizer.pad_token = "<pad>"
    config = AutoConfig.from_pretrained("gpt2",
                                        vocab_size=len(tokenizer),
                                        pad_token_id=tokenizer.pad_token_id,
                                        max_length=1024,
                                        n_layer=6).to_dict()
    config = gpt2.GPT2Config(**config)

    text = "def hello"

    op = generate(text, config, tokenizer, top_k=2, top_p=0.9)
    print(op['generated_text'])
