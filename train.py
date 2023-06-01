import torch
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, get_scheduler
from datasets import load_dataset
from gpt2 import GPT2CasualLM, GPT2Config
from generate import generate
from accelerate import Accelerator
import wandb
from torch.optim import AdamW
from huggingface_hub import HfApi
from argparse import Namespace
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm


class CodeDataset(IterableDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        for sample in self.dataset:
            buffer = sample['content']
            tokenizer_buffer = self.tokenizer(buffer, truncation=True, max_length=1023)
            tokenizer_buffer['input_ids'].append(self.tokenizer.eos_token_id)
            tokenizer_buffer['attention_mask'].append(1)
            yield tokenizer_buffer


class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


def create_dataloaders(const_len=True):
    train_data = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset('codeparrot/codeparrot-clean-valid', split="train", streaming=True)

    if const_len:
        const_train_dataset = ConstantLengthDataset(tokenizer, train_data,
                                                    seq_length=args.seq_length)

        const_valid_dataset = ConstantLengthDataset(tokenizer, valid_data,
                                                    seq_length=args.seq_length)
        const_len_train_dataloader = DataLoader(const_train_dataset, batch_size=args.train_batch_size)
        const_len_eval_dataloader = DataLoader(const_valid_dataset, batch_size=args.valid_batch_size)
        return const_len_train_dataloader, const_len_eval_dataloader

    train_dataset = CodeDataset(train_data, tokenizer)
    eval_dataset = CodeDataset(train_data, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.valid_batch_size, collate_fn=data_collator)

    return train_dataloader, eval_dataloader


train_config = {"train_batch_size": 1,
                "valid_batch_size": 1,
                "weight_decay": 0.1,
                "shuffle_buffer": 1000,
                "learning_rate": 5e-4,
                "lr_scheduler_type": "cosine",
                "num_warmup_steps": 2000,
                "gradient_accumulation_steps": 1,
                "max_train_steps": 150000,
                "max_eval_steps": 2000,
                "seq_length": 1024,
                "seed": 1,
                "save_checkpoint_steps": 5000}

args = Namespace(**train_config)

model_ckpt = "rootacess/FlashCoder"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.add_tokens('<pad>')
tokenizer.pad_token = "<pad>"

model_config = AutoConfig.from_pretrained("gpt2",
                                          vocab_size=len(tokenizer),
                                          pad_token_id=tokenizer.pad_token_id,
                                          max_length=1024,
                                          n_layer=6).to_dict()
config = GPT2Config(**model_config)

wandb.init(project="FlashCoder",
           config={**train_config, **model_config})

model = GPT2CasualLM(config)
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps, )

train_dataloader, eval_dataloader = create_dataloaders()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model,
                                                                          optimizer,
                                                                          train_dataloader,
                                                                          eval_dataloader)


def log_metrics(step, metrics):
    if accelerator.is_main_process:
        wandb.log(metrics)


def current_lr():
    return optimizer.param_groups[0]['lr']


def evaluate():
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        # uncomment the below line if using CodeDataset
        # batch = batch['input_ids']
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs[1].repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


api = HfApi()
model.train()
completed_steps = 0
for step, batch in tqdm(enumerate(train_dataloader, start=1), total=args.max_train_steps):
    # Uncomment the below line if using CodeDataset
    # batch = batch['input_ids']
    loss = model(batch, labels=batch)[1]
    metrics = {
        'lr': current_lr(),
        'samples': step * samples_per_step,
        'steps': completed_steps,
        'loss/train': loss.item()
    }
    log_metrics(step, metrics)
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)

    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1

    if step % args.save_checkpoint_steps == 0:
        print(f'Evaluating and saving model checkpoint at step: {step}')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            model_path = f"model.bin"
            torch.save(unwrapped_model.state_dict(), model_path)
            api.upload_file(path_or_fileobj=model_path,
                            path_in_repo="pytorch_model.bin",
                            repo_id="rootacess/FlashCoder",
                            repo_type="model")
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
print('Evaluating and saving FINAL model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    model_path = f"final_model.bin"
    torch.save(unwrapped_model.state_dict(), model_path)
    api.upload_file(path_or_fileobj=model_path,
                    path_in_repo="pytorch_model.bin",
                    repo_id="rootacess/FlashCoder",
                    repo_type="model")

# Testing the final model
text = "d"
checkpoint = model_path
op = generate(text, config, tokenizer, checkpoint=checkpoint)
print(op['input_ids'].shape)
print(op['generated_text'])
