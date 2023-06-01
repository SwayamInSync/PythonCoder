import torch
from transformers import AutoTokenizer, AutoConfig
from gpt2 import GPT2CasualLM, GPT2Config
from generate import generate


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
model = GPT2CasualLM(config)

# loading from a checkpoint
# get the checkpoint till 10k steps from here:
# https://drive.google.com/file/d/1QpBwTMqeHRIkFOIL3ZMSIAt05Qt1Z6Fn/view?usp=sharing

checkpoint = "pytorch_model.bin"
model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

# generating text:
text = '''def hello():
# print hello
'''

op = generate(text, config, tokenizer, checkpoint=checkpoint, top_k=1, top_p=0.9, temperature=0.2)
print(op['input_ids'].shape)
print(op['generated_text'])

'''
Expected OP:
def hello():
# print hello
def self self():def hello self self self self self hello self self self self self self self hello
'''
