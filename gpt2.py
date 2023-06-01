import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from masking import PaddingMask, FutureMasking
from embeddings import WordEmbeddings, PositionalEmbedding, PositionalEncoding
from feedforward import PositionwiseFeedForward
from attention import MultiHeadAttention
from mqa import MultiQueryAttention
from transformers import AutoConfig, AutoTokenizer


class Block(nn.Module):
    def __init__(self, heads, dims, rate, dropout):
        super(Block, self).__init__()
        self.attn = MultiQueryAttention(heads, dims, dropout)
        self.fc = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = nn.LayerNorm(dims)
        self.ln_fc = nn.LayerNorm(dims)

    def forward(self, x, past, mask):
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)

        x = x + a
        x = x + self.fc(self.ln_fc(x))
        return x if self.training else (x, past)


class GPT2(nn.Module):
    def __init__(self,
                 layers: int,
                 pad_idx: int,
                 vocab_size: int,
                 max_seq_len: int,
                 heads: int,
                 dims: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super(GPT2, self).__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PaddingMask(pad_idx)
        self.future_masking = FutureMasking()

        self.positional_embedding = PositionalEmbedding(max_seq_len, dims)
        self.token_embedding = WordEmbeddings(vocab_size, dims, pad_idx)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            Block(heads, dims, rate, dropout)
            for _ in range(layers)])
        self.ln_head = nn.LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past=None,
                use_grad_ckpt: bool = False
                ):
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)

            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        return x if self.training else (x, present)


class GPT2CasualLM(nn.Module):
    def __init__(self, config, **kwargs):
        super(GPT2CasualLM, self).__init__()
        self.dims = config.n_embd
        self.pad_idx = config.pad_token_id
        self.bidirectional = kwargs.get("bidirectional", True)
        self.model = GPT2(layers=config.n_layer,
                          pad_idx=config.pad_token_id,
                          vocab_size=config.vocab_size,
                          max_seq_len=config.max_length,
                          heads=config.n_head,
                          dims=config.n_embd,
                          dropout=config.embd_pdrop,
                          bidirectional=self.bidirectional)

        self.fc = nn.Linear(config.n_embd, config.vocab_size)
        self.gelu = nn.GELU()

    def forward(self, x, past=None, labels=None):
        present = []
        if not self.training:
            x, present = self.model(x, past)
        else:
            x = self.model(x)
        x = self.gelu(x)  # (batch, seq_len, n_embd)
        x = self.fc(x)  # (batch, seq_len, vocab_size)

        if labels is not None:
            shifted_x = x[..., :-1, :].contiguous()  # (batch, seq_len-1, vocab_size)
            labels = labels[..., 1:].contiguous()  # (batch, seq_len-1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fct(shifted_x.view(-1, shifted_x.size(-1)), labels.view(-1))
            return (x, loss) if self.training else (x, loss, present)

        return x if self.training else (x, present)


class GPT2Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        items = {key: getattr(self, key) for key in dir(self) if not key.startswith("__") and not key.startswith("_")}
        return str(items)

    def __repr__(self):
        items = {key: getattr(self, key) for key in dir(self) if not key.startswith("__") and not key.startswith("_")}
        return str(items)


if __name__ == "__main__":
    import torch

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model parameters
    layers = 6
    pad_idx = 0
    vocab_size = 10000
    max_seq_len = 208
    heads = 8
    dims = 512
    rate = 4
    dropout = 0.1
    bidirectional = False

    # Create a sample input tensor
    batch_size = 1
    input_seq_len = 1024
    input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, input_seq_len))
    print(input_tensor.shape)

    model_ckpt = "rootacess/FlashCoder"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.add_tokens('<pad>')
    tokenizer.pad_token = "<pad>"

    model_config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                                              max_length=1024).to_dict()
    config = GPT2Config(**model_config)
    model = GPT2CasualLM(config)

    # Perform a forward pass
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)
    print("target shape: ", input_tensor.view(-1).shape)
