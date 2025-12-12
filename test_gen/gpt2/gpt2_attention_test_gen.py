import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
from safetensors.torch import save_file

torch.manual_seed(42)

# Parameters
batch_size = 2
seq_len = 10
embed_dim = 64
num_heads = 4
head_dim = embed_dim // num_heads
dropout = 0.0 # Deterministic for testing

config = GPT2Config(
    n_embd=embed_dim,
    n_head=num_heads,
    attn_pdrop=dropout,
    resid_pdrop=dropout,
    n_layer=12,
)
config._attn_implementation = 'eager'

attn = GPT2Attention(config, layer_idx=0)
attn.eval() # Ensure dropout is off

hidden_states = torch.randn(batch_size, seq_len, embed_dim)
# Attention mask in HF GPT2 is usually (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
# For simple causal attention, we can pass None or a specific mask.
# Let's use a simple mask for testing.
attention_mask = torch.ones(batch_size, 1, 1, seq_len)
# Make the last token masked out to test masking
attention_mask[:, :, :, -1] = 0.0 
# HF attention mask is usually 0.0 for keep, and vast negative for mask in some versions, 
# BUT GPT2Model usually processes the mask before passing to attention.
# GPT2Attention expects `attention_mask` of shape (batch_size, num_heads, seq_len, total_seq_len)
# wait, actually GPT2Attention logic:
# if attention_mask is not None:
#    attn_weights = attn_weights + attention_mask
# So it expects the additive mask (0 for keep, large negative for mask).
# Let's create an additive mask manually to avoid confusion.
additive_attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
additive_attention_mask[:, :, :, -1] = -10000.0

# Forward pass
print("Running forward pass...")
# GPT2Attention forward signature:
# forward(hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False)
output_tuple = attn(
    hidden_states,
    attention_mask=additive_attention_mask,
)
attn_output = output_tuple[0]
present = output_tuple[1]

print("Output shape:", attn_output.shape)

# Helper to get weights
def get_linear_weights(linear_layer):
    # HF Conv1D weights are (in_features, out_features)
    # Dart Linear weights are (out_features, in_features)
    # So we need to transpose.
    w = linear_layer.weight
    b = linear_layer.bias
    return w.t().contiguous(), b.contiguous()

c_attn_w, c_attn_b = get_linear_weights(attn.c_attn)
c_proj_w, c_proj_b = get_linear_weights(attn.c_proj)

tensors = {
    "hidden_states": hidden_states.contiguous(),
    "attention_mask": additive_attention_mask.contiguous(),
    "output": attn_output.contiguous(),
    # Weights
    "c_attn.weight": c_attn_w,
    "c_attn.bias": c_attn_b,
    "c_proj.weight": c_proj_w,
    "c_proj.bias": c_proj_b,
}

save_file(tensors, "gpt2_attention.safetensors")
print("Saved gpt2_attention.safetensors")
