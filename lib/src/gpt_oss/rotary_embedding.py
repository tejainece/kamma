import torch
from safetensors.torch import save_file
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb
from transformers import GptOssConfig

torch.manual_seed(42)

# Parameters
dim = 64
max_position_embeddings = 2048
base = 10000.0
seq_len = 10
batch_size = 2
head_dim = dim 
n_heads = 4

config = GptOssConfig(
    hidden_size=dim * n_heads,
    num_attention_heads=n_heads,
    max_position_embeddings=max_position_embeddings,
    rope_theta=base,
)

rotary_emb = GptOssRotaryEmbedding(config=config)

# Inputs
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) # [batch, seq]
q = torch.randn(batch_size, n_heads, seq_len, head_dim)
k = torch.randn(batch_size, n_heads, seq_len, head_dim)

cos, sin = rotary_emb(q, position_ids)

q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

tensors = {
    "inv_freq": rotary_emb.inv_freq.contiguous(),
    "position_ids": position_ids.contiguous(),
    "cos": cos.contiguous(),
    "sin": sin.contiguous(),
    "q": q.contiguous(),
    "k": k.contiguous(),
    "q_out": q_out.contiguous(),
    "k_out": k_out.contiguous(),
}

save_file(tensors, "gpt_oss_rotary_embedding.safetensors")
print("Generated gpt_oss_rotary_embedding.safetensors using transformers.GptOssRotaryEmbedding")
