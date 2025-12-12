import torch
from safetensors.torch import save_file
import json
import os
from transformers import AutoModelForCausalLM, AutoConfig

torch.manual_seed(42)

def main():
    model_path = "/home/tejag/projects/dart/ai/tensor/models/llm/gpt_oss"
    print(f"Loading model from {model_path}...")
    
    # Load model with low_cpu_mem_usage to avoid OOM if possible, and float16
    # Note: trust_remote_code=True might be needed if it's a custom model not in standard transformers lib yet,
    # but since we have the code in transformers (implied by previous context), we might not need it.
    # However, 'gpt_oss' type suggests it might be custom.
    # We will try loading it.
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
        # Ensure model is in float16 as requested/planned, though for rotary logic float32 is fine too. 
        # But let's stick to plan for consistency with real usage.
        model.to(dtype=torch.float16)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")
    
    # Access Rotary Embedding
    # Attempt to locate it
    # Structure typically: model.model.layers[0].self_attn.rotary_emb
    try:
        # Adjust access path based on inspection if needed
        # For typical GPT-NeoX / Llama / etc it varies.
        # Let's inspect the first layer.
        if hasattr(model, "model"):
            layers = model.model.layers
        elif hasattr(model, "gpt_oss"):
            layers = model.gpt_oss.layers # config said "model_type": "gpt_oss"
        else:
            print(f"Unknown model structure. Attributes: {dir(model)}")
            # Fallback search
            layers = getattr(model, "layers", [])
            
        if len(layers) > 0:
            layer_0 = layers[0]
            # Try to find attention
            attn = getattr(layer_0, "self_attn", getattr(layer_0, "attention", None))
            if attn is None:
                print("Could not find attention module in layer 0")
                return
            
            rotary_emb = getattr(attn, "rotary_emb", None)
            if rotary_emb is None:
                print(f"Could not find rotary_emb in attention module. Attributes: {dir(attn)}")
                return
        else:
            print("No layers found.")
            return

    except Exception as e:
        print(f"Error accessing rotary embedding: {e}")
        return

    print(f"Found rotary embedding: {rotary_emb}")
    print(f"Inv Freq: {rotary_emb.inv_freq.shape}")

    # Generate Inputs
    config = model.config
    head_dim = config.head_dim
    n_heads = config.num_attention_heads
    batch_size = 2
    # gpt-oss uses sliding window attention, let's use a meaningful seq len
    seq_len = 128 

    # Determine device from rotary_emb
    # Some implementations have 'inv_freq' buffer
    device = torch.device("cpu")
    if hasattr(rotary_emb, "inv_freq"):
        device = rotary_emb.inv_freq.device
    elif hasattr(model, "device"):
        device = model.device
    
    print(f"Using device: {device}")

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16) 
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Forward Pass
    # Note: The model's rotary implementation might expect specific inputs.
    # Usually (x, position_ids)
    
    with torch.no_grad():
        cos, sin = rotary_emb(q, position_ids)
        
        # We also want to use the model's apply function if possible, 
        # but it might be a module function or static method.
        # from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
        # We can try to import it again or use what we imported before.
        # If it's custom code, it should be in the `trust_remote_code` path usually, 
        # but here we rely on the installed `transformers` package having `gpt_oss`?
        # The user's env seemed to accept the import in the previous run's 'Research' phase code (even if it wasn't run).
        # Wait, the previous run failed because of 'module not found torch'.
        # The VENV run worked!
        # So we can import `apply_rotary_pos_emb`.
        
        try:
             from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
        except ImportError:
             # If `trust_remote_code` is used, the class is dynamic. The function might not be easily importable.
             # However, typically the forward of attention calls it.
             # We can copy the apply function logic or assume the one we imported is correct if the library is installed.
             # Given the `Makefile` downloads weights, but the code relies on `transformers.models.gpt_oss`,
             # it implies `transformers` has this model or it's installed in the venv from source/plugin.
             pass

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

    # Save Data
    tensors = {
        "position_ids": position_ids.contiguous(),
        "cos": cos.contiguous(),
        "sin": sin.contiguous(),
        "q": q.contiguous(),
        "k": k.contiguous(),
        "q_out": q_out.contiguous(),
        "k_out": k_out.contiguous(),
    }
    
    output_file = "gpt_oss_rotary_embedding_20b.safetensors"
    save_file(tensors, output_file)
    print(f"Saved {output_file}")
    
    # Verification print
    print("Shapes:")
    print(f"q: {q.shape}")
    print(f"cos: {cos.shape}")
    print(f"q_out: {q_out.shape}")

if __name__ == "__main__":
    main()
