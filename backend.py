from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import time
import traceback

# Import your transformer modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mla.transformer import (
    ModelArgs, 
    AttentionType, 
    MultiLatentAttentionNaive, 
    MultiHeadAttention,
    MultiLatentAttentionKVCache,
    MultiHeadAttentionKVCache
)

app = FastAPI(title="MLA vs MHA Backend", description="Real transformer operations comparison")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tokenizer
tokenizer = None

class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = 16

class ModelConfig(BaseModel):
    dim: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 8
    q_compressed_dim: int = 128
    kv_compressed_dim: int = 128
    max_length: int = 16

class TransformerResponse(BaseModel):
    tokens: List[str]
    token_ids: List[int]
    embeddings_shape: List[int]
    # Actual tensor data for visualization
    input_embeddings: List[List[float]]  # [seq_len, dim] 
    mla_operations: Dict[str, Any]
    mha_operations: Dict[str, Any]
    timing_comparison: Dict[str, float]
    memory_comparison: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global tokenizer
    try:
        # Use a lightweight tokenizer for demo
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        # Fallback to simple tokenization
        tokenizer = None

def simple_tokenize(text: str) -> tuple[List[str], List[int]]:
    """Fallback tokenization if HuggingFace tokenizer fails"""
    tokens = text.lower().split()
    # Simple vocabulary mapping
    vocab = {token: i for i, token in enumerate(set(tokens))}
    token_ids = [vocab[token] for token in tokens]
    return tokens, token_ids

@app.post("/tokenize", response_model=Dict[str, Any])
async def tokenize_text(input_data: TextInput):
    """Tokenize input text and return tokens with metadata"""
    try:
        if tokenizer:
            # Use HuggingFace tokenizer
            encoded = tokenizer(
                input_data.text, 
                max_length=input_data.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
            token_ids = encoded.input_ids[0].tolist()
            attention_mask = encoded.attention_mask[0].tolist()
        else:
            # Fallback tokenization
            tokens, token_ids = simple_tokenize(input_data.text)
            # Pad or truncate
            if len(tokens) < input_data.max_length:
                tokens.extend(['<pad>'] * (input_data.max_length - len(tokens)))
                token_ids.extend([0] * (input_data.max_length - len(token_ids)))
            else:
                tokens = tokens[:input_data.max_length]
                token_ids = token_ids[:input_data.max_length]
            attention_mask = [1 if token != '<pad>' else 0 for token in tokens]
        
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "actual_length": sum(attention_mask),
            "padded_length": len(tokens)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

class ComparisonRequest(BaseModel):
    text: str
    max_length: Optional[int] = 16
    config: ModelConfig

@app.post("/compare_attention", response_model=TransformerResponse)
async def compare_attention_mechanisms(request: ComparisonRequest):
    """Compare MLA vs MHA on actual input"""
    try:
        # Create text input for tokenization
        text_input = TextInput(text=request.text, max_length=request.max_length)
        
        # Tokenize input
        tokenize_result = await tokenize_text(text_input)
        tokens = tokenize_result["tokens"]
        token_ids = torch.tensor([tokenize_result["token_ids"]])
        attention_mask = torch.tensor([tokenize_result["attention_mask"]], dtype=torch.float32)
        
        # Create model args
        config = request.config
        args = ModelArgs(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            q_compressed_dim=config.q_compressed_dim,
            kv_compressed_dim=config.kv_compressed_dim,
            max_position_embeddings=config.max_length,
            attention_type=AttentionType.MULTI_LATENT_ATTENTION_NAIVE
        )
        
        # Create embeddings (random for demo)
        batch_size, seq_len = token_ids.shape
        embeddings = torch.randn(batch_size, seq_len, args.dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        
        # Convert 2D attention mask to 4D format expected by attention layers
        # Create causal mask and apply padding mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
        # Expand attention mask to [batch, 1, seq_len, seq_len]
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0).unsqueeze(0)
        # Convert to additive mask (0 for attend, -inf for mask)
        attention_mask_4d = torch.where(attention_mask_4d == 0, float('-inf'), 0.0)
        
        # Initialize attention mechanisms
        mla_attention = MultiLatentAttentionNaive(args, layer_idx=0)
        mha_attention = MultiHeadAttention(args, layer_idx=0)
        
        # MLA Forward Pass with timing and memory tracking
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mla_start_time = time.time()
        
        with torch.no_grad():
            # Track MLA operations step by step with actual data
            mla_ops = {}
            
            # Step 1: Q Compression
            compressed_q = mla_attention.q_norm(mla_attention.w_dq(embeddings))
            mla_ops["q_compression"] = {
                "input_shape": list(embeddings.shape),
                "output_shape": list(compressed_q.shape),
                "params": args.dim * args.q_compressed_dim,
                "data": compressed_q.squeeze(0).tolist()[:seq_len]  # [seq_len, compressed_dim]
            }
            
            # Step 2: KV Compression  
            compressed_kv = mla_attention.kv_norm(mla_attention.w_dkv(embeddings))
            mla_ops["kv_compression"] = {
                "input_shape": list(embeddings.shape),
                "output_shape": list(compressed_kv.shape),
                "params": args.dim * args.kv_compressed_dim,
                "data": compressed_kv.squeeze(0).tolist()[:seq_len]  # [seq_len, compressed_dim]
            }
            
            # Step 3: Q Projections
            q_nope = mla_attention.w_uq(compressed_q)
            q_rope = mla_attention.w_qr(compressed_q)
            mla_ops["q_projections"] = {
                "nope_shape": list(q_nope.shape),
                "rope_shape": list(q_rope.shape),
                "total_params": (args.q_compressed_dim * args.n_heads * 96) + (args.q_compressed_dim * args.n_heads * 32),
                "q_nope_data": q_nope.squeeze(0).tolist()[:seq_len],  # [seq_len, n_heads * 96]
                "q_rope_data": q_rope.squeeze(0).tolist()[:seq_len]   # [seq_len, n_heads * 32]
            }
            
            # Full forward pass
            mla_output = mla_attention(embeddings, position_ids, attention_mask_4d)
            mla_ops["output"] = {
                "shape": list(mla_output.shape),
                "data": mla_output.squeeze(0).tolist()[:seq_len]  # [seq_len, dim]
            }
            
        mla_time = time.time() - mla_start_time
        
        # MHA Forward Pass with timing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mha_start_time = time.time()
        
        with torch.no_grad():
            # Track MHA operations with actual data
            mha_ops = {}
            
            # Direct QKV projections
            q = mha_attention.w_q(embeddings)
            k = mha_attention.w_k(embeddings) 
            v = mha_attention.w_v(embeddings)
            
            mha_ops["direct_projections"] = {
                "q_shape": list(q.shape),
                "k_shape": list(k.shape), 
                "v_shape": list(v.shape),
                "q_params": args.dim * args.n_heads * args.head_dim,
                "k_params": args.dim * args.n_kv_heads * args.head_dim,
                "v_params": args.dim * args.n_kv_heads * args.head_dim,
                "q_data": q.squeeze(0).tolist()[:seq_len],  # [seq_len, n_heads * head_dim]
                "k_data": k.squeeze(0).tolist()[:seq_len],  # [seq_len, n_kv_heads * head_dim]  
                "v_data": v.squeeze(0).tolist()[:seq_len]   # [seq_len, n_kv_heads * head_dim]
            }
            
            # Full forward pass
            mha_output = mha_attention(embeddings, position_ids, attention_mask_4d)
            mha_ops["output"] = {
                "shape": list(mha_output.shape),
                "data": mha_output.squeeze(0).tolist()[:seq_len]  # [seq_len, dim]
            }
            
        mha_time = time.time() - mha_start_time
        
        # Memory comparison (parameter count)
        mla_params = (
            args.dim * args.q_compressed_dim +  # w_dq
            args.dim * args.kv_compressed_dim + # w_dkv
            args.q_compressed_dim * args.n_heads * (96 + 32) + # w_uq + w_qr
            args.kv_compressed_dim * args.n_kv_heads * 64 + # w_uk
            args.kv_compressed_dim * args.n_kv_heads * 256 + # w_uv
            args.dim + # w_kr (special case)
            args.n_heads * 256 * args.dim # w_o
        )
        
        mha_params = (
            args.dim * args.n_heads * args.head_dim + # w_q
            args.dim * args.n_kv_heads * args.head_dim + # w_k  
            args.dim * args.n_kv_heads * args.head_dim + # w_v
            args.n_heads * args.head_dim * args.dim # w_o
        )
        
        return TransformerResponse(
            tokens=tokens,
            token_ids=tokenize_result["token_ids"],
            embeddings_shape=list(embeddings.shape),
            input_embeddings=embeddings.squeeze(0).tolist()[:seq_len],  # [seq_len, dim]
            mla_operations=mla_ops,
            mha_operations=mha_ops,
            timing_comparison={
                "mla_time_ms": mla_time * 1000,
                "mha_time_ms": mha_time * 1000,
                "speedup": mha_time / mla_time if mla_time > 0 else 1.0
            },
            memory_comparison={
                "mla_parameters": int(mla_params),
                "mha_parameters": int(mha_params), 
                "memory_reduction": int(mha_params - mla_params),
                "reduction_percentage": float((mha_params - mla_params) / mha_params * 100)
            }
        )
        
    except Exception as e:
        print(f"Error in compare_attention: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Attention comparison failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tokenizer_loaded": tokenizer is not None,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)