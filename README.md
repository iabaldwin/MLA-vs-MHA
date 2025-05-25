# MLA vs MHA: Real-Time Comparison

A beautiful web interface that compares Multi-Head Latent Attention (MLA) with traditional Multi-Head Attention (MHA) using **real transformer operations** and **actual tokenization**.

## 🚀 Quick Start

### 1. Start the Backend
```bash
python start_backend.py
```
This will:
- Install required dependencies
- Start the FastAPI backend on `http://localhost:8000`
- Enable real transformer operations

### 2. Open the Frontend
Open `index.html` in your browser. The interface will connect to the backend automatically.

## ✨ Features

### Real Transformer Operations
- **Actual tokenization** using HuggingFace transformers (GPT-2 tokenizer)
- **Real tensor operations** from your `mla/transformer.py` 
- **Performance timing** comparison between MLA and MHA
- **Memory usage** analysis with actual parameter counts

### Interactive Visualization
- **Live text tokenization**: Type "the cat sat on the mat" and see real tokens
- **Dynamic dimensions**: All tensor shapes update based on actual operations
- **Side-by-side comparison**: Watch both architectures process the same input
- **Low-rank factorization**: Visual explanation of compression benefits

### Key Insights You'll See
1. **Tokenization**: Real subword tokens (e.g., "cat" → "cat", "sitting" → "sit", "ting")
2. **Compression**: Actual 75% parameter reduction in MLA vs MHA
3. **Performance**: Real timing differences between architectures
4. **Memory**: Precise parameter counts from your model implementations

## 🧪 Try These Examples

### Simple Text
```
"the cat sat on the mat"
```
See how basic words are tokenized and processed.

### Complex Text  
```
"Transformers revolutionized natural language processing"
```
Watch subword tokenization and longer sequences.

### Technical Text
```
"Multi-head latent attention compresses embeddings efficiently"
```
See how technical vocabulary gets tokenized.

## 📊 What You'll Learn

### 1. **Real Tokenization**
- How text becomes tokens (not just word splitting!)
- Subword tokenization with byte-pair encoding
- Padding and truncation handling

### 2. **Actual Tensor Shapes**
- Exact dimensions flowing through each layer
- Memory usage comparisons with real numbers
- Parameter count differences

### 3. **Performance Differences**
- Actual timing measurements (milliseconds)
- Memory efficiency gains
- Computational complexity differences

### 4. **Low-Rank Mathematics**
- Visual matrix factorization: `(1024,1024)` → `(1024,128) × (128,1024)`
- Parameter reduction percentages
- Rank constraint implications

## 🏗️ Architecture

```
Frontend (index.html)
    ↓ HTTP requests
Backend (FastAPI)
    ↓ imports
Your Transformer Code (mla/transformer.py)
    ↓ uses
PyTorch + HuggingFace Transformers
```

## 🔧 API Endpoints

- `POST /tokenize` - Real tokenization with attention masks
- `POST /compare_attention` - Run MLA vs MHA comparison
- `GET /health` - Check backend status

## 🎯 Key Differences Highlighted

| Aspect | MHA | MLA |
|--------|-----|-----|
| **Step 1** | Direct QKV projections | Latent compression first |
| **Parameters** | ~3M (full rank) | ~750K (low rank) |
| **Memory** | Full embeddings | Compressed representations |
| **Position** | RoPE on full tensors | NoPE/RoPE split |

## 🐛 Troubleshooting

### Backend Issues
- **Port 8000 in use**: Change port in `backend.py`
- **Missing dependencies**: Run `pip install -r requirements-backend.txt`
- **Import errors**: Ensure `mla/transformer.py` exists

### Frontend Issues  
- **CORS errors**: Backend must be running first
- **Fetch failures**: Check console for backend connection issues

## 🔮 What Makes This Special

Unlike static visualizations, this shows **real computational differences**:
- Actual PyTorch operations from your code
- Real tokenizer behavior (not word splitting)
- Measurable performance differences
- True memory usage patterns

You can see why MLA matters: **real compression**, **real speedups**, **real memory savings**!