# 📚 RAG Legal QA System - EnglishforIT

Hệ thống Q&A thông minh về văn bản luật sử dụng RAG (Retrieval-Augmented Generation)

## 🎯 Tổng quan

Project xây dựng hệ thống hỏi đáp tự động về các văn bản luật Việt Nam, kết hợp:
- **Retrieval**: Tìm kiếm hybrid (BM25 + Dense Embedding)  
- **Generation**: LLM tổng hợp và trả lời (sẽ implement ở giai đoạn 4)

## 📂 Cấu trúc Project

```
EnglishforIT/
├── requirements.txt              # Dependencies chung cho toàn project
├── README.md                     # File này
│
├── data/
│   └── input/                    # Dữ liệu JSON đã làm sạch
│       ├── luatdedieu.json       # 48 điều
│       ├── luatkhituongthuyvan.json  # 57 điều  
│       ├── luatphongchongthientai.json  # 47 điều
│       └── luatthuyloi.json      # 60 điều
│
├── step/
│   ├── 2_ingestion/              # Giai đoạn 2: Vector Database
│   │   ├── ingestion_pipeline.py
│   │   ├── demo_retrieval.py
│   │   ├── README.md
│   │   └── output/
│   │       └── law_documents_index/  # FAISS index
│   │
│   └── 3_retrieval/              # Giai đoạn 3: Hybrid Search
│       ├── hybrid_retrieval.py
│       ├── demo_search.py
│       └── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/Nguyen15idhue/EnglishforIT.git
cd EnglishforIT
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Dependencies chính**:
- `langchain` + `langchain-community` + `langchain-huggingface`: RAG framework
- `sentence-transformers`: Multilingual embeddings
- `faiss-cpu`: Vector database
- `rank-bm25`: BM25 keyword search

## 📊 Dữ liệu

**Tổng cộng**: 212 điều luật từ 4 văn bản

| Văn bản | Số điều | Doc ID | Tỷ lệ |
|---------|---------|--------|-------|
| Luật Đê điều | 48 | VBHN_05_2020 | 22.6% |
| Luật Khí tượng thủy văn | 57 | VBHN_06_2020 | 26.9% |
| Luật Phòng chống thiên tai | 47 | VBHN_04_2020 | 22.2% |
| Luật Thủy lợi | 60 | VBHN_05_2020 | 28.3% |

**Data Quality**: 100/100
- ✅ Tất cả IDs unique
- ✅ Citations chính xác
- ✅ Metadata đầy đủ

## 🔄 Pipeline

### Giai đoạn 2: Ingestion (✅ Hoàn thành)

Chuyển đổi JSON → FAISS vector database

```bash
cd step/2_ingestion
python ingestion_pipeline.py
```

**Output**:
- 212 vector embeddings (384 dimensions)
- FAISS index (~766 KB)
- Query time: <100ms

Chi tiết: [step/2_ingestion/README.md](step/2_ingestion/README.md)

### Giai đoạn 3: Hybrid Retrieval (✅ Hoàn thành)

Kết hợp BM25 + Dense Embedding (50/50)

```bash
cd step/3_retrieval
python hybrid_retrieval.py          # Demo tự động
python demo_search.py               # Interactive search
```

**Features**:
- BM25: Keyword-based search
- Dense: Semantic search
- Hybrid: Kết hợp cả hai
- Top-K results với metadata đầy đủ

Chi tiết: [step/3_retrieval/README.md](step/3_retrieval/README.md)

### Giai đoạn 4: RAG Pipeline (🔜 Sắp triển khai)

Tích hợp LLM để generate câu trả lời

## 🧪 Test nhanh

### Test ingestion

```bash
cd step/2_ingestion
python demo_retrieval.py
```

### Test retrieval

```bash
cd step/3_retrieval
python demo_search.py "Quy định về bảo vệ đê điều"
```

## ⚙️ Configuration

### Model Embedding

**Hiện tại**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector dim: 384
- Size: ~471 MB
- Languages: 50+ (bao gồm tiếng Việt)

### Retrieval Weights

Trong [step/3_retrieval/hybrid_retrieval.py](step/3_retrieval/hybrid_retrieval.py):

```python
BM25_WEIGHT = 0.5    # 50% keyword matching
DENSE_WEIGHT = 0.5   # 50% semantic matching
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Total documents | 212 |
| Vector dimension | 384 |
| Index size | ~766 KB |
| Embedding time | ~30-60s (CPU) |
| Query time | <100ms |
| RAM usage | ~2-3 GB |

## 🛠️ Tech Stack

- **Python**: 3.13
- **LangChain**: RAG framework
- **FAISS**: Vector search engine
- **Sentence Transformers**: Multilingual embeddings
- **Rank-BM25**: Keyword search

## 📝 Changelog

### 2026-02-01
- ✅ Tạo requirements.txt chung cho toàn project
- ✅ Implement custom EnsembleRetriever (thay thế deprecated class)
- ✅ Hoàn thành Giai đoạn 3: Hybrid Retrieval
- ✅ Cập nhật documentation

### 2026-01-31
- ✅ Hoàn thành Giai đoạn 2: Ingestion Pipeline
- ✅ Fix duplicate IDs và citations
- ✅ Tạo FAISS index với 212 documents

## 🎓 Tài liệu tham khảo

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Rank-BM25](https://github.com/dorianbrown/rank_bm25)

## 📧 Contact

Repository: https://github.com/Nguyen15idhue/EnglishforIT

---

**Status**: 🟢 In Progress  
**Current Phase**: 3/4 (Retrieval completed)  
**Next**: RAG Pipeline with LLM
