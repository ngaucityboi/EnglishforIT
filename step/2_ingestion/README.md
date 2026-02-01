# 📥 GIAI ĐOẠN 2: INGESTION PIPELINE

## 📌 Tổng quan

Hệ thống ingestion chuyển đổi dữ liệu văn bản luật đã làm sạch thành vector database FAISS để phục vụ tìm kiếm ngữ nghĩa trong RAG system.

## 🎯 Mục tiêu

- Đọc dữ liệu từ 4 file JSON đã được làm sạch (212 điều luật)
- Chuyển đổi thành LangChain Document objects với metadata đầy đủ
- Tạo vector embeddings bằng model đa ngôn ngữ (384 dimensions)
- Lưu trữ vào FAISS index để retrieval nhanh (<100ms)
- Đảm bảo metadata chính xác 100% (đã fix duplicate IDs và citations)

## 📂 Cấu trúc thư mục

```
2_ingestion/
├── ingestion_pipeline.py      # Pipeline chính - xử lý từ JSON → FAISS
├── demo_retrieval.py          # Script demo tìm kiếm semantic search
├── README.md                  # Tài liệu này
└── output/                    # Output directory (tự động tạo)
    ├── law_documents_index_config.json  # Metadata cấu hình (317 bytes)
    └── law_documents_index/
        ├── index.faiss        # FAISS vector index (325 KB)
        └── index.pkl          # Document metadata & docstore (441 KB)
```

**Lưu ý**: Dependencies được quản lý tập trung tại [requirements.txt](../../requirements.txt) ở thư mục gốc.

## 🚀 Hướng dẫn sử dụng

### Bước 1: Cài đặt dependencies

```bash
cd F:\3.Laptrinh\EnglishforIT
pip install -r requirements.txt
```

Packages cần thiết (xem [requirements.txt](../../requirements.txt)):
- `langchain` + `langchain-community` + `langchain-huggingface`: Framework RAG
- `sentence-transformers`: Tạo embeddings
- `faiss-cpu`: Vector database
- `rank-bm25`: BM25 retrieval
- `numpy`: Xử lý mảng

### Bước 2: Chạy ingestion pipeline

```bash
cd F:\3.Laptrinh\EnglishforIT\step\2_ingestion
python ingestion_pipeline.py
```

Pipeline sẽ:
1. Load 4 JSON files từ `../../data/input/`
2. Tạo 212 LangChain Documents
3. Generate embeddings (384-dim vectors)
4. Build FAISS index
5. Save to `output/law_documents_index/`
6. Auto-test với query mẫu

**Thời gian chạy**: ~30-60 giây (CPU), ~10-15 giây (GPU)

### Bước 3: Test retrieval

```bash
python demo_retrieval.py
```

Demo bao gồm:
- 5 test queries có sẵn
- Interactive mode để test query tùy ý
- Hiển thị kết quả với score và metadata

### Bước 4: Load index trong code

```bash
python
```

Trong Python shell hoặc script:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.load_local(
    "output/law_documents_index",
    embeddings,
    allow_dangerous_deserialization=True
)

results = vectorstore.similarity_search("Quy định về bảo vệ đê điều", k=5)
```

## 🔧 Chi tiết kỹ thuật

### Embedding Model

**Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Thông số**:
- Vector dimension: 384
- Model size: ~471 MB (download lần đầu)
- Support: 50+ ngôn ngữ (bao gồm tiếng Việt)
- Tốc độ: ~1000 docs/phút (CPU), ~5000 docs/phút (GPU)

**Ưu điểm**:
- Lightweight nhưng accurate cho semantic search
- Cân bằng tốt giữa tốc độ và chất lượng
- Tối ưu cho multilingual content

**Lưu ý**: Lần chạy đầu model sẽ tự động download từ HuggingFace (~471 MB)

### Document Structure

Mỗi LangChain Document bao gồm:

**page_content**: Nội dung từ field `content_for_embedding` trong JSON

**metadata** (9 fields):
- `id`: Unique ID cho mỗi điều (VD: "VBHN_05_2020_C1_D1")
- `doc_id`: ID văn bản (VD: "05/VBHN-VPQH")
- `doc_name`: Tên văn bản (VD: "Văn bản hợp nhất Luật Đê điều")
- `chapter_no`: Số chương (VD: "I", "II")
- `chapter_name`: Tên chương
- `article_no`: Số điều (VD: "1", "2")
- `article_name`: Tên điều
- `type`: Loại văn bản (VD: "phap_quy")
- `citation`: Trích dẫn đầy đủ (VD: "Điều 1, Luật Đê điều (VBHN 05/VBHN-VPQH)")

**Chú ý**: Metadata khớp 100% với JSON source, không thêm/bớt fields

### FAISS Configuration

**Index type**: IndexFlatL2 (exact nearest neighbor search)

**Đặc điểm**:
- Similarity metric: L2 distance
- Exact search (không approximate)
- Tốt cho dataset nhỏ-trung (<1M vectors)
- Query time: <100ms

**Storage**:
- `index.faiss`: Binary FAISS index (325 KB)
- `index.pkl`: Docstore + metadata mapping (441 KB)

## 📊 Dữ liệu & Performance

### Input Data (data/input/)

| File | Records | Tỷ lệ | Doc ID | Status |
|------|---------|-------|--------|--------|
| luatdedieu.json | 48 | 22.6% | VBHN_05_2020 | ✅ Clean |
| luatkhituongthuyvan.json | 57 | 26.9% | VBHN_06_2020 | ✅ Fixed (ID + citations) |
| luatphongchongthientai.json | 47 | 22.2% | VBHN_04_2020 | ✅ Clean |
| luatthuyloi.json | 60 | 28.3% | VBHN_05_2020 (khác với Đê điều) | ✅ Clean |
| **TOTAL** | **212** | **100%** | | ✅ All unique IDs |

### Performance Metrics

| Metric | Value | Environment |
|--------|-------|-------------|
| Embedding time | ~30-60s | Intel CPU |
| Embedding time | ~10-15s | NVIDIA GPU |
| Index build time | <5s | Any |
| Index size | ~766 KB | 212 vectors |
| Vector dimension | 384 | Fixed |
| Query time | <100ms | Average |
| Memory usage | ~2 GB | Runtime |

### Validation Results

✅ **Data Quality**: 100/100 score
- Tất cả 212 IDs unique (đã fix 20 duplicates)
- Citations chính xác 100% (đã fix 57 citations)
- Metadata structure khớp hoàn toàn với JSON
- Không có missing/null values

✅ **Search Quality**: Tested với query "Quy định về bảo vệ đê điều"
- Top results liên quan chính xác
- Metadata đầy đủ và đúng
- Citations trỏ đúng nguồn

## 📋 Data Cleaning History

### Issues đã fix

1. **Duplicate IDs** (20 records):
   - Vấn đề: luatkhituongthuyvan.json và luatthuyloi.json cùng dùng doc_id `VBHN_05_2020`
   - Fix: Đổi doc_id → `VBHN_06_2020` cho luatkhituongthuyvan.json
   - File backup: `luatkhituongthuyvan.json.backup`

2. **Wrong Citations** (57 records):
   - Vấn đề: Citations trong luatkhituongthuyvan.json hiển thị "Luật Thủy lợi"
   - Fix: Sửa thành "Luật Khí tượng thủy văn (VBHN 06/VBHN-VPQH)"
   - Script: `fix_citations.py` (đã xóa sau khi hoàn thành)

3. **Metadata Structure**:
   - Vấn đề: Code ban đầu thêm field `clause_no: None` không có trong JSON
   - Fix: Remove field khỏi metadata creation logic

## ⚙️ Configuration Options

### Tùy chỉnh trong ingestion_pipeline.py

```python
# Embedding device
EMBEDDING_DEVICE = "cpu"  # Đổi thành "cuda" nếu có GPU

# Batch size
BATCH_SIZE = 32  # Tăng nếu có RAM nhiều

# Model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Alternative Models

Có thể thay đổi model tùy mục đích:

**Tối ưu tiếng Việt**:
- `keepitreal/vietnamese-sbert`
- `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`

**Chất lượng cao hơn** (trade-off: chậm hơn, nặng hơn):
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim)

**Nhanh hơn** (trade-off: kém accurate hơn):
- `sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2` (384-dim, nhẹ hơn)

## 🔄 Workflow Integration

### Input (từ Giai đoạn 1)
- Đọc từ: `../../data/input/*.json`
- Format: Cleaned JSON với fields chuẩn
- Validation: Đã qua data cleaning pipeline

### Output (cho Giai đoạn 3)
- FAISS index tại: `output/law_documents_index/`
- Load method: `FAISS.load_local()`
- Usage: Retrieval trong RAG pipeline

### Next Steps (Giai đoạn 3 - RAG)
1. Load FAISS index
2. Integrate với LLM (GPT/Claude/Gemini)
3. Implement retrieval + generation
4. Add citation tracking
5. Build chatbot interface

## ⚠️ Lưu ý quan trọng

### Lần chạy đầu tiên
- Model tự động download (~471 MB)
- Cần kết nối internet
- Thời gian: ~5-10 phút (tùy tốc độ mạng)
- Cache tại: `~/.cache/huggingface/`

### System Requirements
- RAM: ≥ 2 GB khả dụng
- Disk: ~1 GB (model + index)
- CPU: Bất kỳ (khuyến nghị multi-core)
- GPU: Optional (tăng tốc ~5-10x)

### Security Warning
- File `index.pkl` chứa pickled objects
- Cần `allow_dangerous_deserialization=True` khi load
- Chỉ load index từ nguồn tin cậy

### Troubleshooting

**Lỗi import**: Cài lại packages
```bash
pip install --upgrade langchain langchain-community langchain-huggingface
```

**Out of memory**: Giảm BATCH_SIZE hoặc dùng CPU
```python
EMBEDDING_DEVICE = "cpu"
BATCH_SIZE = 16
```

**Model download chậm**: Dùng mirror HuggingFace hoặc download manual

## 📈 Scalability

### Hiện tại (212 docs)
- Index type: IndexFlatL2 (exact search)
- Query time: <100ms
- Phù hợp với dataset size hiện tại

### Mở rộng (>10K docs)
- Chuyển sang IndexIVFFlat (approximate search)
- Sử dụng GPU cho faster embedding
- Implement batch processing
- Consider distributed FAISS

### Production Deployment
- Cache embeddings để tránh re-compute
- Monitor query latency
- Set up index versioning
- Implement incremental updates

## 📚 Tài liệu tham khảo

- **LangChain**: https://python.langchain.com/docs/integrations/vectorstores/faiss
- **FAISS**: https://faiss.ai/
- **Sentence Transformers**: https://www.sbert.net/
- **HuggingFace Models**: https://huggingface.co/sentence-transformers

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026-01-31  
**Version**: 2.0  
**Data Quality**: 100/100  
**Total Vectors**: 212
