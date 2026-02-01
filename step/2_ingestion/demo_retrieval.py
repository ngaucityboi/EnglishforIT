"""
DEMO SCRIPT - Test FAISS Vector Store
Kiểm tra khả năng retrieval của hệ thống
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Đường dẫn tuyệt đối từ file hiện tại
SCRIPT_DIR = Path(__file__).parent
INDEX_PATH = SCRIPT_DIR / "output" / "law_documents_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Test queries
TEST_QUERIES = [
    "Quy định về bảo vệ đê điều",
    "Trách nhiệm của Nhà nước trong phòng chống thiên tai",
    "Nguyên tắc hoạt động trong lĩnh vực thủy lợi",
    "Các hành vi bị nghiêm cấm",
    "Chính sách đầu tư xây dựng công trình"
]

# ============================================================================
# LOAD VECTOR STORE
# ============================================================================

print("=" * 80)
print("DEMO: FAISS VECTOR STORE RETRIEVAL")
print("=" * 80)

print("\n📦 Đang load FAISS index...")

# Khởi tạo embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load vector store
vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"✅ Đã load thành công!")
print(f"📊 Tổng số vectors: {vectorstore.index.ntotal}")
print(f"📐 Vector dimension: {vectorstore.index.d}")

# ============================================================================
# TEST QUERIES
# ============================================================================

print("\n" + "=" * 80)
print("🔍 TEST RETRIEVAL")
print("=" * 80)

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n{'─' * 80}")
    print(f"Query {i}: {query}")
    print('─' * 80)
    
    # Tìm kiếm top 3 kết quả
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    for j, (doc, score) in enumerate(results, 1):
        print(f"\n[{j}] Score: {score:.4f}")
        print(f"📄 {doc.metadata['citation']}")
        print(f"📖 Văn bản: {doc.metadata['doc_name']}")
        print(f"📑 Chương {doc.metadata['chapter_no']}: {doc.metadata['chapter_name']}")
        print(f"💬 Nội dung:")
        print(f"   {doc.page_content[:200]}...")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "=" * 80)
print("💬 INTERACTIVE MODE")
print("=" * 80)
print("Nhập câu hỏi của bạn (hoặc 'quit' để thoát):\n")

while True:
    user_query = input("❓ Câu hỏi: ").strip()
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Tạm biệt!")
        break
    
    if not user_query:
        continue
    
    print(f"\n🔍 Đang tìm kiếm...\n")
    
    results = vectorstore.similarity_search_with_score(user_query, k=5)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"─ [{i}] Độ tương đồng: {1 - score:.2%} ─")
        print(f"📌 {doc.metadata['citation']}")
        print(f"📄 {doc.metadata['doc_name']}")
        print(f"💬 {doc.page_content[:150]}...")
        print()
    
    print("─" * 80 + "\n")

print("\n✅ Demo hoàn tất!")
