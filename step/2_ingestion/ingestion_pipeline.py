"""
INGESTION PIPELINE - GIAI ĐOẠN 2
Đưa dữ liệu luật pháp vào Vector Database (FAISS)

Author: AI Engineer
Date: 2026-01-31
"""

import json
import os
import pickle
from typing import List, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================================
# CONFIGURATION
# ============================================================================

# Đường dẫn input/output
INPUT_DIR = Path("../../data/input")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Các file JSON cần xử lý
INPUT_FILES = [
    "luatdedieu.json",
    "luatkhituongthuyvan.json",
    "luatphongchongthientai.json",
    "luatthuyloi.json"
]

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"  # Đổi thành "cuda" nếu có GPU

# FAISS index configuration
FAISS_INDEX_NAME = "law_documents_index"


# ============================================================================
# STEP 1: ĐỌC DỮ LIỆU TỪ JSON FILES
# ============================================================================

def load_json_data(file_path: Path) -> List[Dict]:
    """
    Đọc dữ liệu từ file JSON
    
    Args:
        file_path: Đường dẫn tới file JSON
        
    Returns:
        List các dictionary chứa dữ liệu văn bản luật
    """
    print(f"📖 Đang đọc file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   ✓ Đã đọc {len(data)} records")
    return data


# ============================================================================
# STEP 2: CHUYỂN ĐỔI THÀNH LANGCHAIN DOCUMENTS
# ============================================================================

def create_documents(json_data: List[Dict]) -> List[Document]:
    """
    Chuyển đổi dữ liệu JSON thành LangChain Document objects
    
    Mỗi Document có:
    - page_content: Nội dung văn bản luật (từ field content_for_embedding)
    - metadata: Giữ nguyên toàn bộ metadata từ JSON (id, doc_id, doc_name, 
                chapter_no, chapter_name, article_no, article_name, type, citation)
    
    Args:
        json_data: List các dictionary từ JSON
        
    Returns:
        List các LangChain Document objects
    """
    print(f"\n🔄 Đang chuyển đổi {len(json_data)} records thành Documents...")
    
    documents = []
    
    for idx, record in enumerate(json_data):
        # Tạo metadata - giữ nguyên như trong JSON, không thêm field mới
        metadata = {
            "id": record["id"],
            "doc_id": record["metadata"]["doc_id"],
            "doc_name": record["metadata"]["doc_name"],
            "chapter_no": record["metadata"]["chapter_no"],
            "chapter_name": record["metadata"]["chapter_name"],
            "article_no": record["metadata"]["article_no"],
            "article_name": record["metadata"]["article_name"],
            "type": record["metadata"]["type"],
            "citation": record["citation"]
        }
        
        # Tạo Document object
        doc = Document(
            page_content=record["content_for_embedding"],
            metadata=metadata
        )
        
        documents.append(doc)
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"   Đã xử lý {idx + 1}/{len(json_data)} documents...")
    
    print(f"   ✓ Hoàn thành! Tổng cộng {len(documents)} Documents")
    return documents


# ============================================================================
# STEP 3: TẠO EMBEDDINGS VÀ LƯU VÀO FAISS
# ============================================================================

def create_vector_store(documents: List[Document]) -> FAISS:
    """
    Tạo embeddings cho documents và lưu vào FAISS vector store
    
    Args:
        documents: List các LangChain Document objects
        
    Returns:
        FAISS vector store đã được tạo
    """
    print(f"\n🧠 Đang khởi tạo Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"   (Model sẽ được tải về lần đầu tiên, có thể mất vài phút...)")
    
    # Khởi tạo embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}  # Chuẩn hóa để tính cosine similarity
    )
    
    print(f"   ✓ Model đã sẵn sàng!")
    
    print(f"\n🔢 Đang tạo embeddings cho {len(documents)} documents...")
    print(f"   (Quá trình này có thể mất vài phút...)")
    
    # Tạo FAISS vector store từ documents
    # FAISS sẽ tự động:
    # 1. Tạo embeddings cho tất cả documents
    # 2. Xây dựng index để tìm kiếm nhanh
    # 3. Lưu trữ metadata kèm theo
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    print(f"   ✓ Hoàn thành! Vector store đã được tạo")
    print(f"   📊 Số lượng vectors: {vectorstore.index.ntotal}")
    print(f"   📐 Vector dimension: {vectorstore.index.d}")
    
    return vectorstore


# ============================================================================
# STEP 4: LƯU FAISS INDEX VÀ METADATA
# ============================================================================

def save_vector_store(vectorstore: FAISS, base_name: str):
    """
    Lưu FAISS vector store ra file để sử dụng lại
    
    Args:
        vectorstore: FAISS vector store cần lưu
        base_name: Tên cơ sở cho các file output
    """
    print(f"\n💾 Đang lưu FAISS index vào {OUTPUT_DIR}")
    
    # Đường dẫn lưu index
    index_path = OUTPUT_DIR / base_name
    
    # Lưu FAISS index (bao gồm vectors và metadata)
    vectorstore.save_local(str(index_path))
    
    print(f"   ✓ Đã lưu index tại: {index_path}")
    print(f"   📁 Files được tạo:")
    print(f"      - index.faiss: FAISS vector index")
    print(f"      - index.pkl: Document metadata và docstore")
    
    # Lưu thông tin cấu hình để reference
    config = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "total_documents": vectorstore.index.ntotal,
        "vector_dimension": vectorstore.index.d,
        "input_files": INPUT_FILES,
        "created_at": "2026-01-31"
    }
    
    config_path = OUTPUT_DIR / f"{base_name}_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"      - {base_name}_config.json: Thông tin cấu hình")
    

# ============================================================================
# STEP 5: TEST LOAD VÀ RETRIEVAL
# ============================================================================

def test_vector_store(base_name: str):
    """
    Test khả năng load lại vector store và thực hiện retrieval
    
    Args:
        base_name: Tên cơ sở của index đã lưu
    """
    print(f"\n🧪 Test load vector store và retrieval...")
    
    # Khởi tạo lại embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load FAISS index từ disk
    index_path = OUTPUT_DIR / base_name
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True  # Cần thiết để load pickle
    )
    
    print(f"   ✓ Đã load thành công vector store")
    print(f"   📊 Số vectors: {vectorstore.index.ntotal}")
    
    # Test query
    test_query = "Quy định về bảo vệ đê điều"
    print(f"\n   🔍 Test query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"   📋 Top 3 kết quả tìm kiếm:")
    for i, doc in enumerate(results, 1):
        print(f"\n   [{i}] {doc.metadata['citation']}")
        print(f"       {doc.page_content[:150]}...")
    
    print(f"\n   ✅ Test thành công!")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main function - Chạy toàn bộ ingestion pipeline
    """
    print("=" * 80)
    print("INGESTION PIPELINE - GIAI ĐOẠN 2")
    print("Đưa dữ liệu luật pháp vào Vector Database")
    print("=" * 80)
    
    # Bước 1: Đọc tất cả JSON files
    all_data = []
    for filename in INPUT_FILES:
        file_path = INPUT_DIR / filename
        if file_path.exists():
            data = load_json_data(file_path)
            all_data.extend(data)
        else:
            print(f"   ⚠️ Warning: File không tồn tại: {file_path}")
    
    print(f"\n📊 Tổng cộng: {len(all_data)} records từ {len(INPUT_FILES)} files")
    
    # Bước 2: Chuyển đổi thành Documents
    documents = create_documents(all_data)
    
    # Bước 3: Tạo embeddings và vector store
    vectorstore = create_vector_store(documents)
    
    # Bước 4: Lưu vector store
    save_vector_store(vectorstore, FAISS_INDEX_NAME)
    
    # Bước 5: Test load và retrieval
    test_vector_store(FAISS_INDEX_NAME)
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH INGESTION PIPELINE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"📦 FAISS index: {FAISS_INDEX_NAME}")
    print(f"📊 Total vectors: {len(documents)}")
    print(f"\n🎯 Bạn có thể sử dụng index này cho Retrieval trong bước tiếp theo!")


if __name__ == "__main__":
    main()
