"""
GIAI ĐOẠN 3: HYBRID RETRIEVAL
Kết hợp BM25 (keyword search) và Dense Embedding (semantic search)
để tìm kiếm thông minh hơn trên dữ liệu văn bản luật
"""

import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field


# ==================== ENSEMBLE RETRIEVER IMPLEMENTATION ====================

class EnsembleRetriever(BaseRetriever):
    """
    Custom Ensemble Retriever kết hợp nhiều retrievers với weights
    Thay thế cho langchain.retrievers.EnsembleRetriever (không còn tồn tại trong version mới)
    """
    
    retrievers: List[BaseRetriever] = Field(description="List of retrievers to ensemble")
    weights: List[float] = Field(description="Weights for each retriever")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Lấy documents từ tất cả retrievers và merge theo weights"""
        
        # Lấy kết quả từ mỗi retriever
        all_results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            # Gán score dựa trên position và weight
            for i, doc in enumerate(docs):
                score = weight * (1.0 / (i + 1))  # Reciprocal rank
                all_results.append((doc, score))
        
        # Merge documents với cùng content
        doc_scores = {}
        for doc, score in all_results:
            doc_id = doc.page_content[:100]  # Dùng 100 ký tự đầu làm key
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score)
            else:
                doc_scores[doc_id] = (doc, score)
        
        # Sort theo score giảm dần
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        
        # Trả về documents (không có score)
        return [doc for doc, _ in sorted_docs]

# ==================== CẤU HÌNH ====================

# Đường dẫn FAISS index từ giai đoạn 2
FAISS_INDEX_PATH = "../2_ingestion/output/law_documents_index"

# Model embedding (phải giống giai đoạn 2)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"

# Tỷ lệ kết hợp retriever
BM25_WEIGHT = 0.5  # 50% BM25
DENSE_WEIGHT = 0.5  # 50% Dense embedding

# Số kết quả trả về
TOP_K = 5

# ==================== KHỞI TẠO RETRIEVERS ====================

def load_faiss_vectorstore():
    """Load FAISS vector store từ giai đoạn 2"""
    print("📂 Đang load FAISS index...")
    
    # Khởi tạo embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )
    
    # Load FAISS index
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ Đã load {vectorstore.index.ntotal} documents từ FAISS")
    return vectorstore


def create_bm25_retriever(vectorstore):
    """
    Tạo BM25 Retriever từ documents trong FAISS
    BM25 = keyword-based search (tìm theo từ khóa chính xác)
    """
    print("\n🔍 Đang tạo BM25 Retriever...")
    
    # Lấy tất cả documents từ FAISS docstore
    documents = list(vectorstore.docstore._dict.values())
    
    # Tạo BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = TOP_K
    
    print(f"✅ BM25 Retriever đã sẵn sàng với {len(documents)} documents")
    return bm25_retriever


def create_dense_retriever(vectorstore):
    """
    Tạo Dense Retriever từ FAISS vector store
    Dense = semantic search (tìm theo ý nghĩa)
    """
    print("\n🧠 Đang tạo Dense Retriever...")
    
    # Convert FAISS vectorstore thành retriever
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )
    
    print(f"✅ Dense Retriever đã sẵn sàng")
    return dense_retriever


def create_hybrid_retriever(bm25_retriever, dense_retriever):
    """
    Kết hợp BM25 và Dense retriever thành Ensemble Retriever
    Hybrid = BM25 + Dense để tận dụng ưu điểm của cả hai
    """
    print("\n🔗 Đang tạo Hybrid Retriever...")
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[BM25_WEIGHT, DENSE_WEIGHT]
    )
    
    print(f"✅ Hybrid Retriever đã sẵn sàng (BM25: {BM25_WEIGHT*100}%, Dense: {DENSE_WEIGHT*100}%)")
    return ensemble_retriever


# ==================== SEARCH FUNCTIONS ====================

def search_with_bm25(bm25_retriever, query):
    """Tìm kiếm chỉ với BM25 (keyword-based)"""
    results = bm25_retriever.invoke(query)
    return results


def search_with_dense(dense_retriever, query):
    """Tìm kiếm chỉ với Dense embedding (semantic)"""
    results = dense_retriever.invoke(query)
    return results


def search_with_hybrid(hybrid_retriever, query):
    """Tìm kiếm với Hybrid (kết hợp BM25 + Dense)"""
    results = hybrid_retriever.invoke(query)
    return results


def format_results(results, query):
    """Format và hiển thị kết quả tìm kiếm"""
    print(f"\n{'='*80}")
    print(f"🔎 Query: '{query}'")
    print(f"📊 Tìm thấy {len(results)} kết quả:")
    print(f"{'='*80}\n")
    
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('citation', 'N/A')}")
        print(f"    Văn bản: {doc.metadata.get('doc_name', 'N/A')}")
        print(f"    Chương {doc.metadata.get('chapter_no', 'N/A')}: {doc.metadata.get('chapter_name', 'N/A')}")
        print(f"    Điều {doc.metadata.get('article_no', 'N/A')}: {doc.metadata.get('article_name', 'N/A')}")
        print(f"    Nội dung: {doc.page_content[:200]}...")
        print()


# ==================== MAIN EXECUTION ====================

def main():
    """Demo hybrid retrieval với các test queries"""
    
    print("="*80)
    print("🚀 KHỞI ĐỘNG HYBRID RETRIEVAL SYSTEM")
    print("="*80)
    
    # 1. Load FAISS vector store
    vectorstore = load_faiss_vectorstore()
    
    # 2. Tạo BM25 retriever (keyword-based)
    bm25_retriever = create_bm25_retriever(vectorstore)
    
    # 3. Tạo Dense retriever (semantic)
    dense_retriever = create_dense_retriever(vectorstore)
    
    # 4. Tạo Hybrid retriever (kết hợp cả hai)
    hybrid_retriever = create_hybrid_retriever(bm25_retriever, dense_retriever)
    
    print("\n" + "="*80)
    print("✅ HỆ THỐNG ĐÃ SẴN SÀNG - BẮT ĐẦU TEST")
    print("="*80)
    
    # ==================== TEST QUERIES ====================
    
    test_queries = [
        "Quy định về bảo vệ đê điều",
        "Trách nhiệm của Ủy ban nhân dân",
        "Xử lý vi phạm pháp luật",
        "Dự báo thiên tai và cảnh báo",
        "Quản lý tài nguyên nước"
    ]
    
    for query in test_queries:
        print("\n" + "█"*80)
        print(f"TEST QUERY: {query}")
        print("█"*80)
        
        # So sánh 3 phương pháp
        print("\n🔹 1. BM25 ONLY (Keyword Search):")
        bm25_results = search_with_bm25(bm25_retriever, query)
        format_results(bm25_results[:3], query)
        
        print("\n🔹 2. DENSE ONLY (Semantic Search):")
        dense_results = search_with_dense(dense_retriever, query)
        format_results(dense_results[:3], query)
        
        print("\n🔹 3. HYBRID (BM25 + Dense):")
        hybrid_results = search_with_hybrid(hybrid_retriever, query)
        format_results(hybrid_results[:TOP_K], query)
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH DEMO")
    print("="*80)
    
    return hybrid_retriever, vectorstore


if __name__ == "__main__":
    hybrid_retriever, vectorstore = main()
