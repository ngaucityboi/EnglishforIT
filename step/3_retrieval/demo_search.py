"""
DEMO INTERACTIVE SEARCH
Cho phép người dùng nhập câu hỏi và test hybrid retrieval system
"""

import sys
from hybrid_retrieval import (
    load_faiss_vectorstore,
    create_bm25_retriever,
    create_dense_retriever,
    create_hybrid_retriever,
    format_results,
    TOP_K
)

def interactive_search():
    """Chế độ tìm kiếm interactive"""
    
    print("="*80)
    print("🔍 HYBRID SEARCH SYSTEM - Interactive Mode")
    print("="*80)
    print("\nĐang khởi tạo hệ thống...")
    
    # Khởi tạo retrievers
    vectorstore = load_faiss_vectorstore()
    bm25_retriever = create_bm25_retriever(vectorstore)
    dense_retriever = create_dense_retriever(vectorstore)
    hybrid_retriever = create_hybrid_retriever(bm25_retriever, dense_retriever)
    
    print("\n" + "="*80)
    print("✅ HỆ THỐNG ĐÃ SẴN SÀNG")
    print("="*80)
    print("\nHướng dẫn:")
    print("  - Nhập câu hỏi để tìm kiếm")
    print("  - Gõ 'exit' hoặc 'quit' để thoát")
    print("  - Gõ 'compare' để so sánh BM25 vs Dense vs Hybrid")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        query = input("📝 Nhập câu hỏi: ").strip()
        
        if not query:
            print("⚠️  Vui lòng nhập câu hỏi!")
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Cảm ơn bạn đã sử dụng! Tạm biệt!")
            break
        
        if query.lower() == 'compare':
            # So sánh 3 phương pháp
            test_query = input("📝 Nhập câu hỏi để so sánh: ").strip()
            if not test_query:
                continue
            
            print("\n" + "█"*80)
            print("SO SÁNH 3 PHƯƠNG PHÁP RETRIEVAL")
            print("█"*80)
            
            print("\n🔹 1. BM25 (Keyword Search):")
            bm25_results = bm25_retriever.invoke(test_query)
            format_results(bm25_results[:3], test_query)
            
            print("\n🔹 2. Dense Embedding (Semantic Search):")
            dense_results = dense_retriever.invoke(test_query)
            format_results(dense_results[:3], test_query)
            
            print("\n🔹 3. Hybrid (BM25 + Dense):")
            hybrid_results = hybrid_retriever.invoke(test_query)
            format_results(hybrid_results[:TOP_K], test_query)
            
        else:
            # Tìm kiếm bình thường với hybrid
            print("\n🔍 Đang tìm kiếm với Hybrid Retrieval...")
            results = hybrid_retriever.invoke(query)
            format_results(results[:TOP_K], query)


def quick_search(query):
    """Tìm kiếm nhanh với 1 query"""
    
    print(f"\n🔍 Quick Search: '{query}'")
    print("="*80)
    
    # Khởi tạo retrievers
    vectorstore = load_faiss_vectorstore()
    bm25_retriever = create_bm25_retriever(vectorstore)
    dense_retriever = create_dense_retriever(vectorstore)
    hybrid_retriever = create_hybrid_retriever(bm25_retriever, dense_retriever)
    
    # Tìm kiếm
    results = hybrid_retriever.invoke(query)
    format_results(results[:TOP_K], query)


if __name__ == "__main__":
    # Nếu có argument, dùng quick search
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        quick_search(query)
    else:
        # Không có argument, chạy interactive mode
        interactive_search()
