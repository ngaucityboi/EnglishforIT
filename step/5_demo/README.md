# 🎨 Stage 5: Web UI - Streamlit App

## 📌 Tổng quan

Giao diện web Streamlit cho Vietnamese Legal Assistant - hệ thống Q&A thông minh về luật.

## 🚀 Cách chạy

### 1️⃣ **Từ thư mục project**

```bash
python run_demo.py
```

Ứng dụng sẽ mở tự động ở `http://localhost:8501`

### 2️⃣ **Hoặc chạy trực tiếp Streamlit**

```bash
streamlit run step/5_demo/app.py
```

### 3️⃣ **Chạy với cổng khác** (nếu 8501 bị chiếm dụng)

```bash
streamlit run step/5_demo/app.py --server.port 8502
```

## 🎯 Tính năng

### ✨ Giao diện chính
- **Input câu hỏi**: Nhập câu hỏi tiếng Việt về luật
- **Nút tìm kiếm**: Nhấn để xử lý câu hỏi
- **Hiển thị câu trả lời**: Kết quả được tạo bởi LLM
- **Tài liệu liên quan**: Danh sách các điều luật được lấy ra
- **Lịch sử câu hỏi**: Theo dõi các câu hỏi vừa hỏi

### ⚙️ Sidebar Settings
- **Nhiệt độ LLM** (0.0 - 1.0)
  - 0.0 = Chính xác, tuân thủ tài liệu
  - 1.0 = Sáng tạo, linh hoạt hơn
  - Mặc định: 0.1 (rất chính xác)

- **Số lượng tài liệu**
  - 1-10 tài liệu
  - Mặc định: 5
  - Nhiều hơn = Context đầy đủ nhưng chậm hơn

### 💡 Gợi ý câu hỏi
- 4 ví dụ câu hỏi mặc định
- Click để sử dụng ngay

### 📊 Thông tin hệ thống
- Tổng số văn bản: 212 điều luật
- Công nghệ: RAG + Gemini AI
- Ngôn ngữ: Tiếng Việt

## 📋 Ví dụ câu hỏi

```
1. Luật Thủy lợi quy định gì về bảo vệ công trình nước?
2. Những trách nhiệm của chủ dự án trong phòng chống thiên tai?
3. Định nghĩa về khí tượng, thủy văn là gì?
4. Công trình đê điều phải đạt những tiêu chuẩn gì?
```

## 🔧 Yêu cầu

### Dependencies
- `streamlit >= 1.28.0`
- `langchain >= 0.1.0`
- `sentence-transformers`
- `google-generativeai`
- Tất cả dependencies từ `requirements.txt`

### Cài đặt

```bash
pip install -r ../../../requirements.txt
pip install streamlit
```

## 🌐 Truy cập

Khi chạy, truy cập ứng dụng tại:
```
http://localhost:8501
```

## 🛑 Dừng ứng dụng

- Nhấn **Ctrl+C** trong terminal
- Hoặc đóng trình duyệt

## 📝 Cấu trúc dự án

```
step/5_demo/
├── app.py          # Streamlit app chính
└── README.md       # File này
```

## 🎨 Styling

- **Gradient answers**: Màu tím gradient để dễ nhận diện
- **Source documents**: Hộp với border trái để phân biệt
- **Responsive design**: Tích hợp sẵn việc responsive của Streamlit
- **Dark mode friendly**: Tương thích cả light mode và dark mode

## ⚡ Performance

- Lần đầu chạy: 20-30 giây (tải model embeddings)
- Các lần sau: Dùng cache (nhanh hơn)
- Thời gian truy vấn: <5 giây/câu hỏi

## 🐛 Troubleshooting

### Lỗi "Connection refused"
```bash
# Kiểm tra cổng 8501 có bị chiếm không
netstat -ano | findstr :8501

# Chạy ở cổng khác
streamlit run step/5_demo/app.py --server.port 8502
```

### Lỗi API Key
```bash
# Kiểm tra file .env
cat .env

# Đảm bảo GOOGLE_API_KEY được set đúng
```

### Lỗi Cache/Model
```bash
# Xóa cache Streamlit
streamlit cache clear

# Hoặc xóa thủ công
rm -rf ~/.streamlit/cache
```

## 📞 Support

Chi tiết thêm tại:
- [step/4_generation/README.md](../4_generation/README.md) - RAG Chain
- [step/3_retrieval/README.md](../3_retrieval/README.md) - Hybrid Search
- README chính của project
