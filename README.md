# naive_bayes

## Mô tả
- Tập dữ liệu bao gồm 5730 emails được lấy từ nguồn https://kaggle.com/balakishan77/spam-or-ham-email-classification đã được phân loại thành tập train và tập test theo tỷ lệ 8:2<br /> 
- Project được xây dựng dựa trên framework Django<br />
- Code có tham khảo từ https://github.com/henrydinh/Naive-Bayes-Text-Classification

## Cài đặt
- Cài đặt Django ```pip install django```
- Chạy lệnh ```python manage.py migrate``` để tạo bảng session (dùng lưu các biến session trong file views.py)
- Cài đặt các thư viện punkt, wordnet của nltk
- Cài đặt numpy, pandas,...
- Mở terminal trong thư mục project, chạy lệnh ``` python manage.py runserver ```
- Truy cập đường dẫn: http://127.0.0.1:8000/training
