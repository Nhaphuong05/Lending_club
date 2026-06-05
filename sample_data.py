import pandas as pd
import numpy as np

file_goc = 'data.csv'
file_sample = 'data_sample.csv'

print("⏳ Đang trích xuất dữ liệu mẫu ngẫu nhiên...")

# Thay vì lấy 50,000 dòng đầu tiên, ta đọc khoảng 200,000 dòng đầu
# rồi lấy ngẫu nhiên 50,000 dòng trong số đó để tăng tính đại diện
df_chunk = pd.read_csv(file_goc, nrows=200000)
df_sample = df_chunk.sample(n=50000, random_state=42) # random_state giúp giữ cố định dữ liệu mỗi lần chạy

# 2. Lưu thành file mẫu mới
df_sample.to_csv(file_sample, index=False)
print("✅ Đã tạo xong file data_sample.csv chuẩn ngẫu nhiên!")