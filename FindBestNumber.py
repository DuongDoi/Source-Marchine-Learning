import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Đọc dữ liệu
read = pd.read_excel('DataML.xlsx')

# Số lượng bộ dữ liệu bạn muốn thử
n_samples_to_try = 350

# Khởi tạo biến để lưu giá trị R-squared cao nhất và số lượng bộ dữ liệu tương ứng
best_r2 = -1
best_num_samples = 0

# Vòng lặp để thử nhiều số lượng bộ dữ liệu
for num_samples in range(10, n_samples_to_try + 1, 1):
    data = read.head(num_samples)
    
    X = data[['Tuoi', 'GioiTinh', 'KinhNghiemLaiXe', 'BeMatDuong', 'AnhSang', 'ThoiTiet', 'NguyenNhanTaiNan']]+1
    y = data['MucDoTaiNan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    # Kiểm tra xem giá trị R-squared có cao hơn giá trị tốt nhất hiện tại không
    if r2 > best_r2:
        best_r2 = r2
        best_num_samples = num_samples
print("Best num: ",best_num_samples)
# In ra giá trị R-squared cao nhất và số lượng bộ dữ liệu tương ứng
print(f"Best R-squared ({best_num_samples} samples): {best_r2}")
