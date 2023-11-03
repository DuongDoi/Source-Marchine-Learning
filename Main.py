#Đã tối ưu
import pandas as pd #Thư viện pandas để xử lí dữ liệu
from sklearn.model_selection import train_test_split #Dùng để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.linear_model import LinearRegression #Nhập vào mô hình hồi quy tuyến tính
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #Nhập vào các hàm đánh giá mô hình
import matplotlib.pyplot as plt #Dùng để biểu diễn mô hình
import numpy as np
import tkinter as tk #Thư viện tkinter tạo giao diện người dùng

# Đọc dữ liệu
read = pd.read_excel('DataML.xlsx')
# Kiểm tra dữ liệu
data = read.head(86) # Chọn 86 bộ dữ liệu đầu tiên vì 86 có độ phù hợp nhất với mô hình
print(data)

# Tạo dữ liệu 
X = data[['Tuoi', 'GioiTinh', 'KinhNghiemLaiXe', 'BeMatDuong', 'AnhSang', 'ThoiTiet', 'NguyenNhanTaiNan']]
y = data['MucDoTaiNan']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)


#ĐÁNH GIÁ MÔ HÌNH
#Đo lường trung bình của sai số tuyệt đối giữa dự đoán và giá trị thực tế
mae = mean_absolute_error(y_test, y_pred)

#Đo lường trung bình của bình phương sai số giữa dự đoán và giá trị thực tế
mse = mean_squared_error(y_test, y_pred)

#Độ lệch chuẩn
rmse = np.sqrt(mse)

#Sự biến đổi trong biến mục tiêu
r2 = r2_score(y_test, y_pred)

# In ra các chỉ số
print("Linear Regression:")
#Hệ số (coefficients) của mô hình
print("Coefficients:", model.coef_)
#Sai số tuyệt đối trung bình (Mean Absolute Error - MAE)
print("Mean Absolute Error (MAE):", mae)
#Sai số bình phương trung bình (Mean Squared Error - MSE)
print("Mean Squared Error (MSE):", mse)
#Độ lệch chuẩn (Root Mean Squared Error - RMSE)
print("Root Mean Squared Error (RMSE):", rmse)
#Hệ số xác định (R-squared - R^2) - Đo lường mức độ mô hình phù hợp với dữ liệu
print("R-squared (R^2) Score:", r2)


#Dự đoán bộ dữ liệu mới
new_data = pd.DataFrame({
    'Tuoi': [1, 4, 2,3],  
    'GioiTinh': [1, 1, 1,0],
    'KinhNghiemLaiXe': [1, 0, 2,0],
    'BeMatDuong': [2, 1, 1,2],
    'AnhSang': [1, 3, 2,3],
    'ThoiTiet': [1, 2, 1,4],
    'NguyenNhanTaiNan': [1, 3, 1,2]
})

# Sử dụng mô hình đã huấn luyện để dự đoán
new_predictions = model.predict(new_data)

#Đưa ra thông số trên tập kiểm tra
print("\n------------------------------------------------")
print("-------------------SO LIEU TEST-------------------")
print("Thuc te         Du doan                    Do lech")
for i in range(len(y_test)):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    absolute_error = abs(actual - predicted)
    print(f"{actual}   {predicted}   {absolute_error}") 

# Đưa ra dự đoán bộ dữ liệu mới
print("\n*****************************\n*****DU DOAN BO DU LIEU MOI****")
print("-----------Note------------------")
print("Tuoi:  1. <18  2. 18-30  3. 31-50  4. >=51")
print("Gioi tinh:  0. Female  1. Male")
print("Kinh nghiem lai xe:  0. Khong co bang lai xe   1. Duoi 2 nam  2. Tren 2 nam")
print("Be mat duong:  1. Duong nhua  2. Duong dat")
print("Anh sang:  1. Troi sang  2. Troi toi va khong co anh sang  3. Troi toi va co anh sang")
print("Thoi tiet:  1. Binh thuong  2.Gio manh  3. Nhieu may  4. Mua  5. Tuyet")
print("Tinh trang lai xe:  1. Di au   2. Say ruou/bia  3. Di xe toc do cao\n")
print("--------------------------------------------------------------------------------")
for i, prediction in enumerate(new_predictions):
    print(f"Bo du lieu {i + 1}: {new_data.iloc[i].to_dict()}")
    print(f"Gia tri du doan: {prediction}")
    print("")

#
#Tạo giao diện người dùng để nhập dữ liệu và dự đoán kết quả
#
def Dudoan():
    new_data = {
        'Tuoi': int(_Tuoi.get()),
        'GioiTinh': int(_GioiTinh.get()),
        'KinhNghiemLaiXe': int(_KinhNghiemLaiXe.get()),
        'BeMatDuong': int(_BeMatDuong.get()),
        'AnhSang': int(_AnhSang.get()),
        'ThoiTiet': int(_ThoiTiet.get()),
        'NguyenNhanTaiNan': int(_NguyenNhan.get())
    }
    
    new_data_df = pd.DataFrame(new_data, index=[0])
    prediction = model.predict(new_data_df)
    ketqua_label.config(text=f'Predicted Severity: {prediction[0]:.2f}')

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Dự đoán mức độ tai nạn giao thông")
window.geometry("800x400")

# CTạo các ô nhập liệu
Tuoi_label = tk.Label(window, text="Tuoi")
_Tuoi = tk.Entry(window)

GioiTinh_label = tk.Label(window, text="Gioi Tinh (0. Nu, 1. Nam)")
_GioiTinh = tk.Entry(window)

KinhNghiemLaiXe_label = tk.Label(window, text="Kinh Nghiem Lai Xe (0. Khong co bang lai, 1. it hon 2 nam, 2. Nhieu hon 2 nam)")
_KinhNghiemLaiXe = tk.Entry(window)

BeMatDuong_label = tk.Label(window, text="Be Mat Duong (1. Duong Nhua, 2. Duong Dat)")
_BeMatDuong = tk.Entry(window)

AnhSang_label = tk.Label(window, text="Anh Sang (1. Ban ngay, 2. Toi va khong co anh sang, 3. Toi va co anh sang)")
_AnhSang = tk.Entry(window)

ThoiTiet_label = tk.Label(window, text="Thoi Tiet (1. Binh thuong, 2. Gio manh, 3. May, 4. Mua, 5. Tuyet)")
_ThoiTiet = tk.Entry(window)

NguyenNhan_label = tk.Label(window, text="Nguyen Nhan Tai Nan (1. Lai xe au, 2. Say ruou/bia, 3. Toc do cao)")
_NguyenNhan = tk.Entry(window)

DuDoan_button = tk.Button(window, text="Du Doan", command=Dudoan)

ketqua_label = tk.Label(window, text="Ket Qua Du Doan: ")

# Chạy giao diện
Tuoi_label.grid(row=0, column=0)
_Tuoi.grid(row=0, column=4)
GioiTinh_label.grid(row=2, column=0)
_GioiTinh.grid(row=2, column=4)
KinhNghiemLaiXe_label.grid(row=4, column=0)
_KinhNghiemLaiXe.grid(row=4, column=4)
BeMatDuong_label.grid(row=6, column=0)
_BeMatDuong.grid(row=6, column=4)
AnhSang_label.grid(row=8, column=0)
_AnhSang.grid(row=8, column=4)
ThoiTiet_label.grid(row=10, column=0)
_ThoiTiet.grid(row=10, column=4)
NguyenNhan_label.grid(row=12, column=0)
_NguyenNhan.grid(row=12, column=4)
DuDoan_button.grid(row=14, column=0, columnspan=2)
ketqua_label.grid(row=16, column=0, columnspan=2)

window.mainloop()




#Trực quan kết quả
# Dự đoán trên tập huấn luyện và tập kiểm tra
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.figure(num=('Group - Linear Regression - Traffic'),figsize=(10, 5))

# Vẽ đường hồi quy cho tập huấn luyện
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, c='blue', label='Training Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Dự đoán')
plt.title('Đường hồi quy cho tập huấn luyện')

# Vẽ đường hồi quy cho tập kiểm tra
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, c='red', label='Testing Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Dự đoán')
plt.title('Đường hồi quy cho tập kiểm tra')

plt.tight_layout() 
plt.show()
