import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# =========================
# Đọc file Excel
# =========================
file_path = "TỔNG HỢP ĐIỂM K58KTP.xlsx"

raw = pd.read_excel(file_path, header=None)

# =========================
# Lấy MSSV và tên sinh viên
# =========================
student_ids = raw.iloc[1, 3:].values
student_names = raw.iloc[2, 3:].values

# =========================
# Lấy bảng điểm
# =========================
score_data = raw.iloc[4:, 3:]

# Chuyển sang số
score_data = score_data.apply(pd.to_numeric, errors='coerce')
score_data = score_data.fillna(0)

# =========================
# Tính điểm trung bình
# =========================
average_scores = score_data.mean(axis=0).round(3)

# Chuyển thành DataFrame
students = pd.DataFrame({
    'MSSV': student_ids,
    'Tên sinh viên': student_names,
    'Điểm trung bình': average_scores.values
})

# =========================
# Chuẩn hóa dữ liệu
# =========================
X = students[['Điểm trung bình']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# K-Means phân 3 cụm
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)

students['Cluster'] = kmeans.fit_predict(X_scaled)

# =========================
# Sắp xếp cụm theo điểm TB
# =========================
cluster_mean = students.groupby('Cluster')['Điểm trung bình'].mean()

# Sort tăng dần
sorted_clusters = cluster_mean.sort_values().index

# Map:
# thấp nhất -> 2 (trung bình)
# giữa -> 1 (khá)
# cao nhất -> 0 (giỏi)
cluster_mapping = {
    sorted_clusters[0]: 2,
    sorted_clusters[1]: 1,
    sorted_clusters[2]: 0
}

students['Cụm'] = students['Cluster'].map(cluster_mapping)

# Xóa cột cluster cũ
students = students.drop(columns=['Cluster'])

# =========================
# In kết quả
# =========================
print(students)

# =========================
# Lưu file Excel
# =========================
students.to_excel("ket_qua_phan_cum.xlsx", index=False)

print("\nĐã lưu file ket_qua_phan_cum.xlsx")

# =========================
# Vẽ biểu đồ phân cụm
# =========================

plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green']

for cluster in sorted(students['Cụm'].unique()):
    
    cluster_data = students[students['Cụm'] == cluster]

    plt.scatter(
        cluster_data.index,
        cluster_data['Điểm trung bình'],
        color=colors[cluster],
        label=f'Cụm {cluster}'
    )

# Hiển thị tên cụm
plt.legend()

# Tiêu đề
plt.title('Phân cụm sinh viên bằng K-Means')

# Tên trục
plt.xlabel('Sinh viên')
plt.ylabel('Điểm trung bình')

# Hiện lưới
plt.grid(True)

# Hiển thị biểu đồ
plt.show()