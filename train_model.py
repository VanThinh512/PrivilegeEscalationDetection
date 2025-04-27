# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã kết hợp
df = pd.read_csv('/mnt/hgfs/KaggleData/NewKaggleData/combined_data.csv')
print("Đã đọc file combined_data.csv với kích thước:", df.shape)

# Kiểm tra nhãn và phân phối dữ liệu
if 'is_anomaly' in df.columns:
    label_col = 'is_anomaly'
    print("Sử dụng cột 'is_anomaly' làm nhãn")
elif 'label' in df.columns:
    label_col = 'label'
    print("Sử dụng cột 'label' làm nhãn")
else:
    raise ValueError("Không tìm thấy cột nhãn (is_anomaly hoặc label) trong dữ liệu")

print("Phân phối nhãn:")
print(df[label_col].value_counts())
print(f"Tỷ lệ phần trăm nhãn dương: {df[label_col].mean() * 100:.2f}%")

# Kiểm tra các cột có trong dữ liệu
print("\nCác cột có trong dữ liệu:")
print(df.columns.tolist())

# Kiểm tra giá trị thiếu
print("\nGiá trị thiếu trong dữ liệu:")
print(df.isnull().sum())

# Xử lý giá trị thiếu
df = df.fillna(0)

# Chuẩn bị dữ liệu
# Loại bỏ các cột không cần thiết
exclude_cols = [label_col, 'date']  # Loại bỏ ngày tháng và nhãn
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[label_col]

# Chuyển đổi các cột không phải số thành one-hot encoding (nếu có)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty:
    print(f"\nCác cột phân loại cần chuyển đổi: {categorical_cols.tolist()}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Chuẩn hóa dữ liệu số
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
if not numeric_cols.empty:
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nKích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# Kiểm tra phân phối nhãn trong tập huấn luyện và kiểm tra
print(f"\nPhân phối nhãn trong tập huấn luyện: {np.bincount(y_train)}")
print(f"Phân phối nhãn trong tập kiểm tra: {np.bincount(y_test)}")

# Tối ưu hóa siêu tham số cho mô hình Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

print("\nBắt đầu tối ưu hóa siêu tham số...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\nCác siêu tham số tốt nhất: {grid_search.best_params_}")
print(f"Điểm F1 tốt nhất: {grid_search.best_score_:.4f}")

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Đánh giá mô hình trên tập kiểm tra
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

print("\nMa trận nhầm lẫn:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Tính toán và vẽ đường cong ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nDiện tích dưới đường cong ROC (AUC): {roc_auc:.4f}")

# Xác định tầm quan trọng của các đặc trưng
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 đặc trưng quan trọng nhất:")
print(feature_importance.head(10))

# Lưu mô hình
model_path = '/mnt/hgfs/KaggleData/NewKaggleData/privilege_detection_model.pkl'
joblib.dump(best_model, model_path)
print(f"\nĐã lưu mô hình vào {model_path}")

# Lưu các đặc trưng đã sử dụng và bộ chuẩn hóa (để sử dụng cho dự đoán)
model_features = {
    'feature_names': X.columns.tolist(),
    'numeric_cols': numeric_cols.tolist(),
    'categorical_cols': categorical_cols.tolist(),
    'scaler': scaler if not numeric_cols.empty else None
}
features_path = '/mnt/hgfs/KaggleData/NewKaggleData/model_features.pkl'
joblib.dump(model_features, features_path)
print(f"Đã lưu thông tin đặc trưng vào {features_path}")

# Tạo chức năng dự đoán đơn giản
def predict_anomaly(csv_file):
    # Đọc model và thông tin đặc trưng
    model = joblib.load('/mnt/hgfs/KaggleData/NewKaggleData/privilege_detection_model.pkl')
    features_info = joblib.load('/mnt/hgfs/KaggleData/NewKaggleData/model_features.pkl')
    
    # Tiền xử lý dữ liệu mới theo cùng cách với dữ liệu huấn luyện
    new_data = pd.read_csv(csv_file)
    
    # Đảm bảo có đủ các cột cần thiết
    for col in features_info['feature_names']:
        if col not in new_data.columns:
            new_data[col] = 0
    
    # Chuẩn hóa dữ liệu
    if features_info['scaler'] is not None:
        numeric_cols = features_info['numeric_cols']
        if numeric_cols:
            new_data[numeric_cols] = features_info['scaler'].transform(new_data[numeric_cols])
    
    # Dự đoán
    predictions = model.predict(new_data[features_info['feature_names']])
    probabilities = model.predict_proba(new_data[features_info['feature_names']])[:, 1]
    
    # Thêm kết quả dự đoán vào dữ liệu
    new_data['predicted_anomaly'] = predictions
    new_data['anomaly_probability'] = probabilities
    
    return new_data

# Lưu hàm dự đoán thành file python độc lập
with open('/mnt/hgfs/KaggleData/NewKaggleData/predict_anomaly.py', 'w') as f:
    f.write("""
import pandas as pd
import joblib

def predict_anomaly(csv_file, output_file=None):
    # Đọc model và thông tin đặc trưng
    model = joblib.load('/mnt/hgfs/KaggleData/NewKaggleData/privilege_detection_model.pkl')
    features_info = joblib.load('/mnt/hgfs/KaggleData/NewKaggleData/model_features.pkl')
    
    # Tiền xử lý dữ liệu mới
    new_data = pd.read_csv(csv_file)
    
    # Đảm bảo có đủ các cột cần thiết
    for col in features_info['feature_names']:
        if col not in new_data.columns:
            new_data[col] = 0
    
    # Chuẩn hóa dữ liệu số nếu cần
    if features_info['scaler'] is not None:
        numeric_cols = features_info['numeric_cols']
        if numeric_cols:
            new_data[numeric_cols] = features_info['scaler'].transform(new_data[numeric_cols])
    
    # Dự đoán
    predictions = model.predict(new_data[features_info['feature_names']])
    probabilities = model.predict_proba(new_data[features_info['feature_names']])[:, 1]
    
    # Thêm kết quả dự đoán vào dữ liệu
    new_data['predicted_anomaly'] = predictions
    new_data['anomaly_probability'] = probabilities
    
    if output_file:
        new_data.to_csv(output_file, index=False)
        print(f"Đã lưu kết quả dự đoán vào {output_file}")
    
    # Báo cáo kết quả
    anomaly_count = new_data['predicted_anomaly'].sum()
    total_records = len(new_data)
    print(f"Phát hiện {anomaly_count} hoạt động bất thường trong tổng số {total_records} bản ghi")
    print(f"Tỷ lệ bất thường: {anomaly_count/total_records*100:.2f}%")
    
    if anomaly_count > 0:
        print("\\nHoạt động bất thường được phát hiện:")
        print(new_data[new_data['predicted_anomaly'] == 1].head(10))
    
    return new_data

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        predict_anomaly(input_file, output_file)
    else:
        print("Sử dụng: python predict_anomaly.py <input_csv_file> [output_csv_file]")
""")

print("\nĐã tạo file dự đoán predict_anomaly.py")