# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("========== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN LEO THANG ĐẶC QUYỀN ==========")

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True
)

print(f"\nKích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# Kiểm tra phân phối nhãn trong tập huấn luyện và kiểm tra
print(f"\nPhân phối nhãn trong tập huấn luyện: {np.bincount(y_train)}")
print(f"Phân phối nhãn trong tập kiểm tra: {np.bincount(y_test)}")

# Đánh giá mức độ mất cân bằng
imbalance_ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
print(f"Tỷ lệ mất cân bằng (negative/positive): {imbalance_ratio:.2f}")

# Nếu dữ liệu mất cân bằng, thêm SMOTE
use_smote = imbalance_ratio > 2.0  # Nếu tỷ lệ mất cân bằng > 2, áp dụng SMOTE
if use_smote:
    print("Áp dụng SMOTE để cân bằng dữ liệu huấn luyện")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Phân phối nhãn sau SMOTE: {np.bincount(y_train_resampled)}")
else:
    X_train_resampled, y_train_resampled = X_train, y_train
    print("Không áp dụng SMOTE vì dữ liệu đủ cân bằng")

# Đơn giản hóa lưới tham số
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

print("\nBắt đầu tối ưu hóa siêu tham số...")
print(f"Số tổ hợp tham số cần đánh giá: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['class_weight'])}")

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tạo RandomForestClassifier với các tham số mặc định tốt để giảm overfitting
base_rf = RandomForestClassifier(
    random_state=42, 
    bootstrap=True,
    max_features='sqrt',  # Giới hạn số đặc trưng để giảm overfitting
    n_jobs=-1
)

grid_search = GridSearchCV(
    base_rf,
    param_grid=param_grid,
    cv=stratified_cv,
    scoring='f1',  # Đơn giản hóa, chỉ sử dụng F1 để tối ưu
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Huấn luyện trên dữ liệu đã cân bằng
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\nCác siêu tham số tốt nhất: {grid_search.best_params_}")
print(f"Điểm F1 tốt nhất trên tập validation: {grid_search.best_score_:.4f}")

# Kiểm tra overfitting
best_index = grid_search.best_index_
train_f1 = grid_search.cv_results_['mean_train_score'][best_index]
test_f1 = grid_search.cv_results_['mean_test_score'][best_index]

print(f"\nĐánh giá overfitting cho mô hình tốt nhất:")
print(f"F1 trên tập huấn luyện: {train_f1:.4f}")
print(f"F1 trên tập validation: {test_f1:.4f}")
print(f"Chênh lệch (train - test): {train_f1 - test_f1:.4f}")

if train_f1 - test_f1 > 0.1:
    print("CẢNH BÁO: Mô hình có dấu hiệu overfitting!")
    # Điều chỉnh tham số để giảm overfitting
    print("Thử điều chỉnh tham số để giảm overfitting...")
    
    # Tạo một mô hình mới với các tham số chặt chẽ hơn
    adjusted_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=min(10, grid_search.best_params_.get('max_depth', 10)),
        min_samples_split=max(5, grid_search.best_params_.get('min_samples_split', 5)),
        min_samples_leaf=max(2, grid_search.best_params_.get('min_samples_leaf', 1)),
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        bootstrap=True
    )
    adjusted_rf.fit(X_train_resampled, y_train_resampled)
    
    # Kiểm tra hiệu suất của mô hình đã điều chỉnh
    cv_scores_adjusted = cross_val_score(adjusted_rf, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
    print(f"Điểm F1 cross-validation của mô hình đã điều chỉnh: {cv_scores_adjusted.mean():.4f}")
    
    # Nếu mô hình điều chỉnh tốt hơn, sử dụng nó
    if cv_scores_adjusted.mean() > test_f1 - 0.05:  # Chấp nhận giảm nhẹ hiệu suất để giảm overfitting
        print("Sử dụng mô hình đã điều chỉnh...")
        best_model = adjusted_rf
    else:
        print("Giữ nguyên mô hình gốc...")
        best_model = grid_search.best_estimator_
else:
    best_model = grid_search.best_estimator_

# Đánh giá mô hình trên tập kiểm tra
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBáo cáo phân loại trên tập kiểm tra:")
print(classification_report(y_test, y_pred))

print("\nMa trận nhầm lẫn:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Tính toán các metrics
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nDiện tích dưới đường cong ROC (AUC): {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
print(f"Điểm Precision-Recall trung bình (AP): {avg_precision:.4f}")

# Xác định tầm quan trọng của các đặc trưng
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 đặc trưng quan trọng nhất:")
print(feature_importance.head(10))

# Thử nghiệm với các ngưỡng dự đoán khác nhau
print("\nKiểm tra hiệu suất với các ngưỡng khác nhau:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred_threshold, output_dict=True)
    
    threshold_results.append({
        'threshold': threshold,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'], 
        'f1': report['1']['f1-score']
    })
    
    print(f"Ngưỡng {threshold}: "
          f"Precision={report['1']['precision']:.3f}, "
          f"Recall={report['1']['recall']:.3f}, "
          f"F1={report['1']['f1-score']:.3f}")

# Tìm ngưỡng tốt nhất dựa trên F1
best_threshold_index = np.argmax([res['f1'] for res in threshold_results])
best_threshold = threshold_results[best_threshold_index]['threshold']
print(f"\nNgưỡng tốt nhất dựa trên F1: {best_threshold}")

# Lưu thông tin ngưỡng tốt nhất
best_threshold_info = threshold_results[best_threshold_index]

# Lưu mô hình với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = '/mnt/hgfs/KaggleData/NewKaggleData/models'
os.makedirs(model_dir, exist_ok=True)
model_path = f'{model_dir}/privilege_detection_model_{timestamp}.pkl'
joblib.dump(best_model, model_path)
print(f"\nĐã lưu mô hình vào {model_path}")

# Lưu các đặc trưng đã sử dụng và bộ chuẩn hóa
model_features = {
    'feature_names': X.columns.tolist(),
    'numeric_cols': numeric_cols.tolist(),
    'categorical_cols': categorical_cols.tolist(),
    'scaler': scaler if not numeric_cols.empty else None,
    'best_params': grid_search.best_params_,
    'best_threshold': best_threshold,
    'best_threshold_metrics': best_threshold_info,
    'training_date': timestamp,
    'class_distribution': {
        'train': np.bincount(y_train).tolist(),
        'test': np.bincount(y_test).tolist()
    }
}
features_path = f'{model_dir}/model_features_{timestamp}.pkl'
joblib.dump(model_features, features_path)
print(f"Đã lưu thông tin đặc trưng vào {features_path}")

# Thêm một link tới mô hình mới nhất
latest_model_path = '/mnt/hgfs/KaggleData/NewKaggleData/privilege_detection_model_latest.pkl'
latest_features_path = '/mnt/hgfs/KaggleData/NewKaggleData/model_features_latest.pkl'
joblib.dump(best_model, latest_model_path)
joblib.dump(model_features, latest_features_path)
print(f"Đã cập nhật mô hình mới nhất tại {latest_model_path}")

# Tạo công cụ dự đoán đơn giản
with open('/mnt/hgfs/KaggleData/NewKaggleData/predict_anomaly.py', 'w') as f:
    f.write("""
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import sys

def predict_anomaly(csv_file, output_file=None, threshold=None, model_path=None, features_path=None, verbose=True):
    \"\"\"
    Dự đoán hành vi bất thường trong dữ liệu lệnh đặc quyền.
    
    Tham số:
        csv_file (str): Đường dẫn đến file CSV chứa dữ liệu cần dự đoán
        output_file (str, optional): Đường dẫn đến file lưu kết quả
        threshold (float, optional): Ngưỡng xác suất để phân loại bất thường (0.0-1.0)
        model_path (str, optional): Đường dẫn đến file mô hình
        features_path (str, optional): Đường dẫn đến file thông tin đặc trưng
        verbose (bool): In thông tin chi tiết
    
    Trả về:
        DataFrame: Dữ liệu gốc với cột dự đoán bổ sung
    \"\"\"
    # Đọc mô hình và thông tin đặc trưng
    if model_path is None:
        model_path = '/mnt/hgfs/KaggleData/NewKaggleData/privilege_detection_model_latest.pkl'
    
    if features_path is None:
        features_path = '/mnt/hgfs/KaggleData/NewKaggleData/model_features_latest.pkl'
    
    try:
        model = joblib.load(model_path)
        features_info = joblib.load(features_path)
        
        if verbose:
            print(f"Đã tải mô hình từ {model_path}")
            print(f"Thông tin mô hình: {features_info.get('training_date', 'N/A')}")
        
        # Sử dụng ngưỡng tốt nhất từ đánh giá nếu không được chỉ định
        if threshold is None and 'best_threshold' in features_info:
            threshold = features_info['best_threshold']
            if verbose:
                print(f"Sử dụng ngưỡng tối ưu từ đánh giá: {threshold}")
        elif threshold is None:
            threshold = 0.5
            if verbose:
                print(f"Sử dụng ngưỡng mặc định: {threshold}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None
    
    # Tiền xử lý dữ liệu
    try:
        if verbose:
            print(f"Đọc dữ liệu từ {csv_file}...")
        new_data = pd.read_csv(csv_file)
        if verbose:
            print(f"Đã đọc {len(new_data)} dòng dữ liệu")
    except Exception as e:
        print(f"Lỗi đọc file CSV: {e}")
        return None
    
    # Chuẩn bị dữ liệu
    try:
        # Đảm bảo có đủ các cột cần thiết
        missing_cols = [col for col in features_info['feature_names'] if col not in new_data.columns]
        if missing_cols:
            if verbose:
                print(f"Thiếu các cột: {missing_cols}")
            for col in missing_cols:
                new_data[col] = 0
        
        # Chuyển đổi các cột phân loại
        categorical_cols = features_info.get('categorical_cols', [])
        if categorical_cols:
            new_data = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)
            
            # Đảm bảo tất cả các cột đặc trưng cần thiết có trong dữ liệu
            for col in features_info['feature_names']:
                if col not in new_data.columns:
                    new_data[col] = 0
        
        # Chuẩn hóa dữ liệu
        scaler = features_info.get('scaler')
        numeric_cols = features_info.get('numeric_cols', [])
        if scaler is not None and numeric_cols:
            try:
                # Chỉ chuẩn hóa các cột có trong dữ liệu
                cols_to_scale = [col for col in numeric_cols if col in new_data.columns]
                if cols_to_scale:
                    new_data[cols_to_scale] = scaler.transform(new_data[cols_to_scale])
            except Exception as e:
                print(f"Cảnh báo: Không thể chuẩn hóa dữ liệu: {e}")
        
        # Xác định các cột đặc trưng có trong dữ liệu
        features = [col for col in features_info['feature_names'] if col in new_data.columns]
        
        if len(features) != len(features_info['feature_names']):
            print(f"Cảnh báo: Chỉ tìm thấy {len(features)}/{len(features_info['feature_names'])} đặc trưng")
            if verbose:
                print(f"Thiếu: {set(features_info['feature_names']) - set(features)}")
        
        # Dự đoán
        if verbose:
            print("Đang dự đoán...")
        probabilities = model.predict_proba(new_data[features])[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Thêm kết quả dự đoán
        new_data['predicted_anomaly'] = predictions
        new_data['anomaly_probability'] = probabilities
        new_data['prediction_confidence'] = np.abs(probabilities - 0.5) * 2
        
        # Lưu kết quả
        if output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = output_file if '.csv' in output_file else f"{output_file}_{timestamp}.csv"
            new_data.to_csv(output_name, index=False)
            if verbose:
                print(f"Đã lưu kết quả dự đoán vào {output_name}")
        
        # Tóm tắt kết quả
        anomaly_count = predictions.sum()
        total_records = len(new_data)
        
        if verbose:
            print(f"\nKẾT QUẢ PHÁT HIỆN BẤT THƯỜNG:")
            print(f"- Tổng số bản ghi: {total_records}")
            print(f"- Số bất thường phát hiện: {anomaly_count} ({anomaly_count/total_records*100:.2f}%)")
        
        if anomaly_count > 0:
            # Sắp xếp các bất thường theo xác suất
            anomalies = new_data[new_data['predicted_anomaly'] == 1].sort_values('anomaly_probability', ascending=False)
            
            # Hiển thị các cột quan trọng
            display_cols = ['anomaly_probability', 'prediction_confidence']
            
            # Thêm cột thời gian và người dùng
            time_cols = [col for col in ['timestamp', 'datetime', 'date', 'hour'] if col in anomalies.columns]
            if 'user' in anomalies.columns:
                display_cols = time_cols + ['user'] + display_cols
            else:
                display_cols = time_cols + display_cols
            
            # Thêm thông tin lệnh
            cmd_cols = [col for col in ['command', 'args', 'key'] if col in anomalies.columns]
            display_cols = display_cols + cmd_cols
            
            # Các cột liên quan đến hành vi bất thường
            behavior_cols = [col for col in ['is_after_hours', 'is_suspicious_command', 'has_suspicious_args'] 
                             if col in anomalies.columns]
            display_cols = display_cols + behavior_cols
            
            # Hiển thị top bất thường
            max_display = min(5, len(anomalies))
            if verbose and max_display > 0:
                print(f"\nTop {max_display} hoạt động bất thường:")
                print(anomalies[display_cols].head(max_display).to_string())
        
        return new_data
    
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dự đoán bất thường trong dữ liệu lệnh đặc quyền')
    parser.add_argument('input_file', help='File CSV đầu vào')
    parser.add_argument('-o', '--output', help='File CSV đầu ra')
    parser.add_argument('-t', '--threshold', type=float, help='Ngưỡng xác suất (0.0-1.0)')
    parser.add_argument('-m', '--model', help='Đường dẫn tới file mô hình')
    parser.add_argument('-f', '--features', help='Đường dẫn tới file thông tin đặc trưng')
    parser.add_argument('-q', '--quiet', action='store_true', help='Không hiển thị thông báo chi tiết')
    
    args = parser.parse_args()
    
    result = predict_anomaly(
        args.input_file,
        args.output,
        args.threshold,
        args.model,
        args.features,
        not args.quiet
    )
    
    if result is None:
        sys.exit(1)
""")

print("\nĐã tạo file dự đoán predict_anomaly.py")
print("\n========== HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH ==========")