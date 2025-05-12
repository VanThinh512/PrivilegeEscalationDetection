
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import sys

def predict_anomaly(csv_file, output_file=None, threshold=None, model_path=None, features_path=None, verbose=True):
    """
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
    """
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
            print(f"KẾT QUẢ PHÁT HIỆN BẤT THƯỜNG:")
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
                print(f"Top {max_display} hoạt động bất thường:")
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
