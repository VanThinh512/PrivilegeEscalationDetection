"""
model_monitor.py - Module giám sát hiệu suất mô hình phát hiện leo thang đặc quyền theo thời gian
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_monitor')

class ModelMonitor:
    """Hệ thống giám sát hiệu suất mô hình theo thời gian"""
    
    def __init__(self, model_path, features_path, metrics_log_path=None, threshold=0.05):
        """
        Khởi tạo hệ thống giám sát
        
        Parameters:
        -----------
        model_path : str
            Đường dẫn đến file mô hình
        features_path : str
            Đường dẫn đến file đặc trưng
        metrics_log_path : str, optional
            Đường dẫn đến file log các chỉ số
        threshold : float, default=0.05
            Ngưỡng phát hiện drift (thay đổi 5%)
        """
        self.model_path = model_path
        self.features_path = features_path
        self.threshold = threshold
        
        # Tạo đường dẫn log mặc định nếu không được cung cấp
        if metrics_log_path is None:
            log_dir = os.path.dirname(model_path)
            self.metrics_log_path = os.path.join(log_dir, 'model_metrics_log.json')
        else:
            self.metrics_log_path = metrics_log_path
        
        # Tải mô hình và thông tin đặc trưng
        try:
            self.model = joblib.load(model_path)
            self.features_info = joblib.load(features_path)
            logger.info(f"Đã tải mô hình từ {model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            raise
        
        # Tải lịch sử chỉ số 
        self.metrics_history = self._load_metrics_history()
        self.baseline_metrics = self._get_baseline_metrics()
        
        logger.info("Đã khởi tạo hệ thống giám sát mô hình")
    
    def _load_metrics_history(self):
        """Tải lịch sử chỉ số từ file log"""
        if os.path.exists(self.metrics_log_path):
            try:
                with open(self.metrics_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Không thể đọc file log chỉ số: {e}. Tạo mới.")
        
        return []
    
    def _get_baseline_metrics(self):
        """Lấy chỉ số cơ sở từ lịch sử hoặc tạo mới"""
        if self.metrics_history:
            # Sử dụng lần đánh giá gần nhất làm cơ sở
            return self.metrics_history[-1]['metrics']
        else:
            # Trả về giá trị mặc định nếu không có lịch sử
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
    
    def _save_metrics_history(self):
        """Lưu lịch sử chỉ số vào file"""
        try:
            with open(self.metrics_log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            logger.info(f"Đã lưu lịch sử chỉ số vào {self.metrics_log_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu lịch sử chỉ số: {e}")
    
    def _calculate_metrics(self, X, y):
        """Tính toán các chỉ số đánh giá"""
        try:
            # Dự đoán
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]
            
            # Tính các chỉ số
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }
            
            # Thêm AUC nếu có cả lớp dương và âm
            if len(np.unique(y)) > 1:
                metrics['auc'] = roc_auc_score(y, y_prob)
            else:
                metrics['auc'] = 0.0
            
            return metrics
        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số: {e}")
            return None
    
    def check_drift(self, X_new, y_new, save_metrics=True):
        """
        Kiểm tra model drift trên dữ liệu mới
        
        Parameters:
        -----------
        X_new : pandas.DataFrame
            Dữ liệu đặc trưng mới
        y_new : pandas.Series
            Nhãn mới
        save_metrics : bool, default=True
            Có lưu kết quả vào lịch sử không
            
        Returns:
        --------
        bool
            True nếu phát hiện drift, False nếu không
        """
        logger.info("Kiểm tra model drift trên dữ liệu mới")
        
        # Chuẩn bị dữ liệu
        feature_names = self.features_info.get('feature_names', [])
        
        # Đảm bảo X_new có đúng các cột như mô hình yêu cầu
        missing_cols = [col for col in feature_names if col not in X_new.columns]
        if missing_cols:
            logger.warning(f"Thiếu các cột: {missing_cols}")
            # Tạo các cột thiếu với giá trị 0
            for col in missing_cols:
                X_new[col] = 0
        
        # Chỉ giữ lại các cột cần thiết theo thứ tự
        X_new = X_new[feature_names]
        
        # Tính toán chỉ số trên dữ liệu mới
        current_metrics = self._calculate_metrics(X_new, y_new)
        if current_metrics is None:
            logger.error("Không thể tính toán chỉ số trên dữ liệu mới")
            return False
        
        # So sánh với chỉ số cơ sở
        drift_detected = False
        drift_details = {}
        
        for metric in self.baseline_metrics:
            if metric in current_metrics:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics[metric]
                
                # Tính phần trăm thay đổi
                if baseline_value > 0:
                    change = abs((current_value - baseline_value) / baseline_value)
                    
                    # Kiểm tra drift
                    if change > self.threshold:
                        drift_details[metric] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'change': change * 100  # Phần trăm
                        }
                        drift_detected = True
        
        # Log kết quả
        if drift_detected:
            logger.warning("PHÁT HIỆN MODEL DRIFT!")
            for metric, detail in drift_details.items():
                logger.warning(f"{metric}: {detail['baseline']:.4f} -> {detail['current']:.4f} (thay đổi {detail['change']:.2f}%)")
        else:
            logger.info("Không phát hiện model drift")
        
        # Lưu vào lịch sử
        if save_metrics:
            self.metrics_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': current_metrics,
                'drift_detected': drift_detected,
                'drift_details': drift_details
            })
            self._save_metrics_history()
        
        return drift_detected
    
    def plot_metrics_trend(self, output_file=None):
        """
        Vẽ biểu đồ xu hướng các chỉ số theo thời gian
        
        Parameters:
        -----------
        output_file : str, optional
            Đường dẫn để lưu biểu đồ
            
        Returns:
        --------
        matplotlib.figure.Figure
            Đối tượng Figure chứa biểu đồ
        """
        if not self.metrics_history:
            logger.warning("Không có đủ dữ liệu để vẽ biểu đồ xu hướng")
            return None
        
        try:
            # Tạo DataFrame từ lịch sử
            history_df = pd.DataFrame([
                {
                    'timestamp': item['timestamp'],
                    **item['metrics']
                }
                for item in self.metrics_history
            ])
            
            # Chuyển đổi timestamp
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Vẽ biểu đồ
            plt.figure(figsize=(12, 8))
            
            # Vẽ các chỉ số chính
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            for metric in metrics_to_plot:
                if metric in history_df.columns:
                    plt.plot(history_df['timestamp'], history_df[metric], marker='o', label=metric.capitalize())
            
            plt.title('Xu hướng các chỉ số đánh giá theo thời gian')
            plt.xlabel('Thời gian')
            plt.ylabel('Giá trị')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Lưu biểu đồ nếu cần
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Đã lưu biểu đồ xu hướng vào {output_file}")
            
            return plt.gcf()
        
        except Exception as e:
            logger.error(f"Lỗi khi vẽ biểu đồ xu hướng: {e}")
            return None
    
    def recommend_retraining(self):
        """
        Đưa ra khuyến nghị có nên huấn luyện lại mô hình hay không
        
        Returns:
        --------
        dict
            Thông tin khuyến nghị
        """
        if not self.metrics_history or len(self.metrics_history) < 2:
            return {
                'should_retrain': False,
                'reason': 'Không đủ dữ liệu lịch sử để đánh giá'
            }
        
        # Lấy các lần đánh giá gần nhất
        recent_evaluations = self.metrics_history[-min(5, len(self.metrics_history)):]
        
        # Đếm số lần phát hiện drift
        drift_count = sum(1 for item in recent_evaluations if item.get('drift_detected', False))
        
        # Kiểm tra xu hướng chỉ số F1
        f1_trend = [eval_item['metrics'].get('f1', 0) for eval_item in recent_evaluations]
        
        # Khuyến nghị dựa trên số lần drift và xu hướng
        if drift_count >= 2:
            return {
                'should_retrain': True,
                'reason': f'Phát hiện model drift {drift_count} lần trong {len(recent_evaluations)} lần đánh giá gần nhất'
            }
        elif len(f1_trend) >= 3 and f1_trend[-1] < f1_trend[0] * 0.9:
            return {
                'should_retrain': True,
                'reason': f'Hiệu suất F1 giảm hơn 10% (từ {f1_trend[0]:.4f} xuống {f1_trend[-1]:.4f})'
            }
        else:
            return {
                'should_retrain': False,
                'reason': 'Mô hình vẫn hoạt động ổn định'
            }

def retrain_pipeline(model_path, features_path, new_data_path, metrics_log_path=None, min_samples=100):
    """
    Pipeline tự động kiểm tra và huấn luyện lại mô hình khi cần
    
    Parameters:
    -----------
    model_path : str
        Đường dẫn đến file mô hình
    features_path : str
        Đường dẫn đến file đặc trưng
    new_data_path : str
        Đường dẫn đến dữ liệu mới
    metrics_log_path : str, optional
        Đường dẫn đến file log các chỉ số
    min_samples : int, default=100
        Số lượng mẫu tối thiểu để huấn luyện lại
    
    Returns:
    --------
    bool
        True nếu huấn luyện lại, False nếu không
    """
    try:
        # Tạo đối tượng giám sát
        monitor = ModelMonitor(model_path, features_path, metrics_log_path)
        
        # Đọc dữ liệu mới
        new_data = pd.read_csv(new_data_path)
        logger.info(f"Đã đọc dữ liệu mới từ {new_data_path}, kích thước: {new_data.shape}")
        
        # Kiểm tra có đủ dữ liệu không
        if len(new_data) < min_samples:
            logger.warning(f"Không đủ dữ liệu mới để đánh giá (cần ít nhất {min_samples} mẫu)")
            return False
        
        # Chuẩn bị dữ liệu
        feature_names = monitor.features_info.get('feature_names', [])
        label_col = 'is_anomaly'
        
        # Kiểm tra dữ liệu có nhãn không
        if label_col not in new_data.columns:
            logger.error(f"Dữ liệu mới không có cột nhãn {label_col}")
            return False
        
        # Kiểm tra drift
        drift_detected = monitor.check_drift(new_data[feature_names], new_data[label_col])
        
        # Kiểm tra khuyến nghị huấn luyện lại
        recommendation = monitor.recommend_retraining()
        
        if drift_detected or recommendation['should_retrain']:
            logger.info(f"Khuyến nghị huấn luyện lại mô hình: {recommendation['reason']}")
            # Gọi script huấn luyện mô hình
            from train_model import train_model_from_data
            
            # Thêm logic huấn luyện lại tại đây (gọi hàm từ train_model.py)
            # train_model_from_data(new_data)
            
            return True
        else:
            logger.info("Không cần huấn luyện lại mô hình")
            return False
            
    except Exception as e:
        logger.error(f"Lỗi trong pipeline huấn luyện lại: {e}")
        return False

if __name__ == "__main__":
    # Ví dụ sử dụng
    model_path = '/home/joe/python_Proj/test/privilege_detection_model_latest.pkl'
    features_path = '/home/joe/python_Proj/test/model_features_latest.pkl'
    
    # Tạo đối tượng giám sát
    try:
        monitor = ModelMonitor(model_path, features_path)
        print("Đã khởi tạo hệ thống giám sát mô hình thành công")
        
        # Vẽ biểu đồ xu hướng nếu có dữ liệu
        if monitor.metrics_history:
            monitor.plot_metrics_trend(output_file='/home/joe/python_Proj/test/metrics_trend.png')
            print("Đã tạo biểu đồ xu hướng")
        else:
            print("Chưa có dữ liệu lịch sử để vẽ biểu đồ")
        
    except Exception as e:
        print(f"Lỗi: {e}")
