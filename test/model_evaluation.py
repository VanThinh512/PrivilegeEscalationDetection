"""
model_evaluation.py - Module đánh giá và giải thích mô hình phát hiện leo thang đặc quyền
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, auc, average_precision_score,
    roc_auc_score, f1_score, precision_score, recall_score
)
import shap

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

def generate_evaluation_report(model, X_test, y_test, output_dir=None, threshold=0.5):
    """
    Tạo báo cáo đánh giá đầy đủ cho mô hình
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Mô hình đã huấn luyện
    X_test : pandas.DataFrame
        Dữ liệu đặc trưng kiểm tra
    y_test : pandas.Series
        Nhãn kiểm tra
    output_dir : str, optional
        Thư mục để lưu báo cáo
    threshold : float, default=0.5
        Ngưỡng dự đoán
        
    Returns:
    --------
    dict
        Kết quả đánh giá
    """
    logger.info("Tạo báo cáo đánh giá mô hình")
    
    # Tạo thư mục đầu ra nếu cần
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Áp dụng ngưỡng
    y_pred_threshold = (y_prob >= threshold).astype(int)
    
    # 1. Báo cáo phân loại
    class_report = classification_report(y_test, y_pred_threshold, output_dict=True)
    
    # In ra báo cáo
    print("\nBÁO CÁO PHÂN LOẠI:")
    print(classification_report(y_test, y_pred_threshold))
    
    # 2. Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred_threshold)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bình thường', 'Bất thường'],
                yticklabels=['Bình thường', 'Bất thường'])
    plt.title('Ma trận nhầm lẫn')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Đường cong ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # 4. Đường cong Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', 
               label=f'Ngẫu nhiên (PR = {sum(y_test)/len(y_test):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # 5. Phân tích theo nhiều ngưỡng
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        threshold_results.append({
            'threshold': t,
            'precision': precision_score(y_test, y_pred_t, zero_division=0),
            'recall': recall_score(y_test, y_pred_t, zero_division=0),
            'f1': f1_score(y_test, y_pred_t, zero_division=0),
            'positives': sum(y_pred_t)
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], 'g-', label='Recall')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], 'r-', label='F1')
    
    # Đánh dấu ngưỡng được chọn
    plt.axvline(x=threshold, color='purple', linestyle='--', 
               label=f'Ngưỡng hiện tại ({threshold})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ngưỡng')
    plt.ylabel('Giá trị')
    plt.title('Hiệu suất theo ngưỡng dự đoán')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
        # Lưu bảng ngưỡng
        threshold_df.to_csv(os.path.join(output_dir, 'threshold_analysis.csv'), index=False)
    plt.close()
    
    # 6. Độ quan trọng của đặc trưng
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 đặc trưng quan trọng nhất')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        # Lưu bảng độ quan trọng
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    plt.close()
    
    # Tổng hợp kết quả
    results = {
        'accuracy': class_report['accuracy'],
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1': class_report['1']['f1-score'],
        'auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
        'feature_importance': feature_importance.to_dict(orient='records')
    }
    
    # Lưu kết quả dưới dạng JSON
    if output_dir:
        import json
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    logger.info("Đã hoàn thành đánh giá mô hình")
    return results

def explain_model_predictions(model, X_sample, feature_names=None, output_dir=None, max_display=20):
    """
    Giải thích dự đoán của mô hình bằng SHAP
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Mô hình đã huấn luyện
    X_sample : pandas.DataFrame
        Dữ liệu mẫu để giải thích
    feature_names : list, optional
        Danh sách tên đặc trưng
    output_dir : str, optional
        Thư mục để lưu đồ thị
    max_display : int, default=20
        Số lượng đặc trưng tối đa để hiển thị
        
    Returns:
    --------
    dict
        Kết quả giải thích
    """
    logger.info("Bắt đầu giải thích dự đoán mô hình")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Khởi tạo SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Tính SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Nếu mô hình phân loại nhị phân, lấy shap cho lớp dương
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Lấy lớp dương (index 1)
        
        # Vẽ summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                          max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        plt.close()
        
        # Vẽ beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                          plot_type='bar', max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_bar.png'))
        plt.close()
        
        # Chọn một số mẫu để giải thích chi tiết
        num_samples = min(5, len(X_sample))
        sample_indices = np.random.choice(len(X_sample), num_samples, replace=False)
        
        explanations = []
        
        for i, idx in enumerate(sample_indices):
            # Vẽ force plot
            plt.figure(figsize=(12, 4))
            shap.force_plot(explainer.expected_value if not isinstance(explainer.expected_value, list)
                           else explainer.expected_value[1],
                           shap_values[idx], X_sample.iloc[idx], feature_names=feature_names, 
                           matplotlib=True, show=False)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'sample_{i+1}_force_plot.png'))
            plt.close()
            
            # Lưu thông tin giải thích
            explanations.append({
                'sample_index': idx,
                'prediction': model.predict_proba(X_sample.iloc[idx:idx+1])[0][1],
                'top_features': [
                    {'feature': feature_names[j] if feature_names else f'feature_{j}',
                     'shap_value': float(shap_values[idx][j])}
                    for j in np.argsort(np.abs(shap_values[idx]))[-5:]
                ]
            })
        
        # Lưu kết quả
        results = {
            'expected_value': float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                                  else explainer.expected_value[1]),
            'sample_explanations': explanations
        }
        
        if output_dir:
            import json
            with open(os.path.join(output_dir, 'shap_explanations.json'), 'w') as f:
                json.dump(results, f, indent=4)
        
        logger.info("Đã hoàn thành giải thích mô hình")
        return results
    
    except Exception as e:
        logger.error(f"Lỗi khi giải thích mô hình: {e}")
        return None

def generate_model_report(model_path, features_path, data_path, output_dir, label_col='is_anomaly', test_size=0.3):
    """
    Tạo báo cáo đầy đủ cho mô hình
    
    Parameters:
    -----------
    model_path : str
        Đường dẫn đến file mô hình
    features_path : str
        Đường dẫn đến file thông tin đặc trưng
    data_path : str
        Đường dẫn đến file dữ liệu
    output_dir : str
        Thư mục để lưu báo cáo
    label_col : str, default='is_anomaly'
        Tên cột nhãn
    test_size : float, default=0.3
        Tỷ lệ dữ liệu kiểm tra
        
    Returns:
    --------
    bool
        True nếu tạo báo cáo thành công, False nếu không
    """
    try:
        # Tải mô hình và thông tin đặc trưng
        model = joblib.load(model_path)
        features_info = joblib.load(features_path)
        
        # Tải dữ liệu
        data = pd.read_csv(data_path)
        
        # Kiểm tra cột nhãn
        if label_col not in data.columns:
            logger.error(f"Không tìm thấy cột nhãn '{label_col}' trong dữ liệu")
            return False
        
        # Lấy danh sách đặc trưng
        feature_names = features_info.get('feature_names', [])
        
        # Chia dữ liệu
        from sklearn.model_selection import train_test_split
        X = data[feature_names]
        y = data[label_col]
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Tạo thư mục báo cáo
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo báo cáo đánh giá
        eval_dir = os.path.join(output_dir, 'evaluation')
        results = generate_evaluation_report(model, X_test, y_test, eval_dir)
        
        # Giải thích mô hình
        explain_dir = os.path.join(output_dir, 'explanation')
        explain_model_predictions(model, X_test.sample(min(100, len(X_test))), feature_names, explain_dir)
        
        # Tạo báo cáo HTML
        create_html_report(results, output_dir)
        
        logger.info(f"Đã tạo báo cáo đầy đủ tại {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo: {e}")
        return False

def create_html_report(results, output_dir):
    """
    Tạo báo cáo HTML từ kết quả đánh giá
    
    Parameters:
    -----------
    results : dict
        Kết quả đánh giá
    output_dir : str
        Thư mục đầu ra
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Báo cáo đánh giá mô hình phát hiện leo thang đặc quyền</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .metric {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            .metric span {{ font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ flex: 1; min-width: 300px; margin: 10px; }}
        </style>
    </head>
    <body>
        <h1>Báo cáo đánh giá mô hình phát hiện leo thang đặc quyền</h1>
        
        <h2>1. Tóm tắt hiệu suất</h2>
        <div class="metric">
            <p><span>Accuracy:</span> {results.get('accuracy', 'N/A'):.4f}</p>
            <p><span>Precision:</span> {results.get('precision', 'N/A'):.4f}</p>
            <p><span>Recall:</span> {results.get('recall', 'N/A'):.4f}</p>
            <p><span>F1 Score:</span> {results.get('f1', 'N/A'):.4f}</p>
            <p><span>AUC-ROC:</span> {results.get('auc', 'N/A'):.4f}</p>
            <p><span>Average Precision:</span> {results.get('avg_precision', 'N/A'):.4f}</p>
        </div>
        
        <h2>2. Biểu đồ</h2>
        <div class="container">
            <div class="chart">
                <h3>Ma trận nhầm lẫn</h3>
                <img src="evaluation/confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="chart">
                <h3>Đường cong ROC</h3>
                <img src="evaluation/roc_curve.png" alt="ROC Curve">
            </div>
        </div>
        <div class="container">
            <div class="chart">
                <h3>Đường cong Precision-Recall</h3>
                <img src="evaluation/precision_recall_curve.png" alt="PR Curve">
            </div>
            <div class="chart">
                <h3>Phân tích ngưỡng</h3>
                <img src="evaluation/threshold_analysis.png" alt="Threshold Analysis">
            </div>
        </div>
        
        <h2>3. Độ quan trọng của đặc trưng</h2>
        <img src="evaluation/feature_importance.png" alt="Feature Importance">
        
        <h2>4. Giải thích mô hình (SHAP)</h2>
        <div class="container">
            <div class="chart">
                <h3>Tổng quan SHAP</h3>
                <img src="explanation/shap_summary.png" alt="SHAP Summary">
            </div>
            <div class="chart">
                <h3>Đặc trưng quan trọng nhất</h3>
                <img src="explanation/shap_bar.png" alt="SHAP Bar">
            </div>
        </div>
        
        <h2>5. Giải thích mẫu</h2>
        <div class="container">
            <div class="chart">
                <h3>Mẫu 1</h3>
                <img src="explanation/sample_1_force_plot.png" alt="Sample 1">
            </div>
            <div class="chart">
                <h3>Mẫu 2</h3>
                <img src="explanation/sample_2_force_plot.png" alt="Sample 2">
            </div>
        </div>
        
        <p><i>Báo cáo được tạo vào {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'model_report.html'), 'w') as f:
        f.write(html_content)
    
    logger.info(f"Đã tạo báo cáo HTML tại {os.path.join(output_dir, 'model_report.html')}")

if __name__ == "__main__":
    # Ví dụ sử dụng
    model_path = '/home/joe/python_Proj/test/privilege_detection_model_latest.pkl'
    features_path = '/home/joe/python_Proj/test/model_features_latest.pkl'
    data_path = '/home/joe/python_Proj/test/combined_data.csv'
    output_dir = '/home/joe/python_Proj/test/model_report'
    
    # Tạo báo cáo đầy đủ
    generate_model_report(model_path, features_path, data_path, output_dir)
