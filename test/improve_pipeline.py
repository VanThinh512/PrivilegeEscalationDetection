"""
improve_pipeline.py - Script tổng hợp cải tiến cho hệ thống phát hiện leo thang đặc quyền
Kết hợp tất cả các module cải tiến: data_enhancement, feature_engineering, model_monitor và model_evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('improvement_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('improve_pipeline')

# Import các module cải tiến
try:
    from data_enhancement import (
        generate_synthetic_logon_data, 
        generate_synthetic_privilege_commands,
        balance_data_with_smote,
        collect_external_logs,
        upsample_minority_class
    )
    
    from feature_engineering import (
        extract_additional_features,
        add_behavioral_features,
        add_inter_feature_relations
    )
    
    from model_monitor import (
        ModelMonitor,
        retrain_pipeline
    )
    
    from model_evaluation import (
        generate_evaluation_report,
        explain_model_predictions,
        generate_model_report
    )
    
    logger.info("Đã import thành công các module cải tiến")
except ImportError as e:
    logger.error(f"Lỗi khi import module: {e}")
    logger.error("Đảm bảo bạn đã cài đặt tất cả các module cần thiết")
    sys.exit(1)

def print_banner():
    """In banner thông tin khi khởi chạy"""
    banner = """
    ███████╗███████╗ ██████╗██╗   ██╗██████╗ ███████╗
    ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝
    ███████╗█████╗  ██║     ██║   ██║██████╔╝█████╗  
    ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  
    ███████║███████╗╚██████╗╚██████╔╝██║  ██║███████╗
    ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
                                                     
    === Privilege Escalation Detection Improvement Pipeline ===
    """
    print(banner)

def setup_environment(base_dir):
    """
    Thiết lập cấu trúc thư mục cho dự án
    
    Parameters:
    -----------
    base_dir : str
        Thư mục gốc
    
    Returns:
    --------
    dict
        Các đường dẫn quan trọng
    """
    # Tạo các thư mục cần thiết
    paths = {
        'raw_data': os.path.join(base_dir, 'raw_data'),
        'enhanced_data': os.path.join(base_dir, 'enhanced_data'),
        'models': os.path.join(base_dir, 'models'),
        'reports': os.path.join(base_dir, 'reports'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Đã tạo thư mục: {path}")
    
    return paths

def enhance_data(paths, args):
    """
    Cải thiện dữ liệu: tạo thêm dữ liệu, cân bằng dữ liệu
    
    Parameters:
    -----------
    paths : dict
        Các đường dẫn quan trọng
    args : argparse.Namespace
        Tham số dòng lệnh
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame đã được cải thiện
    """
    logger.info("=== BẮT ĐẦU CẢI THIỆN DỮ LIỆU ===")
    
    # 1. Đọc dữ liệu gốc
    data_file = args.input_data
    if not os.path.exists(data_file):
        logger.error(f"Không tìm thấy file dữ liệu: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    logger.info(f"Đã đọc dữ liệu gốc, kích thước: {df.shape}")
    
    # 2. Tạo dữ liệu giả lập nếu cần
    if args.synthetic_data:
        # Tạo dữ liệu đăng nhập
        if args.generate_logon:
            logon_df = generate_synthetic_logon_data(
                base_df=df if 'user' in df.columns else None,
                num_samples=args.synthetic_samples,
                output_path=os.path.join(paths['enhanced_data'], 'synthetic_logon.csv')
            )
            logger.info(f"Đã tạo {len(logon_df)} bản ghi đăng nhập giả lập")
        
        # Tạo dữ liệu lệnh đặc quyền
        if args.generate_privilege:
            privilege_df = generate_synthetic_privilege_commands(
                base_df=df if 'command' in df.columns else None,
                num_samples=args.synthetic_samples // 2,
                output_path=os.path.join(paths['enhanced_data'], 'synthetic_privilege.csv')
            )
            logger.info(f"Đã tạo {len(privilege_df)} bản ghi lệnh đặc quyền giả lập")
    
    # 3. Thu thập thêm dữ liệu từ các nguồn khác nếu có
    if args.external_logs and os.path.exists(args.external_logs):
        external_df = collect_external_logs(args.external_logs)
        if not external_df.empty:
            logger.info(f"Đã thu thập {len(external_df)} bản ghi từ các nguồn bên ngoài")
            
            # Kết hợp nếu có thể
            # Đơn giản hóa: thêm vào dưới cùng nếu có cấu trúc tương tự
            if set(external_df.columns).intersection(set(df.columns)):
                common_cols = list(set(external_df.columns).intersection(set(df.columns)))
                external_df = external_df[common_cols]
                df = pd.concat([df, external_df], ignore_index=True)
                logger.info(f"Đã kết hợp dữ liệu bên ngoài, kích thước mới: {df.shape}")
    
    # 4. Cân bằng dữ liệu
    if 'is_anomaly' in df.columns:
        label_col = 'is_anomaly'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        logger.warning("Không tìm thấy cột nhãn để cân bằng dữ liệu")
        label_col = None
    
    if label_col:
        # Kiểm tra tỷ lệ mất cân bằng
        positive_ratio = df[label_col].mean()
        logger.info(f"Tỷ lệ nhãn dương: {positive_ratio*100:.2f}%")
        
        if positive_ratio < 0.3 or positive_ratio > 0.7:
            # Cân bằng bằng upsampling nếu mất cân bằng
            df = upsample_minority_class(df, class_col=label_col)
            logger.info(f"Đã cân bằng dữ liệu bằng upsampling, kích thước mới: {df.shape}")
    
    # 5. Lưu dữ liệu đã cải thiện
    enhanced_data_file = os.path.join(paths['enhanced_data'], 'enhanced_data.csv')
    df.to_csv(enhanced_data_file, index=False)
    logger.info(f"Đã lưu dữ liệu đã cải thiện vào: {enhanced_data_file}")
    
    logger.info("=== HOÀN THÀNH CẢI THIỆN DỮ LIỆU ===")
    return df, enhanced_data_file

def engineer_features(df, paths, args):
    """
    Bổ sung các đặc trưng mới
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame gốc
    paths : dict
        Các đường dẫn quan trọng
    args : argparse.Namespace
        Tham số dòng lệnh
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame với các đặc trưng mới
    """
    logger.info("=== BẮT ĐẦU BỔ SUNG ĐẶC TRƯNG ===")
    
    # 1. Trích xuất các đặc trưng cơ bản
    df_with_features = extract_additional_features(df)
    logger.info(f"Đã bổ sung các đặc trưng cơ bản, số cột mới: {len(df_with_features.columns) - len(df.columns)}")
    
    # 2. Bổ sung đặc trưng hành vi
    df_with_behaviors = add_behavioral_features(df_with_features)
    logger.info(f"Đã bổ sung các đặc trưng hành vi, số cột mới: {len(df_with_behaviors.columns) - len(df_with_features.columns)}")
    
    # 3. Bổ sung đặc trưng kết hợp
    final_df = add_inter_feature_relations(df_with_behaviors)
    logger.info(f"Đã bổ sung các đặc trưng kết hợp, số cột mới: {len(final_df.columns) - len(df_with_behaviors.columns)}")
    
    # 4. Lưu dữ liệu với đặc trưng mới
    features_data_file = os.path.join(paths['enhanced_data'], 'features_enriched_data.csv')
    final_df.to_csv(features_data_file, index=False)
    logger.info(f"Đã lưu dữ liệu với đặc trưng mới vào: {features_data_file}")
    
    # In ra các cột mới
    new_columns = set(final_df.columns) - set(df.columns)
    logger.info(f"Tổng số đặc trưng mới: {len(new_columns)}")
    logger.info(f"Danh sách đặc trưng mới: {', '.join(new_columns)}")
    
    logger.info("=== HOÀN THÀNH BỔ SUNG ĐẶC TRƯNG ===")
    return final_df, features_data_file

def prepare_data_for_model(df, model_features_path):
    """
    Chuẩn bị dữ liệu để phù hợp với định dạng mô hình
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame cần chuẩn bị
    model_features_path : str
        Đường dẫn đến file thông tin đặc trưng của mô hình
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame đã chuẩn bị phù hợp với mô hình
    """
    logger.info("Chuẩn bị dữ liệu cho mô hình")
    
    # Tải thông tin đặc trưng của mô hình
    try:
        features_info = joblib.load(model_features_path)
        feature_names = features_info.get('feature_names', [])
        
        # Kiểm tra các cột thiếu
        missing_cols = [col for col in feature_names if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Thiếu {len(missing_cols)} cột trong dữ liệu mới. Đang thêm các cột thiếu với giá trị 0.")
            # Tạo các cột thiếu với giá trị 0
            for col in missing_cols:
                df[col] = 0
        
        # Lọc và sắp xếp các cột theo đúng thứ tự
        result_df = df[feature_names].copy()
        
        logger.info(f"Đã chuẩn bị dữ liệu với {len(feature_names)} đặc trưng")
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị dữ liệu: {e}")
        # Trả về DataFrame gốc nếu có lỗi
        return df

def evaluate_model(model_path, features_path, data_file, paths, args):
    """
    Đánh giá và giải thích mô hình
    
    Parameters:
    -----------
    model_path : str
        Đường dẫn đến file mô hình
    features_path : str
        Đường dẫn đến file thông tin đặc trưng
    data_file : str
        Đường dẫn đến file dữ liệu
    paths : dict
        Các đường dẫn quan trọng
    args : argparse.Namespace
        Tham số dòng lệnh
    
    Returns:
    --------
    dict
        Kết quả đánh giá
    """
    logger.info("=== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH ===")
    
    # Tạo thư mục báo cáo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(paths['reports'], f'model_report_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    # Đọc dữ liệu
    try:
        data = pd.read_csv(data_file)
        logger.info(f"Đã đọc dữ liệu từ {data_file}, kích thước: {data.shape}")
        
        if args.label_column not in data.columns:
            logger.error(f"Không tìm thấy cột nhãn '{args.label_column}' trong dữ liệu")
            return report_dir
        
        # Tách dữ liệu và nhãn
        X = data.drop(columns=[args.label_column])
        y = data[args.label_column]
        
        # Chuẩn bị dữ liệu cho mô hình
        X_prepared = prepare_data_for_model(X, features_path)
        
        # Đánh giá toàn diện nếu được yêu cầu
        if args.comprehensive:
            try:
                # Tải mô hình
                model = joblib.load(model_path)
                
                # Chuẩn bị dữ liệu kiểm tra
                from sklearn.model_selection import train_test_split
                _, X_test, _, y_test = train_test_split(X_prepared, y, test_size=0.3, random_state=42)
                
                # Tạo thư mục báo cáo đánh giá
                eval_dir = os.path.join(report_dir, 'evaluation')
                os.makedirs(eval_dir, exist_ok=True)
                
                # Gọi hàm đánh giá từ model_evaluation
                from model_evaluation import generate_evaluation_report, explain_model_predictions
                
                # Tạo báo cáo đánh giá
                results = generate_evaluation_report(model, X_test, y_test, eval_dir)
                
                # Giải thích mô hình
                explain_dir = os.path.join(report_dir, 'explanation')
                os.makedirs(explain_dir, exist_ok=True)
                
                # Lấy tối đa 100 mẫu để giải thích
                sample_size = min(100, len(X_test))
                explain_model_predictions(model, X_test.iloc[:sample_size], list(X_test.columns), explain_dir)
                
                # Tạo báo cáo HTML
                from model_evaluation import create_html_report
                create_html_report(results, report_dir)
                
                logger.info(f"Đã tạo báo cáo đánh giá toàn diện tại: {report_dir}")
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo báo cáo đánh giá: {e}")
                logger.error("Không thể tạo báo cáo đánh giá toàn diện")
        
        # Kiểm tra model drift nếu được yêu cầu
        if args.check_drift:
            try:
                monitor = ModelMonitor(model_path, features_path)
                
                # Kiểm tra drift với dữ liệu đã chuẩn bị
                drift_detected = monitor.check_drift(X_prepared, y)
                
                if drift_detected:
                    logger.warning("Phát hiện model drift! Nên huấn luyện lại mô hình.")
                else:
                    logger.info("Mô hình vẫn ổn định, không cần huấn luyện lại.")
                
                # Vẽ biểu đồ xu hướng
                monitor.plot_metrics_trend(os.path.join(report_dir, 'metrics_trend.png'))
                
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra model drift: {e}")
        
    except Exception as e:
        logger.error(f"Lỗi tổng quát trong quá trình đánh giá mô hình: {e}")
        return report_dir
    
    logger.info("=== HOÀN THÀNH ĐÁNH GIÁ MÔ HÌNH ===")
    return report_dir

def main():
    """Hàm chính thực thi pipeline cải tiến"""
    print_banner()
    
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Pipeline cải tiến cho hệ thống phát hiện leo thang đặc quyền')
    
    parser.add_argument('--input-data', '-i', required=True,
                        help='Đường dẫn đến file dữ liệu đầu vào')
    parser.add_argument('--model-path', '-m',
                        default='/home/joe/python_Proj/test/privilege_detection_model_latest.pkl',
                        help='Đường dẫn đến file mô hình')
    parser.add_argument('--features-path', '-f',
                        default='/home/joe/python_Proj/test/model_features_latest.pkl',
                        help='Đường dẫn đến file thông tin đặc trưng')
    parser.add_argument('--output-dir', '-o',
                        default='/home/joe/python_Proj/test/improved',
                        help='Thư mục đầu ra')
    
    # Tham số cải thiện dữ liệu
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Tạo dữ liệu giả lập')
    parser.add_argument('--synthetic-samples', type=int, default=500,
                        help='Số lượng mẫu giả lập cần tạo')
    parser.add_argument('--generate-logon', action='store_true',
                        help='Tạo dữ liệu đăng nhập giả lập')
    parser.add_argument('--generate-privilege', action='store_true',
                        help='Tạo dữ liệu lệnh đặc quyền giả lập')
    parser.add_argument('--external-logs',
                        help='Thư mục chứa log bên ngoài để bổ sung')
    
    # Tham số đánh giá mô hình
    parser.add_argument('--comprehensive', action='store_true',
                        help='Tạo báo cáo đánh giá toàn diện')
    parser.add_argument('--check-drift', action='store_true',
                        help='Kiểm tra model drift')
    parser.add_argument('--label-column', default='is_anomaly',
                        help='Tên cột nhãn')
    
    # Tùy chọn pipeline
    parser.add_argument('--skip-data-enhancement', action='store_true',
                        help='Bỏ qua bước cải thiện dữ liệu')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Bỏ qua bước bổ sung đặc trưng')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Bỏ qua bước đánh giá mô hình')
    
    args = parser.parse_args()
    
    try:
        # Thiết lập môi trường
        paths = setup_environment(args.output_dir)
        
        # Pipeline cải tiến
        data_file = args.input_data
        
        # Bước 1: Cải thiện dữ liệu
        if not args.skip_data_enhancement:
            df, data_file = enhance_data(paths, args)
        else:
            logger.info("Bỏ qua bước cải thiện dữ liệu")
            df = pd.read_csv(data_file)
        
        # Bước 2: Bổ sung đặc trưng
        if not args.skip_feature_engineering:
            df, data_file = engineer_features(df, paths, args)
        else:
            logger.info("Bỏ qua bước bổ sung đặc trưng")
        
        # Bước 3: Đánh giá mô hình
        if not args.skip_evaluation:
            report_dir = evaluate_model(args.model_path, args.features_path, data_file, paths, args)
            logger.info(f"Báo cáo đánh giá đã được lưu tại: {report_dir}")
        else:
            logger.info("Bỏ qua bước đánh giá mô hình")
        
        logger.info("Pipeline cải tiến đã hoàn thành!")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Chạy hàm chính
    main()
