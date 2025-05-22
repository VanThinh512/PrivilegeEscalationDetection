"""
feature_engineering.py - Module bổ sung các đặc trưng mới cho hệ thống phát hiện leo thang đặc quyền
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

def extract_additional_features(df):
    """
    Bổ sung các đặc trưng mới vào DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame gốc cần bổ sung đặc trưng
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame với các đặc trưng mới
    """
    logger.info("Bắt đầu trích xuất đặc trưng mới")
    result_df = df.copy()
    
    # Đảm bảo có cột timestamp hoặc date
    date_col = None
    for col in ['timestamp', 'date', 'datetime']:
        if col in result_df.columns:
            date_col = col
            break
    
    if date_col:
        # Chuyển đổi cột ngày tháng sang định dạng datetime nếu cần
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
            try:
                # Thử nhiều định dạng khác nhau
                result_df[date_col] = pd.to_datetime(result_df[date_col], errors='coerce')
            except Exception as e:
                logger.warning(f"Không thể chuyển đổi cột {date_col} sang datetime: {e}")
        
        # Chỉ tiếp tục nếu chuyển đổi thành công
        if pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
            # 1. Thêm đặc trưng thời gian trong ngày
            result_df['hour'] = result_df[date_col].dt.hour
            result_df['time_of_day'] = pd.cut(
                result_df['hour'], 
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            # 2. Thêm đặc trưng ngày trong tuần
            result_df['day_of_week'] = result_df[date_col].dt.dayofweek
            result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
            
            # 3. Thêm đặc trưng thời gian làm việc
            result_df['is_business_hours'] = (
                (result_df['hour'] >= 8) & 
                (result_df['hour'] < 18) & 
                (~result_df['is_weekend'])
            ).astype(int)
            
            logger.info("Đã thêm các đặc trưng liên quan đến thời gian")
    
    # Thêm đặc trưng liên quan đến người dùng và lệnh 
    if 'user' in result_df.columns:
        # Nếu có nhiều bản ghi cho mỗi người dùng
        if len(result_df) > len(result_df['user'].unique()):
            # 4. Tần suất hoạt động của người dùng
            user_counts = result_df['user'].value_counts()
            result_df['user_activity_frequency'] = result_df['user'].map(user_counts)
            
            # 5. Thêm rank percentile cho tần suất người dùng
            result_df['user_freq_percentile'] = result_df['user_activity_frequency'].rank(pct=True)
            
            logger.info("Đã thêm các đặc trưng liên quan đến tần suất người dùng")
    
    # Thêm đặc trưng liên quan đến lệnh và tham số 
    if 'command' in result_df.columns:
        # 6. Đếm số lệnh đặc quyền cho mỗi người dùng
        if 'user' in result_df.columns:
            # Nhóm theo user
            user_cmd_counts = result_df.groupby('user')['command'].count()
            result_df['user_command_count'] = result_df['user'].map(user_cmd_counts)
            
            # 7. Đánh dấu người dùng có nhiều lệnh bất thường
            if 'is_suspicious_command' in result_df.columns:
                susp_cmd_by_user = result_df[result_df['is_suspicious_command'] > 0].groupby('user').size()
                result_df['suspicious_command_count'] = result_df['user'].map(susp_cmd_by_user).fillna(0)
                
                # Tỷ lệ lệnh đáng ngờ trên tổng số lệnh
                result_df['suspicious_command_ratio'] = result_df['suspicious_command_count'] / result_df['user_command_count']
                result_df['suspicious_command_ratio'] = result_df['suspicious_command_ratio'].fillna(0)
                
                logger.info("Đã thêm các đặc trưng liên quan đến tỷ lệ lệnh đáng ngờ")
    
    # Thêm đặc trưng thời gian giữa các sự kiện 
    if date_col and pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
        if 'user' in result_df.columns:
            # 8. Thời gian kể từ hoạt động trước của người dùng
            result_df = result_df.sort_values([date_col])
            result_df['prev_activity'] = result_df.groupby('user')[date_col].shift(1)
            result_df['time_since_last_activity'] = (result_df[date_col] - result_df['prev_activity']).dt.total_seconds() / 60
            
            # Xử lý giá trị NaN (hoạt động đầu tiên của mỗi người dùng)
            result_df['time_since_last_activity'] = result_df['time_since_last_activity'].fillna(-1)
            
            # 9. Đánh dấu hoạt động bất thường về thời gian
            time_threshold = result_df['time_since_last_activity'].quantile(0.95)  # ngưỡng 95%
            result_df['is_time_anomaly'] = (result_df['time_since_last_activity'] > time_threshold).astype(int)
            
            logger.info("Đã thêm các đặc trưng liên quan đến thời gian giữa các hoạt động")
    
    # Xử lý đặc trưng phân loại 
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if col not in [date_col, 'user'] and result_df[col].nunique() < 10:
            # 10. One-hot encoding cho các đặc trưng phân loại ít giá trị
            dummies = pd.get_dummies(result_df[col], prefix=col)
            result_df = pd.concat([result_df, dummies], axis=1)
    
    # Loại bỏ các cột tạm thời không cần
    cols_to_drop = ['prev_activity']
    result_df = result_df.drop(columns=[col for col in cols_to_drop if col in result_df.columns])
    
    logger.info(f"Đã trích xuất tổng cộng {len(result_df.columns) - len(df.columns)} đặc trưng mới")
    return result_df

def add_behavioral_features(df):
    """
    Thêm các đặc trưng liên quan đến hành vi người dùng
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame gốc
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame với các đặc trưng hành vi
    """
    logger.info("Bổ sung đặc trưng hành vi người dùng")
    result_df = df.copy()
    
    # Đảm bảo có đủ các cột cần thiết
    if 'user' not in result_df.columns or len(result_df) < 10:
        logger.warning("Không đủ dữ liệu hoặc không có cột user để thêm đặc trưng hành vi")
        return result_df
    
    # 1. Mẫu hành vi người dùng theo giờ
    if 'hour' in result_df.columns:
        # Tính tần suất hoạt động theo giờ cho mỗi người dùng
        user_hour_matrix = pd.crosstab(result_df['user'], result_df['hour'])
        user_hour_profiles = user_hour_matrix.div(user_hour_matrix.sum(axis=1), axis=0)
        
        # Tạo profile chuẩn (trung bình của tất cả người dùng)
        standard_profile = user_hour_profiles.mean()
        
        # Tính độ lệch của mỗi người dùng so với profile chuẩn
        for hour in range(24):
            if hour in user_hour_profiles.columns:
                col_name = f'hour_{hour}_deviation'
                # Ánh xạ độ lệch vào DataFrame gốc
                hour_deviations = user_hour_profiles[hour] - standard_profile[hour]
                result_df[col_name] = result_df['user'].map(hour_deviations)
        
        logger.info("Đã thêm đặc trưng độ lệch hành vi theo giờ")
    
    # 2. Phát hiện mẫu hành vi bất thường
    if 'command' in result_df.columns and result_df['command'].nunique() > 1:
        # Tạo ma trận tần suất lệnh cho mỗi người dùng
        cmd_freq = pd.crosstab(result_df['user'], result_df['command'])
        cmd_profiles = cmd_freq.div(cmd_freq.sum(axis=1), axis=0)
        
        # Tính entropy của phân phối lệnh cho mỗi người dùng (đa dạng lệnh)
        cmd_entropy = -(cmd_profiles * np.log2(cmd_profiles.replace(0, 1))).sum(axis=1)
        result_df['command_entropy'] = result_df['user'].map(cmd_entropy)
        
        # Người dùng có entropy thấp có thể bất thường (ít loại lệnh)
        entropy_threshold = cmd_entropy.quantile(0.1)  # ngưỡng 10% thấp nhất
        result_df['is_low_entropy_user'] = (result_df['command_entropy'] < entropy_threshold).astype(int)
        
        logger.info("Đã thêm đặc trưng entropy và phát hiện entropy thấp")
    
    # 3. Phát hiện hành vi phân cụm
    if 'date' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['date']):
        # Tạo đặc trưng khoảng thời gian giữa các chuỗi lệnh
        result_df = result_df.sort_values(['user', 'date'])
        result_df['next_cmd_time'] = result_df.groupby('user')['date'].shift(-1)
        result_df['time_to_next_cmd'] = (result_df['next_cmd_time'] - result_df['date']).dt.total_seconds() / 60
        
        # Đánh dấu các lệnh là một phần của "burst" (nhiều lệnh trong thời gian ngắn)
        burst_threshold = 1.0  # 1 phút
        result_df['is_in_burst'] = (result_df['time_to_next_cmd'] < burst_threshold).astype(int)
        
        # Đếm số lệnh trong mỗi burst
        result_df['burst_id'] = ((result_df['is_in_burst'] == 0) | (result_df['user'] != result_df['user'].shift())).cumsum()
        burst_sizes = result_df.groupby(['user', 'burst_id']).size()
        burst_lookup = burst_sizes.reset_index().rename(columns={0: 'burst_size'})
        
        # Ánh xạ kích thước burst vào DataFrame gốc
        result_df = result_df.merge(burst_lookup, on=['user', 'burst_id'], how='left')
        
        # Đánh dấu các burst lớn bất thường
        size_threshold = burst_sizes.quantile(0.9)
        result_df['is_large_burst'] = (result_df['burst_size'] > size_threshold).astype(int)
        
        logger.info("Đã thêm đặc trưng phát hiện burst commands")
    
    # Loại bỏ các cột tạm thời
    cols_to_drop = ['next_cmd_time', 'burst_id']
    result_df = result_df.drop(columns=[col for col in cols_to_drop if col in result_df.columns])
    
    logger.info(f"Đã bổ sung tổng cộng {len(result_df.columns) - len(df.columns)} đặc trưng hành vi")
    return result_df

def add_inter_feature_relations(df):
    """
    Thêm các đặc trưng kết hợp (interaction features)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame gốc
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame với các đặc trưng kết hợp
    """
    logger.info("Thêm các đặc trưng kết hợp (interaction features)")
    result_df = df.copy()
    
    # 1. Kết hợp đặc trưng thời gian và lệnh đáng ngờ
    if all(col in result_df.columns for col in ['is_weekend', 'is_suspicious_command']):
        result_df['weekend_suspicious_cmd'] = result_df['is_weekend'] * result_df['is_suspicious_command']
    
    if all(col in result_df.columns for col in ['is_business_hours', 'is_suspicious_command']):
        result_df['after_hours_suspicious_cmd'] = (1 - result_df['is_business_hours']) * result_df['is_suspicious_command']
    
    # 2. Kết hợp đặc trưng người dùng và hoạt động bất thường
    if all(col in result_df.columns for col in ['user_freq_percentile', 'is_suspicious_command']):
        # Người dùng ít hoạt động nhưng có lệnh đáng ngờ
        result_df['rare_user_suspicious_cmd'] = ((result_df['user_freq_percentile'] < 0.2) & 
                                                (result_df['is_suspicious_command'] > 0)).astype(int)
    
    # 3. Kết hợp đặc trưng lệnh đáng ngờ và tham số đáng ngờ
    if all(col in result_df.columns for col in ['is_suspicious_command', 'has_suspicious_args']):
        result_df['suspicious_cmd_and_args'] = result_df['is_suspicious_command'] * result_df['has_suspicious_args']
    
    # 4. Kết hợp nhiều đặc trưng bất thường
    anomaly_cols = [col for col in result_df.columns if 'suspicious' in col or 'abnormal' in col or 'anomaly' in col]
    if len(anomaly_cols) >= 2:
        # Số lượng các dấu hiệu bất thường
        result_df['total_anomaly_signals'] = result_df[anomaly_cols].sum(axis=1)
        
        # Đánh dấu các bản ghi có nhiều dấu hiệu bất thường
        result_df['multiple_anomaly_signals'] = (result_df['total_anomaly_signals'] >= 2).astype(int)
    
    logger.info(f"Đã thêm {len(result_df.columns) - len(df.columns)} đặc trưng kết hợp")
    return result_df

if __name__ == "__main__":
    # Kiểm tra chức năng của các hàm trên dữ liệu mẫu
    try:
        # Tạo dữ liệu mẫu
        data = {
            'user': ['user1', 'user2', 'user1', 'user3', 'user2'],
            'date': pd.to_datetime(['2023-01-01 09:30', '2023-01-01 10:15', 
                                    '2023-01-01 14:45', '2023-01-02 08:00',
                                    '2023-01-02 17:30']),
            'command': ['sudo', 'su', 'chmod', 'chown', 'dd'],
            'is_suspicious_command': [0, 0, 0, 0, 1],
            'has_suspicious_args': [0, 0, 0, 0, 1]
        }
        
        sample_df = pd.DataFrame(data)
        
        # Áp dụng các hàm
        enhanced_df = extract_additional_features(sample_df)
        behavioral_df = add_behavioral_features(enhanced_df)
        final_df = add_inter_feature_relations(behavioral_df)
        
        print(f"Số cột ban đầu: {len(sample_df.columns)}")
        print(f"Số cột sau khi bổ sung: {len(final_df.columns)}")
        print(f"Các cột mới: {set(final_df.columns) - set(sample_df.columns)}")
        
    except Exception as e:
        print(f"Lỗi khi thử nghiệm: {e}")
