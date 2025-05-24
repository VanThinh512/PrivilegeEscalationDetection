"""
data_enhancement.py - Module cải thiện dữ liệu cho hệ thống phát hiện leo thang đặc quyền
Cung cấp các phương pháp để tạo dữ liệu tổng hợp và mở rộng nguồn dữ liệu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import os
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_enhancement')

def generate_synthetic_logon_data(base_df=None, num_samples=1000, output_path=None):
    """
    Tạo dữ liệu đăng nhập giả lập (synthetic logon data)
    
    Parameters:
    -----------
    base_df : pandas.DataFrame, optional
        DataFrame gốc để lấy thông tin người dùng/máy chủ
    num_samples : int, default=1000
        Số lượng mẫu dữ liệu cần tạo
    output_path : str, optional
        Đường dẫn để lưu dữ liệu tổng hợp
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame chứa dữ liệu đăng nhập giả lập
    """
    logger.info(f"Bắt đầu tạo {num_samples} bản ghi đăng nhập giả lập")
    
    # Tạo danh sách người dùng nếu không có DataFrame gốc
    if base_df is None or 'user' not in base_df.columns:
        users = [f'user{i:02d}' for i in range(1, 11)] + ['admin', 'root', 'system']
    else:
        users = base_df['user'].unique().tolist()
    
    # Tạo danh sách máy chủ nếu không có DataFrame gốc
    if base_df is None or 'pc' not in base_df.columns:
        pcs = [f'pc{i:02d}' for i in range(1, 16)]
    else:
        pcs = base_df['pc'].unique().tolist() if 'pc' in base_df.columns else [f'pc{i:02d}' for i in range(1, 16)]
    
    # Tạo danh sách hoạt động
    activities = ['Logon', 'Logoff', 'Failed_logon']
    activity_probs = [0.45, 0.45, 0.1]  # 10% là đăng nhập thất bại
    
    # Tạo timestamps ngẫu nhiên trong khoảng 30 ngày gần đây
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(num_samples)]
    
    # Tạo dữ liệu
    synthetic_data = {
        'id': range(1, num_samples + 1),
        'date': [ts.strftime('%d/%m/%Y %H:%M') for ts in timestamps],
        'user': np.random.choice(users, num_samples, p=None),
        'pc': np.random.choice(pcs, num_samples, p=None),
        'activity': np.random.choice(activities, num_samples, p=activity_probs),
    }
    
    # Tạo dữ liệu bất thường (10% dữ liệu)
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.1), replace=False)
    
    # Chuyển thành DataFrame
    df = pd.DataFrame(synthetic_data)
    
    # Thêm một số bất thường: đăng nhập vào giờ muộn
    for idx in anomaly_indices[:len(anomaly_indices)//2]:
        hour = np.random.randint(22, 24) if np.random.random() < 0.5 else np.random.randint(0, 5)
        minute = np.random.randint(0, 60)
        current_date = df.loc[idx, 'date'].split()[0]
        df.loc[idx, 'date'] = f"{current_date} {hour:02d}:{minute:02d}"
    
    # Thêm một số bất thường: đăng nhập vào cuối tuần
    for idx in anomaly_indices[len(anomaly_indices)//2:]:
        # Chuyển sang ngày cuối tuần (thứ 7 hoặc chủ nhật)
        ts = timestamps[idx]
        days_to_add = 5 - ts.weekday() if ts.weekday() < 5 else 0
        weekend_date = (ts + timedelta(days=days_to_add)).strftime('%d/%m/%Y')
        hour = np.random.randint(9, 18)
        minute = np.random.randint(0, 60)
        df.loc[idx, 'date'] = f"{weekend_date} {hour:02d}:{minute:02d}"
    
    logger.info(f"Đã tạo {len(df)} bản ghi đăng nhập với {len(anomaly_indices)} bất thường")
    
    # Lưu dữ liệu nếu cần
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu dữ liệu đăng nhập giả lập vào {output_path}")
    
    return df

def generate_synthetic_privilege_commands(base_df=None, num_samples=500, output_path=None):
    """
    Tạo dữ liệu lệnh đặc quyền giả lập
    
    Parameters:
    -----------
    base_df : pandas.DataFrame, optional
        DataFrame gốc để lấy thông tin người dùng/lệnh
    num_samples : int, default=500
        Số lượng mẫu dữ liệu cần tạo
    output_path : str, optional
        Đường dẫn để lưu dữ liệu tổng hợp
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame chứa dữ liệu lệnh đặc quyền giả lập
    """
    logger.info(f"Bắt đầu tạo {num_samples} bản ghi lệnh đặc quyền giả lập")
    
    # Định nghĩa người dùng
    if base_df is None or 'user' not in base_df.columns:
        users = [f'user{i:02d}' for i in range(1, 11)] + ['admin', 'root', 'system']
    else:
        users = base_df['user'].unique().tolist()
    
    # Định nghĩa các lệnh đặc quyền thông thường và các lệnh nguy hiểm
    normal_commands = [
        'sudo', 'su', 'passwd', 'chown', 'chmod', 'usermod', 'groupadd', 'visudo'
    ]
    
    suspicious_commands = [
        'sudo bash', 'sudo su -', 'sudo -i', 'pkexec', 'gksudo', 'setuid', 
        'setcap', 'chattr', 'dd', 'curl | bash', 'wget | bash',
        'cat /etc/passwd', 'cat /etc/shadow', 'cat /etc/crontab',
        "python3 -c 'import os; os.setuid(0); os.system(\"/bin/sh\")'",'find / -perm -4000'
    ]
    
    # Định nghĩa đối số bình thường và đáng ngờ
    normal_args = [
        '-l', '-a', '-v', '-h', '--help', '-R', '-r', '-p', '-g',
        '/etc/passwd', '/var/log', '/var/www', '/opt', '/usr/local/bin'
    ]
    
    suspicious_args = [
        '/etc/shadow', '/etc/sudoers', '/*', '/root/.ssh', '/dev/sda', 
        'if=/dev/zero', 'chmod u+s', 'chmod 4755', '--payload', '--exploit'
    ]
    
    # Tạo timestamps ngẫu nhiên trong khoảng 30 ngày gần đây
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(num_samples)]
    
    # Tạo dữ liệu
    data = []
    for i in range(num_samples):
        # 80% là lệnh bình thường, 20% là lệnh đáng ngờ
        if np.random.random() < 0.8:
            command = np.random.choice(normal_commands)
            # 90% là đối số bình thường, 10% là đáng ngờ
            args = np.random.choice(normal_args if np.random.random() < 0.9 else suspicious_args)
        else:
            command = np.random.choice(suspicious_commands)
            # 50% là đối số bình thường, 50% là đáng ngờ
            args = np.random.choice(normal_args if np.random.random() < 0.5 else suspicious_args)
        
        # Tạo key từ command và args
        key = f"{command}_{args.replace('/', '_')}"
        
        # Thêm vào dữ liệu
        data.append({
            'timestamp': timestamps[i].strftime('%d/%m/%Y %H:%M'),
            'user': np.random.choice(users),
            'command': command,
            'args': args,
            'key': key
        })
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Lưu dữ liệu nếu cần
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu dữ liệu lệnh đặc quyền giả lập vào {output_path}")
    
    return df

def balance_data_with_smote(X, y, random_state=42):
    """
    Cân bằng dữ liệu sử dụng SMOTE
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Dữ liệu đặc trưng
    y : pandas.Series
        Nhãn
    random_state : int, default=42
        Giá trị khởi tạo cho quá trình tạo dữ liệu ngẫu nhiên
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    logger.info("Áp dụng SMOTE để cân bằng dữ liệu")
    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Log thông tin
        original_pos = sum(y == 1)
        original_neg = sum(y == 0)
        new_pos = sum(y_resampled == 1)
        new_neg = sum(y_resampled == 0)
        
        logger.info(f"Phân phối ban đầu: {original_neg} negative, {original_pos} positive")
        logger.info(f"Phân phối sau SMOTE: {new_neg} negative, {new_pos} positive")
        
        return X_resampled, y_resampled
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng SMOTE: {e}")
        return X, y

def collect_external_logs(logs_dir=None):
    """
    Thu thập thêm log từ các nguồn bên ngoài
    
    Parameters:
    -----------
    logs_dir : str, optional
        Thư mục chứa các file log bổ sung
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame chứa dữ liệu log đã thu thập
    """
    logger.info("Thu thập log từ các nguồn bên ngoài")
    
    if logs_dir is None or not os.path.exists(logs_dir):
        logger.warning(f"Thư mục {logs_dir} không tồn tại. Trả về DataFrame rỗng.")
        return pd.DataFrame()
    
    combined_logs = []
    
    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, filename)
        
        # Chỉ xử lý file, không xử lý thư mục
        if not os.path.isfile(file_path):
            continue
        
        try:
            # Xử lý theo loại file
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                combined_logs.append(df)
                logger.info(f"Đã đọc {len(df)} bản ghi từ {filename}")
            elif filename.endswith('.json'):
                df = pd.read_json(file_path)
                combined_logs.append(df)
                logger.info(f"Đã đọc {len(df)} bản ghi từ {filename}")
            elif filename.endswith('.log') or filename.endswith('.txt'):
                # Đơn giản hóa: giả định log có dạng "timestamp,user,action"
                logs = []
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            logs.append({
                                'timestamp': parts[0],
                                'user': parts[1],
                                'action': parts[2],
                                'source': filename
                            })
                if logs:
                    df = pd.DataFrame(logs)
                    combined_logs.append(df)
                    logger.info(f"Đã đọc {len(logs)} bản ghi từ {filename}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {filename}: {e}")
    
    # Kết hợp tất cả log
    if combined_logs:
        result = pd.concat(combined_logs, ignore_index=True)
        logger.info(f"Đã thu thập tổng cộng {len(result)} bản ghi log")
        return result
    else:
        logger.warning("Không tìm thấy log phù hợp. Trả về DataFrame rỗng.")
        return pd.DataFrame()

def upsample_minority_class(df, class_col='is_anomaly', random_state=42):
    """
    Tăng số lượng mẫu cho lớp thiểu số bằng phương pháp upsampling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame gốc
    class_col : str, default='is_anomaly'
        Tên cột chứa nhãn lớp
    random_state : int, default=42
        Giá trị khởi tạo cho quá trình tạo dữ liệu ngẫu nhiên
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame đã cân bằng
    """
    logger.info("Tăng số lượng mẫu cho lớp thiểu số")
    
    # Tách dữ liệu thành hai nhóm: lớp đa số và lớp thiểu số
    majority = df[df[class_col] == 0]
    minority = df[df[class_col] == 1]
    
    logger.info(f"Phân phối ban đầu: {len(majority)} mẫu lớp đa số, {len(minority)} mẫu lớp thiểu số")
    
    # Nếu lớp thiểu số có ít hơn một nửa số mẫu lớp đa số, tăng số lượng mẫu
    if len(minority) < len(majority) / 2:
        # Tăng số lượng mẫu lớp thiểu số lên bằng lớp đa số
        minority_upsampled = resample(
            minority,
            replace=True,  # Cho phép lấy mẫu với replacement
            n_samples=len(majority),
            random_state=random_state
        )
        
        # Kết hợp lại
        df_balanced = pd.concat([majority, minority_upsampled])
        
        logger.info(f"Phân phối sau upsampling: {len(majority)} mẫu lớp đa số, {len(minority_upsampled)} mẫu lớp thiểu số")
        return df_balanced
    else:
        logger.info("Dữ liệu đã đủ cân bằng, không cần upsampling")
        return df

if __name__ == "__main__":
    # Thử nghiệm các hàm
    output_dir = '/home/joe/python_Proj/test/enhanced_data'
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo dữ liệu đăng nhập giả lập
    logon_df = generate_synthetic_logon_data(
        num_samples=1000,
        output_path=os.path.join(output_dir, 'synthetic_logon.csv')
    )
    
    # Tạo dữ liệu lệnh đặc quyền giả lập
    privilege_df = generate_synthetic_privilege_commands(
        num_samples=500,
        output_path=os.path.join(output_dir, 'synthetic_privilege.csv')
    )
    
    print(f"Đã tạo {len(logon_df)} bản ghi đăng nhập và {len(privilege_df)} bản ghi lệnh đặc quyền")
    print(f"Dữ liệu đã được lưu vào thư mục {output_dir}")
