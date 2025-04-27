import pandas as pd
import numpy as np

def preprocess_logon(file_path):
    try:
        # Thử đọc với dấu phẩy hoặc dấu chấm phẩy
        try:
            df = pd.read_csv(file_path, sep=';')
        except:
            df = pd.read_csv(file_path)
        
        print("Columns in logon DataFrame:", df.columns)
        
        # Xử lý các cột không cần thiết
        # Loại bỏ các cột không cần thiết: Unnamed: 5, Unnamed: 6, ...
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        # Kiểm tra và chuyển đổi cột thời gian
        time_col = None
        for col in ['date', 'timestamp', 'datetime']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError("Không tìm thấy cột thời gian trong file logon.csv")
        
        # Xem mẫu giá trị trước khi chuyển đổi
        print(f"Mẫu giá trị cột {time_col}:", df[time_col].head())
        
        # Chuyển đổi cột thời gian với nhiều định dạng khác nhau
        try:
            # Thử format DD/MM/YYYY HH:MM
            df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M', errors='coerce')
        except:
            try:
                # Thử format YYYY-MM-DD HH:MM:SS
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            except:
                print(f"Không thể chuyển đổi cột {time_col}. Sẽ tạo cột giờ nhân tạo.")
                # Nếu không chuyển đổi được, tạo giá trị nhân tạo
                df['hour'] = 12  # Giá trị mặc định giữa ngày
                df['is_after_hours'] = 0
                df['is_abnormal_logon'] = 0
                df['day_of_week'] = 2  # Giả định là thứ 3
                df['is_weekend'] = 0
                df['date'] = pd.Timestamp('2023-01-01')
                return df
        
        # Loại bỏ hàng có giá trị thời gian NaT
        df = df.dropna(subset=[time_col])
        
        # Tạo cột hour để phân tích thời gian
        df['hour'] = df[time_col].dt.hour
        
        # Tạo nhãn cho đăng nhập ngoài giờ (sau 18h hoặc trước 8h)
        df['is_after_hours'] = ((df['hour'] >= 18) | (df['hour'] < 8)).astype(int)
        
        # Kiểm tra cột 'activity'
        if 'activity' in df.columns:
            # Nếu có cột 'activity', tạo nhãn cho logon bất thường
            df['is_abnormal_logon'] = ((df['activity'] == 'Logon') & df['is_after_hours']).astype(int)
        else:
            df['is_abnormal_logon'] = df['is_after_hours']
        
        # Thêm cột ngày để phân tích
        df['date'] = df[time_col].dt.date
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
        
    except Exception as e:
        print(f"Lỗi xử lý file logon: {e}")
        # Tạo DataFrame giả với các cột cần thiết để tiếp tục xử lý
        dummy_df = pd.DataFrame({
            'user': ['dummy_user'],
            'hour': [12],
            'is_after_hours': [0],
            'is_abnormal_logon': [0],
            'date': [pd.Timestamp('2023-01-01').date()],
            'day_of_week': [2],
            'is_weekend': [0]
        })
        return dummy_df

def preprocess_privilege_commands(file_path):
    try:
        # Thử đọc với dấu phẩy hoặc dấu chấm phẩy
        try:
            df = pd.read_csv(file_path, sep=';')
        except:
            df = pd.read_csv(file_path)
        
        print("Columns in privilege commands DataFrame:", df.columns)
        
        # Xác định cột timestamp/datetime
        time_col = None
        for col in ['timestamp', 'datetime', 'date']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError("Không tìm thấy cột thời gian trong file privilege_commands")
        
        # Chuyển đổi cột thời gian thành datetime
        try:
            # Kiểm tra mẫu dữ liệu
            print(f"Mẫu giá trị cột {time_col}:", df[time_col].head())
            
            # Thử format DD/MM/YYYY HH:MM
            df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M', errors='coerce')
        except:
            try:
                # Thử format mặc định
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            except:
                print(f"Không thể chuyển đổi cột {time_col}. Tạo cột giờ nhân tạo.")
                df['hour'] = 12
                df['is_after_hours'] = 0
        
        # Loại bỏ hàng có giá trị thời gian NaT
        df = df.dropna(subset=[time_col])
        
        # Tạo cột hour để phân tích thời gian
        df['hour'] = df[time_col].dt.hour
        
        # Tạo nhãn cho lệnh thực hiện ngoài giờ
        df['is_after_hours'] = ((df['hour'] >= 18) | (df['hour'] < 8)).astype(int)
        
        # Tạo nhãn cho lệnh đặc quyền đáng ngờ
        key_col = 'key' if 'key' in df.columns else ('matched_key' if 'matched_key' in df.columns else None)
        cmd_col = 'command' if 'command' in df.columns else None
        args_col = 'args' if 'args' in df.columns else None
        
        if key_col:
            # Các lệnh đáng ngờ
            suspicious_keys = ['chmod_exec', 'chown_exec', 'sudo_exec', 'su_exec', 'find_exec', 'gcc_exec', 'vim_exec', 'nano_exec', 'ip_exec','python3_exec', 'getcap_exec', 'id_exec', 'ifconfig_exec']
            df['is_suspicious_command'] = df[key_col].apply(lambda x: 1 if x in suspicious_keys else 0)
        elif cmd_col:
            # Nếu không có key column, kiểm tra command
            suspicious_cmds = ['chmod', 'chown', 'sudo', 'su', 'find', 'gcc', 'vim', 'nano', 'ip','python3', 'getcap', 'id', 'ifconfig']
            df['is_suspicious_command'] = df[cmd_col].apply(lambda x: 1 if x in suspicious_cmds else 0)
        else:
            df['is_suspicious_command'] = 0
        
        # Kiểm tra các tham số đáng ngờ trong args
        if args_col:
            suspicious_args = ['4755', 'u+s', 'chmod+s', 'root:root', '-perm', '-04000', 
                               '/etc/shadow', '/etc/passwd', '/etc/sudoers', 'SUID', 'writable']
            
            df['has_suspicious_args'] = df[args_col].astype(str).apply(
                lambda x: 1 if any(arg in x for arg in suspicious_args) else 0)
        else:
            df['has_suspicious_args'] = 0
        
        # Tạo nhãn bất thường tổng hợp
        df['is_abnormal_activity'] = (
            (df['is_after_hours'] & (df['is_suspicious_command'] == 1)) | 
            (df['is_after_hours'] & (df['has_suspicious_args'] == 1)) |
            ((df['has_suspicious_args'] == 1) & (df['is_suspicious_command'] == 1))
        ).astype(int)
        
        # Thêm is_anomaly column để tương thích với phần sau
        df['is_anomaly'] = df['is_abnormal_activity']
        
        # Thêm cột ngày để phân tích
        df['date'] = df[time_col].dt.date
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    except Exception as e:
        print(f"Lỗi xử lý file privilege commands: {e}")
        return pd.DataFrame()

def combine_data(logon_df, privilege_df):
    """Kết hợp dữ liệu logon và privilege để tạo bộ dữ liệu tổng hợp"""
    # Kiểm tra DataFrame trước khi kết hợp
    if logon_df.empty and privilege_df.empty:
        print("Cả hai DataFrame đều trống, không thể kết hợp dữ liệu.")
        return pd.DataFrame()
    
    # Trường hợp chỉ có dữ liệu đặc quyền
    if logon_df.empty:
        print("Chỉ có dữ liệu đặc quyền, sử dụng nó làm dữ liệu kết hợp.")
        # Đảm bảo có cột is_anomaly
        if 'is_anomaly' not in privilege_df.columns:
            privilege_df['is_anomaly'] = privilege_df['is_abnormal_activity']
        return privilege_df
    
    # Trường hợp chỉ có dữ liệu đăng nhập
    if privilege_df.empty:
        print("Chỉ có dữ liệu đăng nhập, sử dụng nó làm dữ liệu kết hợp.")
        # Đảm bảo có cột is_anomaly
        if 'is_anomaly' not in logon_df.columns:
            logon_df['is_anomaly'] = logon_df['is_abnormal_logon']
        return logon_df
    
    # Trường hợp có cả hai dữ liệu
    try:
        # Tổng hợp dữ liệu logon
        logon_summary = logon_df.groupby(['user', 'date', 'hour']).agg({
            'is_abnormal_logon': 'sum',
            'is_weekend': 'max'
        }).reset_index()
        
        # Tổng hợp dữ liệu privilege
        priv_summary = privilege_df.groupby(['user', 'date', 'hour']).agg({
            'is_suspicious_command': 'sum',
            'has_suspicious_args': 'sum',
            'is_abnormal_activity': 'sum'
        }).reset_index()
        
        # Kết hợp hai bảng
        combined = pd.merge(logon_summary, priv_summary, 
                            on=['user', 'date', 'hour'], 
                            how='outer').fillna(0)
        
        # Tạo nhãn bất thường tổng thể
        combined['is_anomaly'] = (
            (combined['is_abnormal_logon'] > 0) | 
            (combined['is_abnormal_activity'] > 0) |
            ((combined['is_suspicious_command'] > 3) & (combined['is_weekend'] > 0))
        ).astype(int)
        
        return combined
    except Exception as e:
        print(f"Lỗi khi kết hợp dữ liệu: {e}")
        # Nếu kết hợp thất bại, trả về DataFrame của privilege
        if 'is_anomaly' not in privilege_df.columns:
            privilege_df['is_anomaly'] = privilege_df['is_abnormal_activity']
        return privilege_df

if __name__ == "__main__":
    # Đường dẫn đến các file
    logon_path = '/mnt/hgfs/KaggleData/NewKaggleData/logon.csv'
    privilege_path = '/mnt/hgfs/KaggleData/NewKaggleData/privilege_commands_2.csv'
    output_path = '/mnt/hgfs/KaggleData/NewKaggleData/combined_data.csv'
    
    # Tiền xử lý dữ liệu
    logon_df = preprocess_logon(logon_path)
    privilege_df = preprocess_privilege_commands(privilege_path)
    
    # In thông tin cơ bản về dữ liệu
    if not logon_df.empty:
        print(f"\nLogon data shape: {logon_df.shape}")
        print(f"Bất thường trong logon: {logon_df['is_abnormal_logon'].sum()}")
    
    if not privilege_df.empty:
        print(f"\nPrivilege commands data shape: {privilege_df.shape}")
        print(f"Lệnh đáng ngờ: {privilege_df['is_suspicious_command'].sum()}")
        print(f"Tham số đáng ngờ: {privilege_df['has_suspicious_args'].sum()}")
        print(f"Hoạt động bất thường: {privilege_df['is_abnormal_activity'].sum()}")
    
    # Kết hợp dữ liệu
    combined_df = combine_data(logon_df, privilege_df)
    
    if not combined_df.empty:
        print(f"\nCombined data shape: {combined_df.shape}")
        try:
            # Đảm bảo cột is_anomaly tồn tại
            if 'is_anomaly' not in combined_df.columns:
                combined_df['is_anomaly'] = 0
                if 'is_abnormal_activity' in combined_df.columns:
                    combined_df['is_anomaly'] = combined_df['is_abnormal_activity']
            
            print(f"Tổng số bất thường phát hiện: {combined_df['is_anomaly'].sum()}")
            
            # Lưu dữ liệu đã kết hợp
            combined_df.to_csv(output_path, index=False)
            print(f"\nDữ liệu đã được lưu vào {output_path}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu: {e}")
    else:
        print("Không thể kết hợp dữ liệu do lỗi trong quá trình xử lý")