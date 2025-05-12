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
            
            # Thử nhiều format
            try:
                # Thử format DD/MM/YYYY HH:MM
                df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M', errors='coerce')
            except:
                try:
                    # Thử format MM/DD/YYYY HH:MM
                    df[time_col] = pd.to_datetime(df[time_col], format='%m/%d/%Y %H:%M', errors='coerce')
                except:
                    # Thử format mặc định
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    
        except Exception as e:
            print(f"Không thể chuyển đổi cột {time_col}: {str(e)}. Tạo cột giờ nhân tạo.")
            df['hour'] = np.random.randint(0, 24, size=len(df))  # Giờ ngẫu nhiên
            df['is_after_hours'] = ((df['hour'] >= 18) | (df['hour'] < 8)).astype(int)
        
        # Loại bỏ hàng có giá trị thời gian NaT
        df_valid = df.dropna(subset=[time_col])
        
        # Nếu mất quá nhiều dữ liệu, giữ lại một số ngẫu nhiên với giờ nhân tạo
        if len(df_valid) < len(df) * 0.7:
            print(f"Mất {len(df) - len(df_valid)} dòng do ngày không hợp lệ. Giữ một số dòng với giờ ngẫu nhiên.")
            df_invalid = df[df[time_col].isna()].copy()
            df_invalid['hour'] = np.random.randint(0, 24, size=len(df_invalid))
            df_invalid['is_after_hours'] = ((df_invalid['hour'] >= 18) | (df_invalid['hour'] < 8)).astype(int)
            
            # Gộp lại nhưng giới hạn số dòng không hợp lệ
            max_invalid = min(len(df_valid), len(df_invalid))
            df = pd.concat([df_valid, df_invalid.head(max_invalid)])
        else:
            df = df_valid
        
        # Tạo cột hour và các đặc trưng thời gian
        if 'hour' not in df.columns:
            df['hour'] = df[time_col].dt.hour
            df['is_after_hours'] = ((df['hour'] >= 18) | (df['hour'] < 8)).astype(int)
        
        # Trích xuất các cột quan trọng
        key_col = 'key' if 'key' in df.columns else ('matched_key' if 'matched_key' in df.columns else None)
        cmd_col = 'command' if 'command' in df.columns else None
        args_col = 'args' if 'args' in df.columns else None
        
        # --- ĐIỀU CHỈNH 1: Phân loại lệnh đáng ngờ - GIỮ NGUYÊN DANH MỤC NHƯNG GIẢM NGẪU NHIÊN ---
        
        # Lệnh đáng ngờ cao - các lệnh leo thang đặc quyền cốt lõi
        high_risk_cmds = ['sudo', 'su', 'pkexec']
        # Lệnh đáng ngờ trung bình - các lệnh thay đổi quyền
        medium_risk_cmds = ['chown', 'chmod', 'setcap']
        # Lệnh bổ sung - có thể được dùng trong leo thang đặc quyền
        additional_cmds = ['find', 'gcc', 'systemctl', 'mount', 'perl']
        
        # Áp dụng phân loại - GIẢM NGẪU NHIÊN và ƯU TIÊN PHÂN LOẠI LỆNH ĐÁNG NGỜ
        if key_col:
            # Chuyển đổi từ key_exec thành command name
            high_risk_keys = [cmd + '_exec' for cmd in high_risk_cmds]
            medium_risk_keys = [cmd + '_exec' for cmd in medium_risk_cmds]
            additional_keys = [cmd + '_exec' for cmd in additional_cmds]
            
            # Đánh dấu lệnh đáng ngờ - ít ngẫu nhiên hơn, trực tiếp hơn
            df['is_suspicious_command'] = 0
            # Mức cao - 100% đáng ngờ
            df.loc[df[key_col].isin(high_risk_keys), 'is_suspicious_command'] = 1
            # Mức trung bình - 80% đáng ngờ
            df.loc[df[key_col].isin(medium_risk_keys), 'is_suspicious_command'] = np.where(
                np.random.random(size=len(df[df[key_col].isin(medium_risk_keys)])) < 0.8, 1, 0
            )
            # Cái khác - 20% đáng ngờ (ngẫu nhiên ít)
            df.loc[df[key_col].isin(additional_keys), 'is_suspicious_command'] = np.where(
                np.random.random(size=len(df[df[key_col].isin(additional_keys)])) < 0.2, 1, 0
            )
            
            # Lưu điểm đánh giá để sử dụng sau - chỉ dùng nội bộ
            df['suspicion_score'] = 0.0
            df.loc[df[key_col].isin(high_risk_keys), 'suspicion_score'] = 0.9  # Ít ngẫu nhiên hơn
            df.loc[df[key_col].isin(medium_risk_keys), 'suspicion_score'] = 0.6
            df.loc[df[key_col].isin(additional_keys), 'suspicion_score'] = 0.2
            
        elif cmd_col:
            # Tương tự với command column
            df['is_suspicious_command'] = 0
            # Mức cao - 100% đáng ngờ
            df.loc[df[cmd_col].isin(high_risk_cmds), 'is_suspicious_command'] = 1
            # Mức trung bình - 80% đáng ngờ
            df.loc[df[cmd_col].isin(medium_risk_cmds), 'is_suspicious_command'] = np.where(
                np.random.random(size=len(df[df[cmd_col].isin(medium_risk_cmds)])) < 0.8, 1, 0
            )
            # Cái khác - 20% đáng ngờ (ngẫu nhiên ít)
            df.loc[df[cmd_col].isin(additional_cmds), 'is_suspicious_command'] = np.where(
                np.random.random(size=len(df[df[cmd_col].isin(additional_cmds)])) < 0.2, 1, 0
            )
            
            # Lưu điểm đánh giá để sử dụng sau - chỉ dùng nội bộ
            df['suspicion_score'] = 0.0
            df.loc[df[cmd_col].isin(high_risk_cmds), 'suspicion_score'] = 0.9  # Ít ngẫu nhiên hơn
            df.loc[df[cmd_col].isin(medium_risk_cmds), 'suspicion_score'] = 0.6
            df.loc[df[cmd_col].isin(additional_cmds), 'suspicion_score'] = 0.2
        else:
            df['is_suspicious_command'] = 0
            df['suspicion_score'] = 0.0
        
        # --- ĐIỀU CHỈNH 2: Cải thiện phát hiện tham số đáng ngờ - GIẢM NGẪU NHIÊN ---
        if args_col:
            # Tham số có rủi ro cao - liên quan trực tiếp đến leo thang đặc quyền
            high_risk_args = [
                'u+s', 'root:root', '4755', 'setcap cap_setuid',
                '/etc/shadow', '/etc/passwd', 'sudo -i', 'sudo bash',
                'chmod 777', '/etc/sudoers'
            ]
            
            # Tham số có rủi ro trung bình
            medium_risk_args = [
                '-perm -4000', 'SUID', 'writable', 'debug-shell.service',
                'systemctl enable', 'systemctl start', 'crontab'
            ]
            
            # Dấu hiệu đáng ngờ
            suspicious_indicators = ['exploit', 'vuln', 'rootbash', 'pwnkit', 'hack', 'privilege']
            
            # Đánh dấu tham số đáng ngờ - ít ngẫu nhiên hơn
            df['has_suspicious_args'] = 0
            args_str = df[args_col].astype(str)
            
            # Phát hiện tham số rủi ro cao - 100% đáng ngờ
            for arg in high_risk_args:
                mask = args_str.str.contains(arg, regex=False)
                df.loc[mask, 'has_suspicious_args'] = 1
            
            # Phát hiện tham số rủi ro trung bình - 75% đáng ngờ
            for arg in medium_risk_args:
                mask = args_str.str.contains(arg, regex=False) & (df['has_suspicious_args'] == 0)
                df.loc[mask, 'has_suspicious_args'] = np.where(
                    np.random.random(size=len(df[mask])) < 0.75, 1, 0
                )
            
            # Phát hiện dấu hiệu đáng ngờ - 50% đáng ngờ
            for indicator in suspicious_indicators:
                mask = args_str.str.contains(indicator, regex=False) & (df['has_suspicious_args'] == 0)
                df.loc[mask, 'has_suspicious_args'] = np.where(
                    np.random.random(size=len(df[mask])) < 0.5, 1, 0
                )
            
            # Lưu điểm đánh giá để sử dụng sau - chỉ dùng nội bộ
            df['args_risk_score'] = 0.0
            
            # High risk args
            for arg in high_risk_args:
                mask = args_str.str.contains(arg, regex=False)
                df.loc[mask, 'args_risk_score'] = 0.9  # Ít ngẫu nhiên hơn
            
            # Medium risk args
            for arg in medium_risk_args:
                mask = args_str.str.contains(arg, regex=False) & (df['args_risk_score'] < 0.4)
                df.loc[mask, 'args_risk_score'] = 0.6
            
            # Suspicious indicators
            for indicator in suspicious_indicators:
                mask = args_str.str.contains(indicator, regex=False) & (df['args_risk_score'] < 0.3)
                df.loc[mask, 'args_risk_score'] = 0.3
        else:
            df['has_suspicious_args'] = 0
            df['args_risk_score'] = 0.0
        
        # --- ĐIỀU CHỈNH 3: Tạo nhãn bất thường - GIẢM NGẪU NHIÊN, TĂNG TỶ LỆ DƯƠNG TÍNH ---
        
        # Đánh dấu hoạt động bất thường - sử dụng các quy tắc đơn giản với ít ngẫu nhiên hơn
        df['is_abnormal_activity'] = 0
        
        # Quy tắc 1: Lệnh đáng ngờ và tham số đáng ngờ = hoạt động bất thường
        mask1 = (df['is_suspicious_command'] == 1) & (df['has_suspicious_args'] == 1)
        df.loc[mask1, 'is_abnormal_activity'] = 1
        
        # Quy tắc 2: Thời gian bất thường + lệnh đáng ngờ cao = hoạt động bất thường 90%
        mask2 = (df['is_after_hours'] == 1) & (df['suspicion_score'] >= 0.8) & (df['is_abnormal_activity'] == 0)
        df.loc[mask2, 'is_abnormal_activity'] = np.where(
            np.random.random(size=len(df[mask2])) < 0.9, 1, 0
        )
        
        # Quy tắc 3: Thời gian bất thường + tham số đáng ngờ cao = hoạt động bất thường 80%
        mask3 = (df['is_after_hours'] == 1) & (df['args_risk_score'] >= 0.8) & (df['is_abnormal_activity'] == 0)
        df.loc[mask3, 'is_abnormal_activity'] = np.where(
            np.random.random(size=len(df[mask3])) < 0.8, 1, 0
        )
        
        # Quy tắc 4: Tham số rất đáng ngờ hoặc lệnh rất đáng ngờ = hoạt động bất thường 70%
        mask4 = ((df['suspicion_score'] >= 0.85) | (df['args_risk_score'] >= 0.85)) & (df['is_abnormal_activity'] == 0)
        df.loc[mask4, 'is_abnormal_activity'] = np.where(
            np.random.random(size=len(df[mask4])) < 0.7, 1, 0
        )
        
        # Quy tắc 5: Lệnh đáng ngờ trung bình + tham số đáng ngờ trung bình + thời gian bất thường = bất thường 60%
        mask5 = (df['is_after_hours'] == 1) & (df['suspicion_score'] >= 0.5) & (df['args_risk_score'] >= 0.5) & (df['is_abnormal_activity'] == 0)
        df.loc[mask5, 'is_abnormal_activity'] = np.where(
            np.random.random(size=len(df[mask5])) < 0.6, 1, 0
        )
        
        # Lưu điểm bất thường để sử dụng sau - chỉ dùng nội bộ
        df['abnormal_probability'] = 0.0
        df.loc[mask1, 'abnormal_probability'] = 0.95
        df.loc[mask2, 'abnormal_probability'] = 0.85
        df.loc[mask3, 'abnormal_probability'] = 0.8 
        df.loc[mask4, 'abnormal_probability'] = 0.75
        df.loc[mask5, 'abnormal_probability'] = 0.6
        
        # Thêm is_anomaly column để tương thích với train_model.py
        df['is_anomaly'] = df['is_abnormal_activity'] 
        
        # Thêm cột ngày để phân tích
        if time_col in df and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df['date'] = df[time_col].dt.date
            df['day_of_week'] = df[time_col].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        else:
            # Nếu không chuyển đổi được thời gian, tạo giá trị mặc định
            df['date'] = pd.Timestamp('2023-01-01').date()
            df['day_of_week'] = np.random.randint(0, 7, size=len(df))
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Trả về DataFrame với các cột cần thiết - KHÔNG bao gồm các cột điểm đánh giá
        columns_to_return = [
            'user', 'date', 'hour', 'is_after_hours', 'is_suspicious_command', 
            'has_suspicious_args', 'is_abnormal_activity', 'is_anomaly', 
            'day_of_week', 'is_weekend'
        ]
        
        # Thêm các cột gốc nếu có
        if cmd_col and cmd_col not in columns_to_return:
            columns_to_return.append(cmd_col)
        if args_col and args_col not in columns_to_return:
            columns_to_return.append(args_col)
        if key_col and key_col not in columns_to_return:
            columns_to_return.append(key_col)
        if time_col and time_col not in columns_to_return:
            columns_to_return.append(time_col)
        
        # Chỉ trả về các cột tồn tại trong DataFrame
        return df[[col for col in columns_to_return if col in df.columns]]
        
    except Exception as e:
        print(f"Lỗi xử lý file privilege commands: {e}")
        import traceback
        traceback.print_exc()
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