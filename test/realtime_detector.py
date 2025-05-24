"""
realtime_detector.py - Ứng dụng phát hiện leo thang đặc quyền trong thời gian thực
"""

import os
import pandas as pd
import numpy as np
import joblib
import time
import subprocess
import logging
import argparse
from datetime import datetime
import re
import json
from flask import Flask, request, jsonify, render_template_string

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('privilege_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('privilege_detector')

# Tạo ứng dụng Flask
app = Flask(__name__)

class PrivilegeEscalationDetector:
    """Phát hiện leo thang đặc quyền trong thời gian thực"""
    
    def __init__(self, model_path, features_path, threshold=0.7):
        """
        Khởi tạo detector
        
        Parameters:
        -----------
        model_path : str
            Đường dẫn đến file mô hình
        features_path : str
            Đường dẫn đến file thông tin đặc trưng
        threshold : float, default=0.7
            Ngưỡng để phát hiện lệnh bất thường
        whitelist_path : str, optional
            Đường dẫn đến file danh sách lệnh an toàn (whitelist), mặc định là None
        """
        self.threshold = threshold
        self.history = []
        
        # Thêm whitelist các lệnh an toàn
        self.safe_commands = set([
            'ls', 'cd', 'echo', 'pwd', 'date', 'uname', 'clear',
            'whoami', 'hostname', 'df', 'du', 'cat /etc/hostname',
            'ps', 'uptime', 'free', 'grep', 'wc', 'sort', 'head', 'tail',
            'less', 'more', 'cat /proc/cpuinfo', 'cat /proc/meminfo',
            'ifconfig', 'ip a', 'netstat', 'ping', 'which', 'whereis',
            'history', 'exit', 'logout', 'man', 'help', 'info'
        ])
        
        # Tải danh sách lệnh an toàn từ file nếu có
        if whitelist_path:
            try:
                with open(whitelist_path, 'r') as f:
                    self.safe_commands.update(f.read().splitlines())
                logger.info(f"Đã tải danh sách lệnh an toàn từ {whitelist_path}")
            except Exception as e:
                logger.error(f"Lỗi khi tải danh sách lệnh an toàn: {e}")
        
        self.suspicious_commands = set([
            'sudo', 'su', 'pkexec', 'gksudo', 'kdesudo', 'doas',
            'sudoedit', 'sudo -s', 'sudo -i', 'sudo su', 'su -',
            'setuid', 'setgid', 'chown', 'chmod u+s', 'chmod g+s',
            'cat /etc/passwd', 'cat /etc/shadow', 'cat /etc/crontab', 'cat /etc/sudoers',
            'find / -perm -4000', 'find / -perm -2000', 'find / -perm -u=s',
            'python3 -c', 'python -c', 'perl -e', 'ruby -e', 'setcap'
        ])
        self.suspicious_args = set([
            '/etc/passwd', '/etc/shadow', '/etc/sudoers', '/etc/crontab',
            '/root/.ssh', '/etc/ssh', 'visudo', '0777', 'os.setuid',
            '4755', 'u+s', 'g+s', 'a+rwx', '777', 'bash -i', '-perm -4000',
            'os.system', '/bin/sh', '/bin/bash', '-u=s', '-perm -2000', 
            'import os', 'exec', 'eval'
        ])
        
        # Tải mô hình và thông tin đặc trưng
        try:
            self.model = joblib.load(model_path)
            self.features_info = joblib.load(features_path)
            logger.info(f"Đã tải mô hình từ {model_path}")
            
            # Lấy danh sách đặc trưng
            self.feature_names = self.features_info.get('feature_names', [])
            if not self.feature_names:
                logger.warning("Không tìm thấy danh sách đặc trưng trong file thông tin")
                
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            raise
    
    def is_suspicious_command(self, command):
        """Kiểm tra xem lệnh có đáng ngờ không"""
        command_parts = command.split()
        if not command_parts:
            return False, False
        
        base_cmd = command_parts[0]
        args = ' '.join(command_parts[1:])
        
        # Kiểm tra lệnh và tham số đáng ngờ
        is_cmd_suspicious = any(susp_cmd in base_cmd for susp_cmd in self.suspicious_commands)
        has_suspicious_args = any(susp_arg in args for susp_arg in self.suspicious_args)
        
        return is_cmd_suspicious, has_suspicious_args
    
    def process_command(self, command, user, timestamp=None):
        """
        Xử lý lệnh và tạo các đặc trưng
        
        Parameters:
        -----------
        command : str
            Lệnh cần phân tích
        user : str
            Người dùng thực hiện lệnh
        timestamp : datetime, optional
            Thời điểm thực hiện lệnh, mặc định là thời gian hiện tại
            
        Returns:
        --------
        dict
            Đặc trưng của lệnh để dự đoán
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Phân tích lệnh
        is_suspicious_command, has_suspicious_args = self.is_suspicious_command(command)
        
        # Tạo đặc trưng cơ bản
        features = {
            'user': user,
            'command': command.split()[0] if command else '',
            'args': ' '.join(command.split()[1:]) if command and len(command.split()) > 1 else '',
            'hour': timestamp.hour,
            'date': timestamp.strftime("%Y-%m-%d"),
            'day_of_week': timestamp.weekday(),
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_business_hours': 1 if 8 <= timestamp.hour < 18 and timestamp.weekday() < 5 else 0,
            'is_suspicious_command': 1 if is_suspicious_command else 0,
            'has_suspicious_args': 1 if has_suspicious_args else 0,
            'is_abnormal_logon': 0,  # Mặc định là 0, sẽ được cập nhật nếu cần
            'is_abnormal_activity': 0  # Mặc định là 0, sẽ được cập nhật nếu cần
        }
        
        # Bổ sung thông tin lịch sử 
        if self.history:
            # Tính thời gian kể từ lệnh cuối cùng
            last_cmd = self.history[-1]
            time_diff = (timestamp - last_cmd['timestamp']).total_seconds() / 60  # phút
            features['time_since_last_activity'] = time_diff
            
            # Đánh dấu bất thường về thời gian
            if time_diff > 0:
                features['is_time_anomaly'] = 1 if time_diff > 120 else 0  # > 2 giờ là bất thường
            
            # Đánh dấu burst activity
            features['is_in_burst'] = 1 if time_diff < 1 else 0  # < 1 phút là burst
            
            # Đếm số lệnh đáng ngờ trong 10 lệnh gần nhất
            recent_cmds = self.history[-10:] if len(self.history) >= 10 else self.history
            suspicious_count = sum(1 for cmd in recent_cmds if cmd.get('is_suspicious_command', 0) == 1)
            features['suspicious_command_count'] = suspicious_count
            
            # Tổng đánh dấu bất thường
            anomaly_signals = (features['is_suspicious_command'] + 
                              features['has_suspicious_args'] + 
                              features['is_time_anomaly'] + 
                              int(suspicious_count >= 3))  # >= 3 lệnh đáng ngờ gần đây
            features['total_anomaly_signals'] = anomaly_signals
            features['multiple_anomaly_signals'] = 1 if anomaly_signals >= 2 else 0
        
        # Lưu vào lịch sử
        self.history.append({
            'command': command,
            'user': user,
            'timestamp': timestamp,
            'is_suspicious_command': features['is_suspicious_command'],
            'has_suspicious_args': features['has_suspicious_args']
        })
        
        # Giới hạn lịch sử chỉ giữ 1000 lệnh gần nhất
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return features
    
    def prepare_features_for_model(self, features_dict):
        """
        Chuẩn bị đặc trưng cho mô hình
        
        Parameters:
        -----------
        features_dict : dict
            Đặc trưng đã thu thập
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame chứa đặc trưng đã chuẩn bị
        """
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame([features_dict])
        
        # Áp dụng one-hot encoding cho cột user
        if 'user' in df.columns:
            user_dummies = pd.get_dummies(df['user'], prefix='user')
            df = pd.concat([df, user_dummies], axis=1)
            df = df.drop('user', axis=1)
        
        # Đảm bảo có đủ các cột như mô hình yêu cầu
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        
        # Loại bỏ các cột thừa và sắp xếp theo thứ tự đúng
        if self.feature_names:
            df = df[self.feature_names]
        
        return df
    
    def predict(self, command, user=None, timestamp=None):
        """
        Dự đoán lệnh có phải là leo thang đặc quyền
            
        Parameters:
        -----------
        command : str
            Lệnh cần dự đoán
        user : str, optional
            Người dùng thực hiện lệnh
        timestamp : str, optional
            Thời điểm thực hiện lệnh
                
        Returns:
        --------
        dict
            Kết quả phân loại
        """
        try:
            if not command:
                return {
                    'is_anomaly': False,
                    'probability': 0.0,
                    'details': 'Lệnh trống'
                }
                
            if user is None:
                try:
                    user = os.getlogin()
                except:
                    user = "unknown"
            
            if timestamp is None:
                timestamp = datetime.now()
                
            # Kiểm tra nếu lệnh nằm trong danh sách an toàn (whitelist)
            command_clean = command.strip()
            for safe_cmd in self.safe_commands:
                if command_clean == safe_cmd or command_clean.startswith(safe_cmd + ' '):
                    # Chỉ đơn giản hoá các lệnh cơ bản như ls -la, grep, etc.
                    if not any(susp_cmd in command_clean for susp_cmd in self.suspicious_commands) and \
                       not any(susp_arg in command_clean for susp_arg in self.suspicious_args):
                        logger.debug(f"Lệnh an toàn: {command_clean}")
                        result = {
                            'command': command,
                            'is_anomaly': False, 
                            'probability': 0.1,
                            'confidence': 0.8,
                            'details': {
                                'user': user,
                                'is_suspicious_command': False,
                                'has_suspicious_args': False,
                                'note': 'Sản lọc bởi whitelist'
                            }
                        }
                        if user:
                            result['user'] = user
                        if timestamp:
                            result['timestamp'] = timestamp
                        return result
            
            # Kiểm tra trực tiếp trước dựa trên danh sách lệnh đáng ngờ
            cmd_parts = command.split()
            base_cmd = cmd_parts[0] if cmd_parts else ""
            args = ' '.join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
            
            is_suspicious_cmd, has_suspicious_args = self.is_suspicious_command(command)
            
            # Nếu không phải lệnh đáng ngờ và không có tham số đáng ngờ, có thể trả về an toàn ngay
            if not is_suspicious_cmd and not has_suspicious_args and base_cmd not in ['su', 'sudo', 'pkexec', 'doas']:
                # Kiểm tra thêm các tiêu chí an toàn khác
                is_likely_safe = True
                if any(danger_term in command_clean for danger_term in ['setuid', 'setgid', 'chmod u+s', '4755', 'os.system']):
                    is_likely_safe = False
                
                if is_likely_safe:
                    features = self.process_command(command, user, timestamp)
                    return {
                        'is_anomaly': False,
                        'probability': 0.1,
                        'confidence': 0.9,
                        'details': {
                            'command': command,
                            'user': user,
                            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp),
                            'is_suspicious_command': False,
                            'has_suspicious_args': False,
                            'note': 'Lệnh thông thường'
                        }
                    }
            
            # Xử lý lệnh và tạo đặc trưng
            features = self.process_command(command, user, timestamp)
            
            # Tính toán xác suất dựa trên các đặc trưng đơn giản nếu không có đủ đặc trưng cho mô hình
            try:
                # Chuẩn bị đặc trưng cho mô hình
                X = self.prepare_features_for_model(features)
                
                # Dự đoán bằng mô hình
                probabilities = float(self.model.predict_proba(X)[:, 1][0])
            except Exception as e:
                logger.warning(f"Lỗi khi dự đoán bằng mô hình: {e}. Sử dụng heuristic thay thế.")
                # Tính toán đơn giản nếu mô hình gặp lỗi
                total_signals = (
                    (1 if features['is_suspicious_command'] else 0) + 
                    (0.7 if features['has_suspicious_args'] else 0) + 
                    (0.3 if features.get('is_time_anomaly', 0) else 0) +
                    (0.5 if features.get('is_in_burst', 0) else 0)
                )
                
                # Chuẩn hóa về thang 0-1
                probabilities = min(1.0, total_signals / 3.0)
            
            # Kiểm tra các trường hợp đặc biệt để tăng hoặc giảm xác suất
            if 'find / -perm -4000' in command_clean or 'find / -perm -u=s' in command_clean:
                probabilities = max(probabilities, 0.8)  # Tăng xác suất
            
            if 'cat /etc/passwd' in command_clean or 'cat /etc/shadow' in command_clean:
                probabilities = max(probabilities, 0.75)  # Tăng xác suất
                
            if "python" in command_clean and "os.setuid" in command_clean:
                probabilities = max(probabilities, 0.9)  # Tăng xác suất rất cao
            
            # Điều chỉnh ngưỡng dựa trên các tiêu chí khác nhau
            adjusted_threshold = self.threshold
            if is_suspicious_cmd:
                # Giảm ngưỡng xuống nếu là lệnh đáng ngờ
                adjusted_threshold *= 0.9
            
            if not has_suspicious_args and not is_suspicious_cmd:
                # Tăng ngưỡng lên nếu không có dấu hiệu rõ ràng
                adjusted_threshold *= 1.2
            
            # Đảm bảo các lệnh sudo, su luôn có xác suất cao nhưng cần thêm kiểm tra
            if base_cmd in ['sudo', 'su', 'pkexec', 'doas'] and args:
                # Nếu là lệnh sudo cho các lệnh bình thường, giảm xác suất 
                if any(safe_arg in args for safe_arg in ['apt', 'apt-get', 'yum', 'dnf', 'pacman', 'systemctl']):
                    probabilities = min(probabilities, 0.6)  # Giảm xác suất cho các lệnh phổ biến
                else:
                    probabilities = max(probabilities, 0.65)  # Tăng xác suất cho các lệnh khác
            
            # Quyết định cuối cùng
            prediction = probabilities >= adjusted_threshold
            
            # Thông tin chi tiết
            details = {
                'command': command,
                'user': user,
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp),
                'is_suspicious_command': bool(features['is_suspicious_command']),
                'has_suspicious_args': bool(features['has_suspicious_args']),
                'anomaly_probability': float(probabilities),
                'threshold': float(adjusted_threshold),
                'anomaly_signals': int(features.get('total_anomaly_signals', 0))
            }
            
            return {
                'is_anomaly': bool(prediction),
                'probability': float(probabilities),
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {e}")
            return {
                'is_anomaly': False,
                'probability': 0.0,
                'error': str(e)
            }
    
    def monitor_command_history(self, interval=5, max_duration=None, csv_log_path=None):
        """
        Giám sát lịch sử lệnh trong terminal hoặc từ file CSV log đặc quyền
        
        Parameters:
        -----------
        interval : int, default=5
            Khoảng thời gian giữa các lần kiểm tra (giây)
        max_duration : int, optional
            Thời gian tối đa để giám sát (giây), mặc định là không giới hạn
        csv_log_path : str, optional
            Đường dẫn đến file CSV log lệnh đặc quyền (ưu tiên)
        """
        start_time = time.time()
        
        if csv_log_path:
            logger.info(f"Đang giám sát file CSV log: {csv_log_path}")
            # Đọc số dòng hiện có để bỏ qua lệnh cũ (bỏ qua dòng tiêu đề)
            try:
                with open(csv_log_path, 'r') as f:
                    lines = f.readlines()
                last_line = len(lines)
                logger.info(f"Đã ghi nhận {last_line-1} lệnh đặc quyền hiện có trong log. Sẽ chỉ giám sát các lệnh mới.")
            except Exception as e:
                logger.error(f"Lỗi khi đọc log CSV ban đầu: {e}")
                last_line = 1
            logger.info(f"Bắt đầu giám sát log lệnh đặc quyền (Ctrl+C để dừng)...")
            try:
                while True:
                    if max_duration and (time.time() - start_time) > max_duration:
                        logger.info(f"Đã đạt thời gian tối đa ({max_duration}s). Dừng giám sát.")
                        break
                    try:
                        with open(csv_log_path, 'r') as f:
                            lines = f.readlines()
                        current_line = len(lines)
                        if current_line > last_line:
                            new_records = lines[last_line:]
                            logger.info(f"Phát hiện {len(new_records)} lệnh mới trong log đặc quyền")
                            import csv
                            from io import StringIO
                            for row in csv.reader(new_records):
                                if len(row) < 4:
                                    continue
                                timestamp, user, command, args = row[0], row[1], row[2], row[3]
                                # Đảm bảo không bị lặp command nếu args đã chứa command
                                args_clean = args.strip().replace('"', '')
                                if args_clean.startswith(command):
                                    cmd_full = args_clean
                                else:
                                    cmd_full = f"{command} {args_clean}".strip()
                                result = self.predict(cmd_full, user=user)
                                if result['is_anomaly']:
                                    logger.warning(f"CẢNH BÁO: {timestamp} - {user} - Phát hiện LEO THANG ĐẶC QUYỀN: {cmd_full}")
                                    logger.warning(f"Độ tin cậy: {result['probability']:.2f}")
                                    logger.warning(f"Chi tiết: {result['details']}")
                                    print(f"\033[91m[!] CẢNH BÁO: {timestamp} - {user} - Lệnh '{cmd_full}' có thể là leo thang đặc quyền (độ tin cậy: {result['probability']:.2f})\033[0m")
                                else:
                                    logger.info(f"{timestamp} - {user} - Lệnh bình thường: {cmd_full}")
                            last_line = current_line
                    except Exception as e:
                        logger.error(f"Lỗi khi đọc log CSV: {e}")
                    time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Đã dừng giám sát log đặc quyền theo yêu cầu người dùng.")
        else:
            # Mặc định: giám sát ~/.bash_history như trước
            history_file = os.path.expanduser('~/.bash_history')
            try:
                with open(history_file, 'r') as f:
                    initial_history_lines = f.readlines()
                last_history_size = len(initial_history_lines)
                logger.info(f"Đã ghi nhận {last_history_size} lệnh hiện có trong lịch sử. Sẽ chỉ giám sát các lệnh mới.")
            except Exception as e:
                logger.error(f"Lỗi khi đọc lịch sử ban đầu: {e}")
                last_history_size = 0
            logger.info(f"Bắt đầu giám sát lịch sử lệnh. Nhấn Ctrl+C để dừng.")
            try:
                while True:
                    if max_duration and (time.time() - start_time) > max_duration:
                        logger.info(f"Đã đạt thời gian tối đa ({max_duration}s). Dừng giám sát.")
                        break
                    try:
                        with open(history_file, 'r') as f:
                            history_lines = f.readlines()
                        current_size = len(history_lines)
                        if current_size > last_history_size:
                            new_commands = history_lines[last_history_size:]
                            logger.info(f"Phát hiện {len(new_commands)} lệnh mới trong lịch sử")
                            for cmd in new_commands:
                                cmd = cmd.strip()
                                if cmd:
                                    result = self.predict(cmd)
                                    if result['is_anomaly']:
                                        logger.warning(f"CẢNH BÁO: Phát hiện LEO THANG ĐẶC QUYỀN: {cmd}")
                                        logger.warning(f"Độ tin cậy: {result['probability']:.2f}")
                                        logger.warning(f"Chi tiết: {result['details']}")
                                        print(f"\033[91m[!] CẢNH BÁO: Lệnh '{cmd}' có thể là leo thang đặc quyền (độ tin cậy: {result['probability']:.2f})\033[0m")
                                    else:
                                        logger.info(f"Lệnh bình thường: {cmd}")
                            last_history_size = current_size
                    except Exception as e:
                        logger.error(f"Lỗi khi đọc lịch sử bash: {e}")
                    time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Đã dừng giám sát theo yêu cầu người dùng.")
    


# API endpoints
@app.route('/')
def index():
    """Trang chủ giám sát"""
    # Tạo template HTML đơn giản
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hệ thống phát hiện leo thang đặc quyền</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .container { max-width: 800px; margin: 0 auto; }
            .alert { padding: 15px; margin-bottom: 20px; border: 1px solid transparent; border-radius: 4px; }
            .alert-danger { background-color: #f2dede; border-color: #ebccd1; color: #a94442; }
            .alert-success { background-color: #dff0d8; border-color: #d6e9c6; color: #3c763d; }
            .form-group { margin-bottom: 15px; }
            input, button { padding: 8px 12px; }
            button { background-color: #337ab7; color: white; border: none; cursor: pointer; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .high { color: red; font-weight: bold; }
            .medium { color: orange; }
            .low { color: green; }
        </style>
        <script>
            // Auto-refresh every 5 seconds
            setInterval(function() {
                fetch('/api/alerts')
                    .then(response => response.json())
                    .then(data => {
                        const alertsTable = document.getElementById('alerts-tbody');
                        alertsTable.innerHTML = '';
                        
                        data.alerts.forEach(alert => {
                            const row = document.createElement('tr');
                            
                            const timeCell = document.createElement('td');
                            timeCell.textContent = alert.timestamp;
                            row.appendChild(timeCell);
                            
                            const commandCell = document.createElement('td');
                            commandCell.textContent = alert.command;
                            row.appendChild(commandCell);
                            
                            const userCell = document.createElement('td');
                            userCell.textContent = alert.user;
                            row.appendChild(userCell);
                            
                            const probabilityCell = document.createElement('td');
                            const prob = parseFloat(alert.probability);
                            probabilityCell.textContent = prob.toFixed(2);
                            if (prob >= 0.7) {
                                probabilityCell.className = 'high';
                            } else if (prob >= 0.4) {
                                probabilityCell.className = 'medium';
                            } else {
                                probabilityCell.className = 'low';
                            }
                            row.appendChild(probabilityCell);
                            
                            const detailsCell = document.createElement('td');
                            detailsCell.textContent = alert.details;
                            row.appendChild(detailsCell);
                            
                            alertsTable.appendChild(row);
                        });
                    });
            }, 5000);
            
            function analyzeCommand() {
                const command = document.getElementById('command-input').value;
                
                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ command: command }),
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    if (data.is_anomaly) {
                        resultDiv.innerHTML = `
                            <div class="alert alert-danger">
                                <strong>CẢNH BÁO!</strong> Lệnh này có thể là leo thang đặc quyền.
                                <p>Độ tin cậy: ${(data.probability * 100).toFixed(2)}%</p>
                                <p>Chi tiết: ${JSON.stringify(data.details)}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="alert alert-success">
                                <strong>AN TOÀN</strong> Lệnh này có vẻ bình thường.
                                <p>Độ tin cậy: ${(100 - data.probability * 100).toFixed(2)}%</p>
                            </div>
                        `;
                    }
                });
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Hệ thống Phát hiện Leo thang Đặc quyền</h1>
            
            <div class="form-group">
                <h2>Phân tích lệnh</h2>
                <input type="text" id="command-input" placeholder="Nhập lệnh cần phân tích" style="width: 70%;">
                <button onclick="analyzeCommand()">Phân tích</button>
            </div>
            
            <div id="result"></div>
            
            <h2>Các cảnh báo gần đây</h2>
            <table>
                <thead>
                    <tr>
                        <th>Thời gian</th>
                        <th>Lệnh</th>
                        <th>Người dùng</th>
                        <th>Độ tin cậy</th>
                        <th>Chi tiết</th>
                    </tr>
                </thead>
                <tbody id="alerts-tbody">
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

# Biến toàn cục để lưu các cảnh báo
alerts = []

@app.route('/analyze', methods=['POST'])
def analyze_command():
    """Phân tích lệnh đầu vào"""
    global alerts
    try:
        data = request.get_json(force=True)
        command = data.get('command', '')
        
        if not command:
            return jsonify({'error': 'Không có lệnh để phân tích'})
        
        # Log lệnh đang phân tích
        logger.info(f"Đang phân tích lệnh: {command}")
        
        # Dự đoán
        result = detector.predict(command)
        
        # Log kết quả
        if result['is_anomaly']:
            logger.warning(f"Lệnh '{command}' được phát hiện là leo thang đặc quyền (độ tin cậy: {result['probability']:.2f})")
        else:
            logger.info(f"Lệnh '{command}' được phân loại là bình thường (độ tin cậy: {1-result['probability']:.2f})")
        
        # Lưu vào danh sách cảnh báo nếu là bất thường
        if result['is_anomaly']:
            alerts.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'command': command,
                'user': result['details']['user'],
                'probability': result['probability'],
                'details': f"Suspicious command: {result['details']['is_suspicious_command']}, Suspicious args: {result['details']['has_suspicious_args']}"
            })
            
            # Giới hạn số lượng cảnh báo lưu trữ
            if len(alerts) > 100:
                alerts = alerts[-100:]
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Lỗi khi phân tích lệnh: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Lấy danh sách cảnh báo"""
    return jsonify({'alerts': alerts})

def main():
    """Hàm chính để khởi chạy ứng dụng"""
    parser = argparse.ArgumentParser(description='Phát hiện leo thang đặc quyền trong thời gian thực')
    
    parser.add_argument('--model', '-m', 
                       default='/home/joe/python_Proj/test/privilege_detection_model_latest.pkl',
                       help='Đường dẫn đến file mô hình')
    
    
    parser.add_argument('--features', '-f',
                       default='/home/joe/python_Proj/test/model_features_latest.pkl',
                       help='Đường dẫn đến file thông tin đặc trưng')
    
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Ngưỡng để phát hiện lệnh bất thường (0.0-1.0)')
    
    parser.add_argument('--mode', choices=['api', 'monitor'], default='api',
                       help='Chế độ hoạt động: api (web server), monitor (giám sát lịch sử)')
    
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Khoảng thời gian giữa các lần kiểm tra (giây)')
    
    parser.add_argument('--csv-log-path', type=str, default=None,
                       help='Đường dẫn tới file CSV log lệnh đặc quyền (ưu tiên cho chế độ monitor)')
    
    parser.add_argument('--port', '-p', type=int, default=5000,
                       help='Cổng cho web server')
    
    parser.add_argument('--debug', action='store_true',
                      help='Bật chế độ debug')
    
    args = parser.parse_args()
    
    # Khởi tạo detector
    global detector
    detector = PrivilegeEscalationDetector(args.model, args.features, args.threshold)
    
    # Chạy theo chế độ được chọn
    if args.mode == 'api':
        logger.info(f"Khởi động web server tại port {args.port}")
        app.run(debug=True, port=args.port)
    elif args.mode == 'monitor':
        detector.monitor_command_history(interval=args.interval, csv_log_path=args.csv_log_path)

    else:
        logger.error(f"Chế độ không hợp lệ: {args.mode}")

if __name__ == "__main__":
    main()
