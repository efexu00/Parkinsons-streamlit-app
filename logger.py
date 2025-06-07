from datetime import datetime

def log_upload(user, filename, model_used, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("upload_logs.txt", "a") as f:
        f.write(f"{timestamp}, {user}, {filename}, {model_used}, {prediction}\n")

