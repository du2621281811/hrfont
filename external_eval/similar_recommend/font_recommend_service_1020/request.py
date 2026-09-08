import requests
import json
from ftplib import FTP
# 启动ftp服务
ftp = FTP()
ftp.connect(host='172.19.52.231', port=21)
ftp.login(user='ftpuser', passwd='12345678')
ttf_path = 'font_files_new/20174@方正字迹-顾建平隶书 简@方正字迹-顾建平隶书@字迹@55-方正字迹-顾建平隶书简繁@FZZJGJPLSFH.otf'
ttf_url = 'target.TTF'
#上传文件到ftp服务
with open(ttf_path, 'rb') as f:
    ftp.storbinary('STOR '+ttf_url, f)
print('ftp success')
# 1. 定义服务器地址和端口
PORT = 11005
BASE_URL = f"http://172.19.52.231:{PORT}"

# 2. 准备要提交的JSON数据
# 这是一个Python字典，requests会自动将其序列化为JSON字符串
payload = {
    'token': '19d93215d30b7608a06634f62730313ac2e95282',
    'ttf_path': ttf_url,
    'font_code': 'GBK'
}

# 3. 拼接完整的URL并发送POST请求
# 我们以 /recommend/en_recommend/ 为例
endpoint = "/recommend/symbol_recommend/"
url = BASE_URL + endpoint
headers = {
            "Content-Type": "application/json"
        }
try:
    # 发起POST请求，并通过 json 参数传递数据
    # requests 会自动设置 Content-Type 为 application/json
    response = requests.post(url, headers=headers,data=json.dumps(payload))

    # 4. 处理响应
    if response.status_code == 200:
        print("请求成功！")
        print("响应内容 (JSON):", response.json())
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("响应内容:", response.text)

except requests.exceptions.RequestException as e:
    print(f"请求发生异常: {e}")
