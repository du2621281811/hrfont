import paramiko
import os
from fuzzywuzzy import process, fuzz

hostname = "172.19.45.60"
port = 65534
username = "liulei"
password = "Founder@liulei"  # 如果用私钥可改为 pkey=private_key

def read_txt_to_list(file_path):
    """
    读取文本文件并将每一行存入列表
    
    Args:
        file_path (str): 文本文件路径
        
    Returns:
        list: 包含文件每一行内容的列表
    """
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []

font_name = read_txt_to_list('files/font_name.txt')
ttf_list = read_txt_to_list('files/fontslist_new_24688.txt')

# 建立连接
transport = paramiko.Transport((hostname, port))
transport.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)
dic_all = {}
for n in ttf_list:
    ttf_info = n.split(',,,')
    ttf_id = ttf_info[0]
    ttf_font_name = ttf_info[1]
    ttf_path = ttf_info[2]
    dic_all[ttf_font_name] = {}
    dic_all[ttf_font_name]['ttf_id'] = ttf_id
    dic_all[ttf_font_name]['ttf_path'] = ttf_path
i = 1
   
for name in font_name:
    ttf_name_list = list(dic_all.keys())
    if name in ttf_name_list:
        ttf_key = name
    else:
        matches = process.extract(name, ttf_name_list, scorer=fuzz.token_sort_ratio, limit=1)
        ttf_key = matches[0][0]
    ttf_font_name = ttf_key
    ttf_id = dic_all[ttf_key]['ttf_id']
    ttf_path = dic_all[ttf_key]['ttf_path']
    remote_path = "/home/data_nas_1/fonts/TFontPack/Allfont"  
    local_path = "font_files"
    save_name = str(ttf_id) + '@' +str(ttf_font_name) + '@' + str(ttf_path)
    remote_path = os.path.join(remote_path,ttf_path)
    local_path = os.path.join(local_path,save_name)
    sftp.get(remote_path, local_path)
    print(i)
    i+=1

# 关闭链接
sftp.close()
transport.close()
