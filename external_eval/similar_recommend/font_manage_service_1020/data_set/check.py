import json
import os
def read_json_file(file_path):
    """
    函数作用：读取并解析一个JSON文件。
    输入参数：
        file_path (str): JSON文件的路径。
    输出参数：
        dict/list: 解析后的JSON数据。如果文件未找到或格式错误，则返回None并打印错误信息。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
    except json.JSONDecodeError:
        print("JSON格式错误")

data_all = read_json_file('files/data_all.json')
path_list = os.listdir('font_files_1')

for path in path_list:
    otf_id = int(path.split('@')[0])
    otf_name = path.split('@')[1]
    if int(data_all[otf_name]['font_id']) != otf_id:
        print(otf_id) 

