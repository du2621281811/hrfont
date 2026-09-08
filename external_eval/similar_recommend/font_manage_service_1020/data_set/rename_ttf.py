import json
import pandas as pd
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
path_list = os.listdir('font_files_new')

df = pd.read_excel('files/配符号字体名称202510.xlsx')

font_name_list = df['中文名TTF']
df_path_list = df['是否兼备TTF']
data_all = read_json_file('files/data_all.json')
family_list = []
id_list = []
for name in font_name_list:
    if name in data_all:
        data_info = data_all[name]
        family_list.append(data_info['font_family'])
        id_list.append(data_info['font_id'])


for i in range(len(font_name_list)):
    real_path = df_path_list[i]
    new_path = str(id_list[i]) + '@' + font_name_list[i] + '@' + family_list[i] + '@' +real_path
    real_path = os.path.join('font_files_new',real_path)
    new_path = os.path.join('font_files_new',new_path)
    print(real_path,new_path)

    os.rename(real_path, new_path)
