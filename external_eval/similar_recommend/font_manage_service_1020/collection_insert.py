import requests
import json
import os
from fontTools.ttLib import TTFont
import unicodedata
import pandas as pd
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

def count_font_symbols(ttf_path):
    # 打开字体文件
    font = TTFont(ttf_path)
    
    # 获取所有 unicode codepoints
    codepoints = set()
    for table in font["cmap"].tables:
        codepoints.update(table.cmap.keys())

    # 汉字范围（CJK Unified Ideographs）
    # GB2312 的6763汉字主要在这个范围内
    def is_chinese(cp):
        return (
            0x4E00 <= cp <= 0x9FFF or  # 基本汉字
            0x3400 <= cp <= 0x4DBF or  # 扩展A
            0x20000 <= cp <= 0x2A6DF or  # 扩展B
            0x2A700 <= cp <= 0x2B73F or  # 扩展C
            0x2B740 <= cp <= 0x2B81F or  # 扩展D
            0x2B820 <= cp <= 0x2CEAF or  # 扩展E
            0xF900 <= cp <= 0xFAFF  # 兼容汉字
        )

    chinese_chars = {cp for cp in codepoints if is_chinese(cp)}
    symbol_chars = codepoints - chinese_chars


    # 返回结果
    return len(symbol_chars)


def test_en_insert():
    path = 'font_files_all'
    path_list = os.listdir(path)
    url = "http://127.0.0.1:11006/enInsert"
    #ttf_url = "font_files/216@方正隶变简体@FZLBJW.TTF"
    num_list = []
    for i in range(len(path_list)):
        ttf_url = os.path.join(path,path_list[i])
        font_info = ttf_url.split('/')[-1].split('@')
        count = count_font_symbols(ttf_url)
        num_list.append(count)
        
        if count<= 1600:
            GB = 'GB'
        else:
            GB = 'GBK'
        font_id = int(font_info[0])
        font_name = font_info[1]
        font_family = font_info[2]
        # 准备请求体，确保字段完整且类型正确
        payload = {
            "font_id": font_id,                  # int or convertible to int
            "font_name": font_name,       # str
            "font_family": font_family,           # str, 如果是非方正字体必须是 "None"
            "is_fz_font": 1,                 # int (0 或 1)
            "ttf_version": "Version 1.00",           # str
            "otf_version": "Version 1.00",           # str
            "token": '19d93215d30b7608a06634f62730313ac2e95282',
            "ttf_url": ttf_url,  # 可选，但至少要有 ttf_url 或 otf_url
            "otf_url": "",                    # 可留空
            "GB_GBK": GB
        }
        print(payload)
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        print("Status Code:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)
        

    


if __name__ == "__main__":
    test_en_insert()
