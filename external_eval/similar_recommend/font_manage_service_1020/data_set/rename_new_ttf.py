import os
import json
from pypinyin import lazy_pinyin, Style
import re
from fuzzywuzzy import process, fuzz  # 若使用 rapidfuzz，替换为 from rapidfuzz import process, fuzz
import shutil

def chinese_to_initials(name: str) -> str:
    """
    将中文名转化为拼音首字母
    :param name: 中文名，例如 "刘磊"
    :return: 拼音首字母，例如 "LL"
    """
    # lazy_pinyin 获取拼音，Style.FIRST_LETTER 获取首字母
    initials = lazy_pinyin(name, style=Style.FIRST_LETTER)
    # 拼接成字符串并转换为大写
    return ''.join(initials).upper()

def strip_extension(filename: str) -> str:
    """去掉最后的扩展名并返回大写结果"""
    return re.sub(r'\.[^.]+$', '', filename).upper()

def clean_choice_text(s: str) -> str:
    """
    去除特殊符号，只保留中文、英文和数字。
    例如 "方正字迹-兰梓星座体 简" -> "方正字迹兰梓星座体简"
    """
    return re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]', '', s)

def initials_from_text(s: str) -> str:
    """
    把清洗后的字符串每个字符转换为首字母（中文取拼音首字母，英文取字母并大写，数字保留）。
    最终结果为大写字符串，例如 "微软雅黑 Bold" -> "MYYHBOLD"（英文也被连接）
    """
    result = []
    for ch in s:
        # 中文字符
        if '\u4e00' <= ch <= '\u9fff':
            # pypinyin 返回小写首字母，这里转为大写
            letter = lazy_pinyin(ch, style=Style.FIRST_LETTER)[0].upper()
            result.append(letter)
        else:
            # 英文或数字：直接取大写（字母），数字保留
            if ch.isalpha():
                result.append(ch.upper())
            else:
                result.append(ch)  # 数字等直接加入
    return ''.join(result)

def match_by_initials(query_filename: str, choices: list, limit: int = 1, scorer=fuzz.ratio):
    """
    query_filename: 如 "FZZJLZXZTFH.otf"（不需要转拼音）
    choices: 原始字体名列表（会被清洗并生成首字母拼音）
    返回：按相似度排序的匹配列表 [(原始字体名, initials, score), ...]
    """
    query_key = strip_extension(query_filename)  # 去掉扩展名并大写

    # 生成 choices -> initials 的映射（保留原始名）
    mapping = {}
    for choice in choices:
        cleaned = clean_choice_text(choice)
        initials = initials_from_text(cleaned)
        mapping[choice] = initials

    # 先尝试精确或前缀匹配优先（如果完全一致或以 query 开头）
    for orig, initials in mapping.items():
        if initials == query_key:
            return [(orig, initials, 100)]
    # 如果没精确匹配，再做模糊匹配（取 top-N）
    initials_list = list(mapping.values())
    matches = process.extract(query_key, initials_list, limit=limit, scorer=scorer)

    # matches 是 [(matched_initials, score), ...]（fuzzywuzzy 的行为）
    # 反查得到原始名字（可能存在多个相同 initials，因此取第一个匹配的原始名）
    results = []
    for matched_initials, score in matches:
        # 找原始名（第一个匹配的）
        orig_names = [k for k, v in mapping.items() if v == matched_initials]
        orig = orig_names[0] if orig_names else None
        results.append((orig, matched_initials, score))

    return results


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

path_all = 'font_files_new'
path_list = os.listdir(path_all)
data_all = read_json_file('files/data_all.json')
keys = list(data_all.keys())
i=0
for name in path_list:
    otf_name = name.split('@')[-1].split('.')[0]
    otf_name = ''.join(re.findall(r'[A-Za-z]', otf_name))
    font_name = name.split('@')[1]
    clean_input = re.sub(r"[^\u4e00-\u9fff]", "", font_name)
    best_match = process.extractOne(clean_input, keys, scorer=fuzz.token_sort_ratio)
    if best_match[1]>=70:
        font_real_name = best_match[0]
        font_info = data_all[font_real_name]
        font_id = font_info['font_id']
        font_family = font_info['font_family']
        real_path = str(font_id) + '@' + font_real_name + '@' + font_family + '@' + name


        old_path = os.path.join(path_all,name)
        target_path = os.path.join('font_files_1',real_path)
        shutil.move(old_path, target_path)
        #print(name,best_match)

    elif best_match[1]<70 and best_match[1]>=0:
        font_pingyin = chinese_to_initials(best_match[0])
        pingyin_score = fuzz.ratio(otf_name,font_pingyin)
        if pingyin_score >= 60:
            font_real_name = best_match[0]
            font_info = data_all[font_real_name]
            font_id = font_info['font_id']
            font_family = font_info['font_family']
            real_path = str(font_id) + '@' + font_real_name + '@' + font_family + '@' + name


            old_path = os.path.join(path_all,name)
            target_path = os.path.join('font_files_1',real_path)
            shutil.move(old_path, target_path)

        else:
            print(name , best_match)
    