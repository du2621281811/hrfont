import os
import shutil
import time

def safe_dest_path(dest_dir, filename):
    """
    如果目标文件已存在，则添加 (_1), (_2), ... 后缀避免覆盖
    """
    base, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{base}(_{counter}){ext}"
        counter += 1
    return os.path.join(dest_dir, candidate)

def contains_forbidden_word(path):
    """
    判断路径（文件夹或文件名）是否包含禁止词
    """
    forbidden_keywords = ["竖排", "T-"]
    return any(keyword in path for keyword in forbidden_keywords)

def rename_and_move_otf_files(root_dir, target_dir, copy_instead=False, log_path=None):
    os.makedirs(target_dir, exist_ok=True)
    log_f = open(log_path, 'w', encoding='utf-8') if log_path else None

    for current_dir, subdirs, files in os.walk(root_dir):
        # 跳过含禁止词的文件夹
        if contains_forbidden_word(current_dir):
            print(f"🚫 跳过文件夹: {current_dir}")
            continue

        # 找出当前目录下所有 .otf 文件，排除含“竖排”或“T-”的文件
        otf_files = [
            f for f in files
            if f.lower().endswith('.otf') and not contains_forbidden_word(f)
        ]
        if not otf_files:
            continue

        # 取修改时间最新的文件
        full_paths = [os.path.join(current_dir, f) for f in otf_files]
        full_paths.sort(key=lambda p: os.path.getmtime(p))
        chosen = full_paths[-1]
        chosen_name = os.path.basename(chosen)

        # 当前和上一级文件夹名
        current_name = os.path.basename(current_dir)
        parent_name = os.path.basename(os.path.dirname(current_dir)) or os.path.basename(root_dir)

        # 新文件名格式：上一级@当前@原文件名
        new_name = f"{parent_name}@{current_name}@{chosen_name}"
        dest_path = safe_dest_path(target_dir, new_name)

        try:
            if copy_instead:
                shutil.copy2(chosen, dest_path)
                action = "复制"
            else:
                shutil.move(chosen, dest_path)
                action = "移动"

            msg = f"{action}: {chosen} -> {dest_path}"
            print(msg)
            if log_f:
                log_f.write(msg + "\n")

        except Exception as e:
            err = f"❌ 错误: 无法处理 {chosen} -> {dest_path} : {e}"
            print(err)
            if log_f:
                log_f.write(err + "\n")

    if log_f:
        log_f.close()

if __name__ == "__main__":
    # ====== 配置区 ======
    root_folder = "rest_file"     # 源文件夹路径
    save_folder = "font_files_new"   # 目标保存路径
    copy_instead_of_move = True             # True=复制, False=移动（默认移动）
    log_file = "rename_move_log.txt"         # 日志文件路径，可设为 None 不生成日志
    # ====================

    rename_and_move_otf_files(root_folder, save_folder, copy_instead_of_move, None)
