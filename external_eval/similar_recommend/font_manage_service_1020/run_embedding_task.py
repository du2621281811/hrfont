import torch
import sys
import json
import argparse
from tools import enVectorManage
from log_init import logger

def run_embedding_task():
    try:
        logger.info(f"开始响应")
        # 初始化模型
        en_vector_manage = enVectorManage()
        
        # 获取字体嵌入
        result_dict = en_vector_manage.get_embedding('FTP_files/target.TTF')
        # 如果推理成功，返回结果
        embedding = result_dict["embedding"]
        logger.info(f"成功响应，返回值为：{embedding}")
        return {"status": 0, "embedding": embedding}
    except Exception as e:
        return {"status": 1, "message": f"Error occurred: {str(e)}"}


if __name__ == "__main__":
    
    
    # 执行模型推理并返回结果
    result = run_embedding_task()
    
    # 打印结果为JSON格式
    print(json.dumps(result))
