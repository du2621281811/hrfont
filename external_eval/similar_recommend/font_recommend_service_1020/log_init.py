from service_config import args
import logging

def init_logger(name='LOG', log_file=None, log_level=logging.INFO):
    """
    :param args: argparse配置文件
    """
    # 创建 logger
    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)  # 设置日志级别
    # 文件日志初始化
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    # 控制台日志初始化
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    # 定义 handler 的输出格式
    formatter = logging.Formatter('%(asctime)s, %(levelname)s -> %(message)s')  # %(name)s
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 给 logger 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = init_logger(name='LOG', log_file=args.log_path)
en_logger = init_logger(name='ENLOG', log_file=args.en_log_path)