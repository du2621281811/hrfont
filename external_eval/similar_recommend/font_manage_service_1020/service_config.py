import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--token', default=['19d93215d30b7608a06634f62730313ac2e95282'])

parser.add_argument('--port', default=11006, help='服务启动所用端口')

parser.add_argument('--milvus_hostname', default='172.19.52.32', help='Milvus部署的服务器地址')
parser.add_argument('--milvus_port', default=19530, help='Milvus部署的服务器端口号')
parser.add_argument('--collection_name', default='font_recommend', help='中文需要调用的collection')
parser.add_argument('--en_collection_name', default='en_similar_symbols', help='西文需要调用的collection')

parser.add_argument('--version_save_path', default='./version/version.json', help='存储版本号的json，任何增删改服务都更新一次版本号')
parser.add_argument('--en_version_save_path', default='./version/en_version.json', help='西文存储版本号的json，任何增删改服务都更新一次版本号')

parser.add_argument('--log_path', default='logs/service.log', help='日志输出位置')
parser.add_argument('--en_log_path', default='logs/en_service.log', help='西文日志输出位置')

parser.add_argument('--font_save_dir', default='/resources/font_recommend_manage_font_dir', help='下载字体保存目录')
parser.add_argument('--en_font_save_dir', default='/resources/en_font_recommend_manage_font_dir', help='下载字体保存目录')

parser.add_argument('--img_save_dir', default='./img_files', help='特征提取所用图像存储目录')
parser.add_argument('--en_img_save_dir', default='./en_img_files', help='西文字体特征提取所用图像存储目录')
parser.add_argument('--sort6763_path', default='./files/sort6763-永和惠风.txt', help='特征提取备选字符排序')

parser.add_argument('--use_string', default='永和惠风', help='用于提取特征的字符串')
parser.add_argument('--en_use_string', default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', help='西文用于提取特征的字符串')
parser.add_argument('--model_path', default='files/class5330_coatnet_weight/epoch_2_step_90000_checkpoints.pth',
                    help='特征提取模型权重路径')
parser.add_argument('--en_model_path', default='./files/en_class14603_coatnet_weight/epoch_49_step_20000_checkpoints.pth',
                    help='西文特征提取模型权重路径')

parser.add_argument('--ftp_host', default='172.19.45.60', help='ftp服务地址')
parser.add_argument('--ftp_port', default=21, help='ftp服务端口号')
parser.add_argument('--ftp_user', default='ftpuser', help='ftp服务用户名')
parser.add_argument('--ftp_password', default='12345678', help='ftp服务密码')

args = parser.parse_args()
