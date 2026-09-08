import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--token', default=['19d93215d30b7608a06634f62730313ac2e95282'])

parser.add_argument('--port', default=11005, help='服务启动所用端口')

parser.add_argument('--default_threshold', default=0.5, help='默认输出阈值，传入参数不指定则取该值')
parser.add_argument('--default_max_output', default=5, help='默认最大输出数量，传入参数不指定则取该值')

parser.add_argument('--milvus_hostname', default='172.19.52.32', help='Milvus部署的服务器地址')
parser.add_argument('--collection_name', default='font_recommend', help='中文需要调用的collection')
parser.add_argument('--en_collection_name', default='en_similar_symbols', help='西文需要调用的collection')

parser.add_argument('--log_path', default='logs/service.log', help='日志输出位置')
parser.add_argument('--en_log_path', default='logs/en_service.log', help='西文日志输出位置')

parser.add_argument('--host', default='172.19.45.60', help='当前服务器网址')

args = parser.parse_args()
