from pymilvus import MilvusClient, DataType
from log_init import logger
from service_config import args
import json
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)

def create(drop):
    '''
        新建Milvus集合并增加分区
    '''
    connections.connect(alias='default', host=args.milvus_hostname, port=args.milvus_port)
        
    # 检查集合是否存在，若存在则删除
    if utility.has_collection(args.en_collection_name):
        if drop:
            utility.drop_collection(args.en_collection_name)
            print(f"{args.en_collection_name} 已存在，当前被删除")
        else:
            print(f"{args.en_collection_name} 已存在，跳过创建")
            return

    # 创建集合字段
    fields = [
        FieldSchema(name="font_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="font_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="font_family", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="ttf_version", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="ttf_url", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="otf_version", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="otf_url", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="GB_GBK", dtype=DataType.VARCHAR, max_length=2000),
    ]

    schema = CollectionSchema(fields, description='Font recommend Collection')
    collection = Collection(name=args.en_collection_name, schema=schema)
    
    # 创建分区
    partitions = ['fz_font', 'other_font']
    for part in partitions:
        if not collection.has_partition(part):
            collection.create_partition(part)
            print(f"分区 {part} 已创建")

    # 为所有的向量字段创建索引
    index_params = {
        'index_type': 'FLAT',
        #'params': {'nlist':128},
        'metric_type': 'COSINE'
    }
    vector_fields = ['embedding']

    for field in vector_fields:
        collection.create_index(field_name=field, index_params=index_params)
        print(f"向量字段 {field} 索引已创建")

    collection.load()
    print("集合加载完成")


if __name__ == "__main__":
    create(drop=True)
