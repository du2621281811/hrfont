import time
from pymilvus import connections
from pymilvus import Collection
from typing import List, Optional

from log_init import logger, en_logger
from service_config import args


class VectorSearch:
    def __init__(self):
        self.connect_alias = 'recommend'
        self.metric_type = 'COSINE'
        connections.connect(
            alias=self.connect_alias,
            host=args.milvus_hostname,
            port='19530'
        )
        logger.info(f"init INFO: Milvus connect success!")
        self.collection_name = args.collection_name
        self.collection = Collection(name=self.collection_name, using=self.connect_alias)
        self.collection.load()
        logger.info(f"init INFO: Collection load success: {self.collection_name}")

    def search(self, font_id, threshold, max_output,
               partition_names: Optional[List[str]] = None):
        """
        检索与推荐流程。
        方正字体有家族/系列字信息，所以过滤同系列字体，推荐结果中也保证每个系列字体仅出现一次。
        :param font_id: 待推荐的字体id
        :param threshold: 相似度过滤阈值
        :param max_output: 输出数量限制
        :param partition_names: 需要搜索的分区，list[string]格式，如果为None则搜索所有分区，
                                目前分为fz_font区和other_font区分别存储方正字体和第三方字体
        :return:
        """
        start = time.time()
        # 检查id是否存在
        query_result = self.collection.query(
            expr=f"font_id == {font_id}",
            offset=0,
            limit=1,
            output_fields=['embedding', 'font_family']
        )
        # 无此字体，返回错误信息
        if len(query_result) == 0:
            result_dict = {"status": 2, "data": [],
                           "message": f"Input font_id dose not exist: {font_id}"}
            return result_dict
        # 待推荐字体信息
        query_embedding = [query_result[0]['embedding']]  # 套一层list
        query_font_family = query_result[0]['font_family']
        # 过滤自己
        expr = f"font_id != {font_id}"
        # 检索参数
        search_params = {
            "metric_type": self.metric_type,
            "offset": 0,  # offset是跳过最相似的n个结果, limit是输出几个相似结果
            "ignore_growing": False,
            "params": {
                "radius": threshold,
                "range_filter": 1  # 检索结果的分数区间为threshold~1.0
            }
        }
        # 开始检索
        search_results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=16384,
            expr=expr,  # 过滤信息
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['font_id', 'font_family'],
            partition_names=partition_names,
            consistency_level="Bounded"
        )
        # 推荐策略
        data = {'result_font_ids': [], 'result_scores': []}
        exist_font_family = [query_font_family]  # 过滤同家族/系列字
        if max_output == -1:  # 传入-1,则不限制输出数量
            max_output = len(search_results[0].ids)
        for i in range(len(search_results[0].ids)):
            font_id = search_results[0][i].id
            score = search_results[0][i].distance
            font_family = search_results[0][i].entity.get('font_family')
            if font_family == 'None':  # 非方正字体，不做系列字过滤，直接加入推荐结果
                data['result_font_ids'].append(font_id)
                data['result_scores'].append(score)
            elif font_family not in exist_font_family:  # 方正字体，且为未出现的系列，加入推荐结果
                data['result_font_ids'].append(font_id)
                data['result_scores'].append(score)
                exist_font_family.append(font_family)
            else:  # 已经存在同系列，过滤掉
                continue
            if len(data['result_font_ids']) >= max_output:  # 数量足够，停止挑选
                break
        logger.info(f"recommend INFO, recommend time used: {round(time.time() - start, 5)}s")
        result_dict = {"status": 0, "data": [data],
                       "message": "Success!"}

        return result_dict

    def search_embedding(self, embedding, partition_names: Optional[List[str]] = None):
        search_params = {
            "metric_type": self.metric_type,
            "offset": 0,
            "ignore_growing": False,
            "params": {"radius": 0, "range_filter": 1}
        }
        search_results = self.collection.search(
            data=embedding,
            anns_field="embedding",
            param=search_params,
            limit=3,
            expr='',
            output_fields=['font_id'],
            partition_names=partition_names,
            consistency_level="Bounded"
        )[0]
        data = {"score": [], "font_id": []}
        for result in search_results:
            data['score'].append(result.score)
            data['font_id'].append(result.id)
        return {"status": 0, "data": data, "message": "Success!"}


class enVectorSearch:
    def __init__(self):
        self.connect_alias = 'en_recommend'
        self.metric_type = 'COSINE'
        connections.connect(
            alias=self.connect_alias,
            host=args.milvus_hostname,
            port='19530'
        )
        en_logger.info(f"init INFO: Milvus connect success!")
        self.en_collection_name = args.en_collection_name
        self.en_collection = Collection(name=self.en_collection_name, using=self.connect_alias)
        self.en_collection.load()
        en_logger.info(f"init INFO: Collection load success: {self.en_collection_name}")
    '''
    def search(self, font_id, threshold, max_output,
               partition_names: Optional[List[str]] = None):
        """
        检索与推荐流程。
        :param font_id: 待推荐的字体id
        :param threshold: 相似度过滤阈值
        :param max_output: 输出数量限制
        :param partition_names: 需要搜索的分区，list[string]格式，如果为None则搜索所有分区，
                                目前西文字体无厂商分区信息，该字端暂留
        :return:
        """
        start = time.time()
        # 检查id是否存在
        query_result = self.en_collection.query(
            expr=f"font_id == {font_id}",
            offset=0,
            limit=1,
            output_fields=['embedding']
        )
        # 无此字体，返回错误信息
        if len(query_result) == 0:
            result_dict = {"status": 2, "data": [],
                           "message": f"Input font_id dose not exist: {font_id}"}
            return result_dict
        # 待推荐字体信息
        query_embedding = [query_result[0]['embedding']]  # 套一层list
        # 过滤自己
        expr = f"font_id != {font_id}"
        # 检索参数
        search_params = {
            "metric_type": self.metric_type,
            "offset": 0,  # offset是跳过最相似的n个结果, limit是输出几个相似结果
            "ignore_growing": False,
            "params": {
                "radius": threshold,
                "range_filter": 1  # 检索结果的分数区间为threshold~1.0
            }
        }
        # 开始检索
        search_results = self.en_collection.search(
            data=query_embedding,
            anns_field="embedding",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=16384,
            expr=expr,  # 过滤信息
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['font_id'],
            partition_names=partition_names,
            consistency_level="Bounded"
        )
        # 推荐策略
        data = {'result_font_ids': [], 'result_scores': []}
        if max_output == -1:  # 传入-1,则不限制输出数量
            max_output = len(search_results[0].ids)
        for i in range(len(search_results[0].ids)):
            font_id = search_results[0][i].id
            score = search_results[0][i].distance
            data['result_font_ids'].append(font_id)
            data['result_scores'].append(score)
            if len(data['result_font_ids']) >= max_output:  # 数量足够，停止挑选
                break
        en_logger.info(f"recommend INFO, recommend time used: {round(time.time() - start, 5)}s")
        result_dict = {"status": 0, "data": [data],
                       "message": "Success!"}

        return result_dict
    '''
    def search(self, embedding, font_code = 'all', partition_names: Optional[List[str]] = None):

        """
        检索与推荐流程。
        :param font_id: 待推荐的字体id
        :param threshold: 相似度过滤阈值
        :param max_output: 输出数量限制
        :param partition_names: 需要搜索的分区，list[string]格式，如果为None则搜索所有分区，
                                目前西文字体无厂商分区信息，该字端暂留
        :return:
        """
        start = time.time()
        # 待推荐字体信息
        query_embedding = embedding
        # 检索参数
        search_params = {
            "metric_type": self.metric_type,
            "offset": 0,  # offset是跳过最相似的n个结果, limit是输出几个相似结果
            "ignore_growing": False,
            "params": {
                "radius": 0,
                "range_filter": 1  # 检索结果的分数区间为threshold~1.0
            }
        }
        if font_code != 'all':
            expr = f"GB_GBK == '{font_code}'"
        else:
            expr = ''
        logger.info(f"检索条件：{expr}")
        # 开始检索
        search_results = self.en_collection.search(
            data=query_embedding,
            anns_field="embedding",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=3,
            expr= expr,  # 过滤信息
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['font_id'],
            partition_names=partition_names,
            consistency_level="Bounded"
        )[0]
        # 推荐策略
        en_logger.info(f"search_result:{search_results}")
        data = {"score":[],"font_id":[]}
        for i in range(len(search_results)):
            data['score'].append(search_results[i].score)
            data['font_id'].append(search_results[i].font_id)
        
        en_logger.info(f"recommend INFO, recommend time used: {round(time.time() - start, 5)}s")
        result_dict = {"status": 0, "data": data,
                       "message": "Success!"}

        return result_dict


if __name__ == "__main__":
    pass
