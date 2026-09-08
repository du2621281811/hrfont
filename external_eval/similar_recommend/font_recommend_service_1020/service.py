# -*- coding: utf-8 -*-
import json
from tornado import web
from tornado import ioloop
from tools import VectorSearch, enVectorSearch
import traceback
import numpy as np
from log_init import logger, en_logger
from service_config import args
import requests

en_vector_search = enVectorSearch()
zh_vector_search = None


def check_input(request_body):
    """
    检查推荐服务的输入参数
    :param request_body:
    :return:
    """
    check_status = False
    input_argument = None
    # 输入转json格式
    try:
        input_argument = json.loads(request_body)
    except:
        result_dict = {"status": 1, "data": [],
                       "message": 'Input argument can not covert to json format!'}
        return check_status, input_argument, result_dict
    # 检查token
    if input_argument.get('token') is None:
        result_dict = {"status": 1, "data": [],
                       "message": "Except input argument: token"}
        return check_status, input_argument, result_dict
    token = input_argument.get('token')
    # token判定
    if token not in args.token:
        result_dict = {"status": 1, "data": [],
                       "message": f"Token not match: {token}"}
        return check_status, input_argument, result_dict

    check_status = True
    result_dict = None
    return check_status, input_argument, result_dict


class FzHandler(web.RequestHandler):
        
    def post(self):
        self.vector_search = VectorSearch()
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数
        if input_argument.get('font_id') is None:
            result_dict = {"status": 1, "data": [],
                           "message": "Except input argument: font_id"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        font_id = input_argument.get('font_id')
        threshold = input_argument.get('threshold', args.default_threshold)
        max_output = input_argument.get('max_output', args.default_max_output)
        # 检查输入参数格式
        try:
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            threshold = float(threshold)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' can not convert to float: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if threshold > 1 or threshold < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need in the range of 0 to 1, "
                                      f"but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            max_output = int(max_output)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'max_output' can not convert to int: {max_output}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if max_output < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need >= 0, but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 参数特殊处理
        if threshold == 0:  # 为0则全部输出
            threshold = -1
        if threshold == 1:  # 最大值为1会减去一个较小的值，保证区间。
            threshold -= 1e-4
        max_output = min(10000, max_output)  # 限制最大输出10000
        # 输入检查通过，进入推荐流程
        try:
            result_dict = self.vector_search.search(font_id, threshold=threshold, max_output=max_output, partition_names=['fz_font'])
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"recommend ERROR: {traceback.format_exc()}")
            result_dict = {"status": 2,
                           "message": "Recommend error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class AllHandler(web.RequestHandler):
        
    def post(self):
        self.vector_search = VectorSearch()
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数
        if input_argument.get('font_id') is None:
            result_dict = {"status": 1, "data": [],
                           "message": "Except input argument: font_id"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        font_id = input_argument.get('font_id')
        threshold = input_argument.get('threshold', args.default_threshold)
        max_output = input_argument.get('max_output', args.default_max_output)
        # 检查输入参数格式
        try:
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            threshold = float(threshold)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' can not convert to float: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if threshold > 1 or threshold < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need in the range of 0 to 1, "
                                      f"but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            max_output = int(max_output)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'max_output' can not convert to int: {max_output}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if max_output < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need >= 0, but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 参数特殊处理
        if threshold == 0:  # 为0则全部输出
            threshold = -1
        if threshold == 1:  # 最大值为1会减去一个较小的值，保证区间。
            threshold -= 1e-4
        max_output = min(10000, max_output)  # 限制最大输出10000
        # 输入检查通过，进入推荐流程
        try:
            result_dict = self.vector_search.search(font_id, threshold=threshold, max_output=max_output, partition_names=[])
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"recommend ERROR: {traceback.format_exc()}")
            result_dict = {"status": 2, "data": [],
                           "message": "Recommend service error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class enHandler(web.RequestHandler):
        
    def post(self):
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数
        if input_argument.get('font_id') is None:
            result_dict = {"status": 1, "data": [],
                           "message": "Except input argument: font_id"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if input_argument.get('only_recommend_fz') is None:
            result_dict = {"status": 1, "data": [],
                           "message": "Except input argument: only_recommend_fz"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        font_id = input_argument.get('font_id')
        only_recommend_fz = input_argument.get('only_recommend_fz')
        threshold = input_argument.get('threshold', args.default_threshold)
        max_output = input_argument.get('max_output', args.default_max_output)
        # 检查输入参数格式
        try:
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            only_recommend_fz = int(only_recommend_fz)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'only_recommend_fz' can not convert to int: {only_recommend_fz}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            threshold = float(threshold)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' can not convert to float: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if threshold > 1 or threshold < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need in the range of 0 to 1, "
                                      f"but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            max_output = int(max_output)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'max_output' can not convert to int: {max_output}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if max_output < 0:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'threshold' need >= 0, but received: {threshold}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 参数特殊处理
        if threshold == 0:  # 为0则全部输出
            threshold = -1
        if threshold == 1:  # 最大值为1会减去一个较小的值，保证区间。
            threshold -= 1e-4
        max_output = min(10000, max_output)  # 限制最大输出10000
        # 输入检查通过，进入推荐流程
        try:
            if only_recommend_fz:
                result_dict = en_vector_search.search(font_id, threshold=threshold, max_output=max_output,
                                                      partition_names=['fz_font'])
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
            else:
                result_dict = en_vector_search.search(font_id, threshold=threshold, max_output=max_output,
                                                      partition_names=[])
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
        except:
            logger.error(f"recommend ERROR: {traceback.format_exc()}")
            result_dict = {"status": 2,
                           "message": "Recommend error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)

class symbolHandler(web.RequestHandler):
    def post(self):
        global zh_vector_search
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        input_dic = json.loads(self.request.body)
        token = input_dic.get('token')
        ttf_url = input_dic.get('ttf_path')
        embedding_type = input_dic.get('embedding_type', 'en')
        return_embedding = input_dic.get('return_embedding', False)
        if embedding_type not in ['en', 'zh']:
            result_dict = {"status": 1, "data": [],
                           "message": "embedding_type only supports 'en' or 'zh'"}
            self.finish(result_dict)
            return
        if not isinstance(return_embedding, bool):
            result_dict = {"status": 1, "data": [],
                           "message": "return_embedding must be boolean"}
            self.finish(result_dict)
            return
        
        
        try:
            font_code = input_dic['font_code']
            
        except:
            font_code = 'all'
        if font_code not in ['GB','GBK','all']:
            result_dict = {"status": 1, "data": [],
                           "message": f"font_code is not right"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        logger.info(f"收到的请求字符集{font_code}")
        url = f"http://{args.host}:11006/engetEmbedding"
        payload = {
            "token": token,
            "ttf_path": ttf_url,
            "embedding_type": embedding_type
        }
        #print(payload)
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        logger.info(f"Status Code: {response.status_code}")
        embedding_result = response.json()
        if embedding_result.get('status') != 0 or 'embedding' not in embedding_result:
            logger.info(f"embedding return INFO: {embedding_result}")
            self.finish(embedding_result)
            return
        embedding = embedding_result['embedding']
        embedding = [np.array(embedding)]
        if embedding_type == 'zh':
            if zh_vector_search is None:
                zh_vector_search = VectorSearch()
            result_dict = zh_vector_search.search_embedding(embedding)
        else:
            result_dict = en_vector_search.search(embedding,font_code)
        if return_embedding:
            result_dict['embedding_type'] = embedding_type
            result_dict['embedding_dimension'] = len(embedding_result['embedding'])
            result_dict['embedding'] = embedding_result['embedding']
        logger.info(f"return INFO: {result_dict}")
        self.finish(result_dict)

if __name__ == "__main__":
    application = web.Application(handlers=[(r"/recommend/fz_recommend/?", FzHandler),
                                            (r"/recommend/all_recommend/?", AllHandler),
                                            (r"/recommend/en_recommend/?", enHandler),
                                            (r"/recommend/symbol_recommend/?", symbolHandler)])
    application.listen(args.port)
    logger.info(f"init INFO: Running on url: http://0.0.0.0:{args.port}")
    ioloop.IOLoop.current().start()
