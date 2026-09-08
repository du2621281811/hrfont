# -*- coding: utf-8 -*-
import os
import json
import traceback
from tornado import web
from tornado import ioloop
from tools import VectorManage, enVectorManage
from service_config import args
from log_init import logger, en_logger
from ftplib import FTP
import subprocess
import re
import tempfile




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


class insertHandler(web.RequestHandler):
        
    def post(self):
        self.vector_manage = VectorManage()
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数有无缺失
        for need_argument in ['font_id', 'font_name', 'font_family', 'is_fz_font']:
            if input_argument.get(need_argument, '') == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        # ttf_version和otf_version格式检查
        ttf_version = input_argument.get('ttf_version', '')
        otf_version = input_argument.get('otf_version', '')
        if not isinstance(ttf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'ttf_version' need string type, "
                                      f"but received type: {type(ttf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(otf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'otf_version' need string type, "
                                      f"but received type: {type(otf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # ttf_url和otf_url检查
        ttf_url = input_argument.get('ttf_url', '')
        otf_url = input_argument.get('otf_url', '')
        if ttf_url == '' and otf_url == '':  # 不能同时为空
            result_dict = {"status": 1,
                           "message": f"Except argument: 'ttf_url' or 'otf_url'"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if ttf_url is not None and ttf_url != '' \
                and os.path.splitext(ttf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'ttf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(ttf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if otf_url is not None and otf_url != '' \
                and os.path.splitext(otf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'otf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(otf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数格式
        font_id = input_argument.get('font_id')
        font_name = input_argument.get('font_name')
        font_family = input_argument.get('font_family')
        is_fz_font = input_argument.get('is_fz_font')
        if not isinstance(font_name, str):  # font_name必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_name' need string type, "
                                      f"but received type: {type(font_name)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(font_family, str):  # font_family必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_family' need string type, "
                                      f"but received type: {type(font_family)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # font_id必须可以转换为int
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # is_fz_font必须可以转换为int
            is_fz_font = int(is_fz_font)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' can not convert to int: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数逻辑限制
        if is_fz_font not in [0, 1]:  # is_fz_font只能为0或1
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' only support 0 and 1, but received: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 1 and font_family == 'None':  # 是方正字体但是font_family为None
            result_dict = {"status": 1, "data": [],
                           "message": "Founder font family can not equal to 'None'."}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 0 and font_family != 'None':  # 非方正字体但是font_family不为None
            result_dict = {"status": 1,
                           "message":
                               f"The font is not founder font, argument 'font_family' can only equal to 'None', "
                               f"but received: {font_family}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 输入检查通过，进入插入流程
        try:
            result_dict = self.vector_manage.insert(font_id, font_name, font_family, is_fz_font,
                                               ttf_version, ttf_url, otf_version, otf_url)
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"insert ERROR: {traceback.format_exc()}")
            result_dict = {"status": 3,
                           "message": "Insert error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)



class updateHandler(web.RequestHandler):
        
    def post(self):
        self.vector_manage = VectorManage()
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数有无缺失
        for need_argument in ['font_id', 'font_name', 'font_family', 'is_fz_font']:
            if input_argument.get(need_argument, '') == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        # ttf_version和otf_version格式检查
        ttf_version = input_argument.get('ttf_version', '')
        otf_version = input_argument.get('otf_version', '')
        if not isinstance(ttf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'ttf_version' need string type, "
                                      f"but received type: {type(ttf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(otf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'otf_version' need string type, "
                                      f"but received type: {type(otf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # ttf_url和otf_url检查
        ttf_url = input_argument.get('ttf_url', '')
        otf_url = input_argument.get('otf_url', '')
        if ttf_url == '' and otf_url == '':  # 不能同时为空
            result_dict = {"status": 1,
                           "message": f"Except argument: 'ttf_url' or 'otf_url'"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if ttf_url is not None and ttf_url != '' \
                and os.path.splitext(ttf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'ttf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(ttf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if otf_url is not None and otf_url != '' \
                and os.path.splitext(otf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'otf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(otf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数格式
        font_id = input_argument.get('font_id')
        font_name = input_argument.get('font_name')
        font_family = input_argument.get('font_family')
        is_fz_font = input_argument.get('is_fz_font')
        if not isinstance(font_name, str):  # font_name必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_name' need string type, "
                                      f"but received type: {type(font_name)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(font_family, str):  # font_family必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_family' need string type, "
                                      f"but received type: {type(font_family)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # font_id必须可以转换为int
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # is_fz_font必须可以转换为int
            is_fz_font = int(is_fz_font)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' can not convert to int: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数逻辑限制
        if is_fz_font not in [0, 1]:  # is_fz_font只能为0或1
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' only support 0 and 1, but received: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 1 and font_family == 'None':  # 是方正字体但是font_family为None
            result_dict = {"status": 1, "data": [],
                           "message": "Founder font family can not equal to 'None'."}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 0 and font_family != 'None':  # 非方正字体但是font_family不为None
            result_dict = {"status": 1,
                           "message":
                               f"The font is not founder font, argument 'font_family' can only equal to 'None', "
                               f"but received: {font_family}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 输入检查通过，进入更新流程
        try:
            result_dict = self.vector_manage.update(font_id, font_name, font_family, is_fz_font,
                                               ttf_version, ttf_url, otf_version, otf_url)
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"update ERROR: {traceback.format_exc()}")
            result_dict = {"status": 3,
                           "message": "Update error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class deleteHandler(web.RequestHandler):
    def post(self):
        self.vector_manage = VectorManage()
        # 记录请求信息
        logger.info(f" request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数有无缺失
        for need_argument in ['font_id']:
            if input_argument.get(need_argument) is None or input_argument.get(need_argument) == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        # 检查输入参数格式
        font_id = input_argument.get('font_id')
        try:  # font_id必须可以转换为int
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 输入检查通过，进入删除流程
        try:
            result_dict = self.vector_manage.delete(font_id)
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"delete ERROR: {traceback.format_exc()}")
            result_dict = {"status": 5,
                           "message": "Delete error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class queryHandler(web.RequestHandler):
        
    def post(self):
        self.vector_manage = VectorManage()
        # 记录请求信息
        logger.info(f" request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数有无缺失
        for need_argument in ['font_id']:
            if input_argument.get(need_argument) is None or input_argument.get(need_argument) == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        # 检查输入参数格式
        font_id = input_argument.get('font_id')
        try:  # font_id必须可以转换为int
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 输入检查通过，进入查询流程
        try:
            result_dict = self.vector_manage.query(font_id)
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"query ERROR: {traceback.format_exc()}")
            result_dict = {"status": 6,
                           "message": "Query error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class getVersionHandler(web.RequestHandler):
        
    def post(self):
        self.vector_manage = VectorManage()
        # 记录请求信息
        logger.info(f" request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return

        # 输入检查通过，进入获取版本号流程
        try:
            result_dict = self.vector_manage.get_version()
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            logger.error(f"getVersion ERROR: {traceback.format_exc()}")
            result_dict = {"status": 7,
                           "message": "Get version error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


###################西文字体推荐服务###################
class enInsertHandler(web.RequestHandler):
        
    def post(self):
        self.en_vector_manage = enVectorManage()
        # 记录请求信息
        en_logger.info(f"request INFO: {self.request}")
        en_logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数有无缺失
        for need_argument in ['font_id', 'font_name', 'font_family', 'is_fz_font']:
            if input_argument.get(need_argument, '') == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        # ttf_version和otf_version格式检查
        ttf_version = input_argument.get('ttf_version', '')
        otf_version = input_argument.get('otf_version', '')
        if not isinstance(ttf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'ttf_version' need string type, "
                                      f"but received type: {type(ttf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(otf_version, str):
            result_dict = {"status": 1,
                           "message": f"Input argument 'otf_version' need string type, "
                                      f"but received type: {type(otf_version)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # ttf_url和otf_url检查
        ttf_url = input_argument.get('ttf_url', '')
        otf_url = input_argument.get('otf_url', '')
        if ttf_url == '' and otf_url == '':  # 不能同时为空
            result_dict = {"status": 1,
                           "message": f"Except argument: 'ttf_url' or 'otf_url'"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if ttf_url is not None and ttf_url != '' \
                and os.path.splitext(ttf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'ttf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(ttf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if otf_url is not None and otf_url != '' \
                and os.path.splitext(otf_url.split('/')[-1])[-1].lower() not in ['.ttf', '.otf']:  # 不支持的字体类型
            result_dict = {"status": 1,
                           "message": f"Un support font type for 'otf_url', "
                                      f"except '.ttf' or '.otf', "
                                      f"but received:{os.path.splitext(otf_url.split('/')[-1])[-1]}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数格式
        font_id = input_argument.get('font_id')
        font_name = input_argument.get('font_name')
        font_family = input_argument.get('font_family')
        is_fz_font = input_argument.get('is_fz_font')
        GB = input_argument.get('GB_GBK')
        if not isinstance(font_name, str):  # font_name必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_name' need string type, "
                                      f"but received type: {type(font_name)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if not isinstance(font_family, str):  # font_family必须为字符串
            result_dict = {"status": 1,
                           "message": f"Input argument 'font_family' need string type, "
                                      f"but received type: {type(font_family)}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # font_id必须可以转换为int
            font_id = int(font_id)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'font_id' can not convert to int: {font_id}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:  # is_fz_font必须可以转换为int
            is_fz_font = int(is_fz_font)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' can not convert to int: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 检查输入参数逻辑限制
        if is_fz_font not in [0, 1]:  # is_fz_font只能为0或1
            result_dict = {"status": 1, "data": [],
                           "message": f"Input argument 'is_fz_font' only support 0 and 1, but received: {is_fz_font}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 1 and font_family == 'None':  # 是方正字体但是font_family为None
            result_dict = {"status": 1, "data": [],
                           "message": "Founder font family can not equal to 'None'."}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        if is_fz_font == 0 and font_family != 'None':  # 非方正字体但是font_family不为None
            result_dict = {"status": 1,
                           "message":
                               f"The font is not founder font, argument 'font_family' can only equal to 'None', "
                               f"but received: {font_family}"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        
        # 输入检查通过，进入插入流程
        try:
            result_dict = self.en_vector_manage.insert(font_id, font_name, font_family, is_fz_font,
                                                  ttf_version, ttf_url, otf_version, otf_url,GB)
            en_logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            en_logger.error(f"insert ERROR: {traceback.format_exc()}")
            result_dict = {"status": 3,
                           "message": "Insert error!"}
            en_logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)


class enGetVersionHandler(web.RequestHandler):
        
    def post(self):
        self.en_vector_manage = enVectorManage()
        # 记录请求信息
        en_logger.info(f" request INFO: {self.request}")
        en_logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            en_logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        # 输入检查通过，进入获取版本号流程
        try:
            result_dict = self.en_vector_manage.get_version()
            en_logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
        except:
            en_logger.error(f"getVersion ERROR: {traceback.format_exc()}")
            result_dict = {"status": 7,
                           "message": "Get version error!"}
            en_logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)

character_vector_manage = None


class characterEmbeddingHandler(web.RequestHandler):
    def post(self):
        global character_vector_manage
        logger.info(f"request INFO: {self.request}")
        check_status, input_argument, result_dict = check_input(self.request.body)
        if not check_status:
            self.finish(result_dict)
            return

        ttf_url = input_argument.get('ttf_path', '')
        characters = input_argument.get('characters', '')
        embedding_type = input_argument.get('embedding_type', 'zh')
        if not ttf_url:
            self.finish({"status": 1, "message": "Except input argument: ttf_path"})
            return
        if not isinstance(characters, str) or not characters:
            self.finish({"status": 1, "message": "characters must be a non-empty string"})
            return
        if len(characters) > 128:
            self.finish({"status": 1, "message": "characters supports at most 128 characters"})
            return
        if embedding_type not in ['en', 'zh']:
            self.finish({"status": 1,
                         "message": "embedding_type only supports 'en' or 'zh'"})
            return

        suffix = os.path.splitext(ttf_url)[-1].lower()
        if suffix not in ['.ttf', '.otf']:
            self.finish({"status": 1, "message": "Only support .ttf or .otf file"})
            return

        os.makedirs('FTP_files', exist_ok=True)
        local_font_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    dir='FTP_files', suffix=suffix, delete=False) as font_file:
                local_font_path = font_file.name
                with FTP() as ftp:
                    ftp.connect(host=args.ftp_host, port=args.ftp_port)
                    ftp.login(user=args.ftp_user, passwd=args.ftp_password)
                    ftp.retrbinary('RETR ' + ttf_url, font_file.write)

            if character_vector_manage is None:
                character_vector_manage = VectorManage(load_collection=False)
            result_dict = character_vector_manage.get_character_embeddings(
                local_font_path, characters, embedding_type
            )
            self.finish(result_dict)
        except Exception:
            logger.error(f"getCharacterEmbeddings ERROR: {traceback.format_exc()}")
            if not self._finished:
                self.finish({"status": 3, "message": "Get character embeddings error!"})
        finally:
            if local_font_path and os.path.exists(local_font_path):
                os.remove(local_font_path)


class getembeddingHandler(web.RequestHandler):
        
    def post(self):
        # 记录请求信息
        logger.info(f"request INFO: {self.request}")
        logger.info(f"request body INFO: {self.request.body}")
        # 请求输入参数转json格式
        check_status, input_argument, result_dict = check_input(self.request.body)
        # 检查输入参数有无缺失
        for need_argument in ['ttf_path']:
            if input_argument.get(need_argument, '') == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return
        ttf_url = input_argument.get('ttf_path')
        embedding_type = input_argument.get('embedding_type', 'en')
        if embedding_type not in ['en', 'zh']:
            result_dict = {"status": 1, "data": [],
                           "message": "embedding_type only supports 'en' or 'zh'"}
            self.finish(result_dict)
            return
        if embedding_type == 'zh':
            vector_manage = VectorManage(load_collection=False)
        else:
            vector_manage = enVectorManage()
        ftp = FTP()
        ftp.connect(host=args.ftp_host, port=args.ftp_port)
        ftp.login(user='ftpuser', passwd='12345678')
        ttf_path = 'FTP_files/target.TTF'
        try:
            # 下载ttf文件
            with open(ttf_path, 'wb') as f:
                ftp.retrbinary('RETR ' + ttf_url, f.write)
        except:
            result_dict = {"status": 1, "data": [],
                           "message": f"download TTF error"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
            return
        try:
            result_dict = vector_manage.get_embedding(ttf_path)
            self.finish(result_dict)
        except:
            logger.error(f"insert ERROR: {traceback.format_exc()}")
            result_dict = {"status": 3,
                           "message": "Get embedding error!"}
            logger.info(f"return INFO: {result_dict}")
            self.finish(result_dict)
'''
class getembeddingHandler(web.RequestHandler):

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

        # 检查输入参数有无缺失
        for need_argument in ['ttf_path']:
            if input_argument.get(need_argument, '') == '':
                result_dict = {"status": 1,
                               "message": f"Except input argument: {need_argument}"}
                logger.info(f"return INFO: {result_dict}")
                self.finish(result_dict)
                return

        ttf_url = input_argument.get('ttf_path')
        ttf_path = 'FTP_files/target.TTF'

        # 使用 FTP 下载 TTF 文件
        try:
            ftp = FTP()
            ftp.connect(host=args.ftp_host, port=args.ftp_port)
            ftp.login(user='ftpuser', passwd='12345678')
            with open(ttf_path, 'wb') as f:
                ftp.retrbinary('RETR ' + ttf_url, f.write)
        except Exception as e:
            result_dict = {"status": 1, "data": [],
                           "message": f"Download TTF error: {str(e)}"}
            logger.error(f"Download TTF failed: {str(e)}")
            self.finish(result_dict)
            return

        # 启动子进程执行模型推理
        try:
            # 启动子进程调用 `run_embedding_task.py` 并传递 `--ttf_path` 参数
            process = subprocess.Popen(
                ['python3', 'run_embedding_task.py'],  # 传递正确的命令行参数
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()  # 等待子进程完成
            logger.info(f"stdout result: {type(stdout.decode())}")
            output = stdout.decode()
            match = re.search(r'(\{.*\})', output)
            logger.info(f"stderr result: {match}")
            if process.returncode == 0:
                # 将子进程输出作为结果
                logger.info(f"Embedding result: {stdout.decode()}")
                result_dict = json.loads(stdout.decode())  # 假设返回的是 JSON 格式的结果
            else:
                logger.error(f"Error during embedding process: {stderr.decode()}")
                result_dict = {"status": 3, "message": "Get embedding error!"}

            self.finish(result_dict)
        except Exception as e:
            logger.error(f"Error starting subprocess: {str(e)}")
            result_dict = {"status": 3, "message": "Error running embedding task!"}
            self.finish(result_dict)
'''

if __name__ == "__main__":
    # 目录初始化
    if not os.path.exists(os.path.dirname(args.version_save_path)):
        os.makedirs(os.path.dirname(args.version_save_path))
        logger.info(f"init INFO: create dir: {os.path.dirname(args.version_save_path)}")
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))
        logger.info(f"init INFO: create dir: {os.path.dirname(args.log_path)}")
    if not os.path.exists(args.font_save_dir):
        os.makedirs(args.font_save_dir)
        logger.info(f"init INFO: create dir: {args.font_save_dir}")
    if not os.path.exists(args.img_save_dir):
        os.makedirs(args.img_save_dir)
        logger.info(f"init INFO: create dir: {args.img_save_dir}")
    # 服务初始化
    application = web.Application(handlers=[(r"/insert/?", insertHandler),
                                            (r"/update/?", updateHandler),
                                            (r"/delete/?", deleteHandler),
                                            (r"/query/?", queryHandler),
                                            (r"/getVersion/?", getVersionHandler),
                                            (r"/enInsert/?", enInsertHandler),
                                            (r"/engetEmbedding/?", getembeddingHandler),
                                            (r"/getCharacterEmbeddings/?", characterEmbeddingHandler),
                                            (r"/enGetVersion/?", enGetVersionHandler)], debug=False)
    application.listen(args.port)
    logger.info(f"init INFO: Running on url: http://0.0.0.0:{args.port}")
    ioloop.IOLoop.current().start()
