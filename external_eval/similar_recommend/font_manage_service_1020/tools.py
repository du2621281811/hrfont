import json
import os
import cv2
import time
import torch
import requests
import numpy as np
from pymilvus import connections, Collection
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms
from networks import coatnet_1
from extract_font_charset import get_charset

from log_init import logger, en_logger
from service_config import args


def unicode2chinese(unicode):
    return chr(int(unicode, 16))


def font2img(font_path, char):
    """
  通过字体文件和字符串渲染图像
  :param font_path:
  :param char:
  :return:
  """
    font = ImageFont.truetype(font_path, size=224)
    left, top, right, bottom = font.getbbox(char)
    font_width, font_height = right - left, bottom - top
    img = Image.new("RGB", (font_width, font_height), (255, 255, 255))  # 背景颜色
    draw = ImageDraw.Draw(img)
    draw.text((0 - left, 0 - top), char, (0, 0, 0), font=font)  # 字体颜色
    img = np.array(img)
    # 保存生成结果
    # generate_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 该图像生成时间
    #img_name = os.path.splitext(os.path.basename(font_path))[0] + f"-{char}.png"
    #cv2.imwrite(os.path.join(args.img_save_dir, img_name), img)

    return img


def download_font_file(url, save_dir):
    """
    通过http链接下载字体文件
    :param url:
    :return:
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Download error, status code: {response.status_code}")
        file_name = url.split('/')[-1]
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            f.write(response.content)
    except:
        return 1  # 1表示文件不存在
    return 0  # 0下载成功


class VectorGenerator:
    def __init__(self):
        self.input_shape = [224, 224]
        self.preprocess = transforms.Compose([transforms.Resize([224, 224]),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            # 中文模型初始化
            self.backbone = coatnet_1(self.input_shape)
            self.weight_state = torch.load(args.model_path,
                                           map_location=self.device)
            self.weight_state = self.weight_state['backbone_state_dict']
            self.backbone.load_state_dict(self.weight_state)
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()
            # 西文模型初始化
            self.en_backbone = coatnet_1(self.input_shape)
            self.weight_state = torch.load(args.en_model_path,
                                           map_location=self.device)
            self.weight_state = self.weight_state['backbone_state_dict']
            self.en_backbone.load_state_dict(self.weight_state)
            self.en_backbone = self.en_backbone.to(self.device)
            self.en_backbone.eval()
        except torch.cuda.OutOfMemoryError:
            logger.warning("init WARNING: CUDA out of memory, use cpu!")
            self.device = 'cpu'
            # 中文模型初始化
            self.backbone = coatnet_1(self.input_shape)
            self.weight_state = torch.load(args.model_path,
                                           map_location=self.device)
            self.weight_state = self.weight_state['backbone_state_dict']
            self.backbone.load_state_dict(self.weight_state)
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()
            # 西文模型初始化
            self.en_backbone = coatnet_1(self.input_shape)
            self.weight_state = torch.load(args.en_model_path,
                                           map_location=self.device)
            self.weight_state = self.weight_state['backbone_state_dict']
            self.en_backbone.load_state_dict(self.weight_state)
            self.en_backbone = self.en_backbone.to(self.device)
            self.en_backbone.eval()

        logger.info(f"init INFO: Use device: {self.device}")
        logger.info(f"init INFO: Model init success!")


    def cv2tensor(self, inputs):
        inputs_tendor = list()
        for inp in inputs:
            img = Image.fromarray(inp).convert('RGB')
            inputs_tendor.append(self.preprocess(img))
        return torch.stack(inputs_tendor)

    @torch.no_grad()
    def generate(self, batch_img):
        batch_input = self.cv2tensor(batch_img)
        batch_input = batch_input.to(self.device)
        batch_feature = self.backbone(batch_input)
        # feature l2 norm
        feature_norm = torch.sqrt(
            torch.sum(torch.square(batch_feature), dim=1))
        feature_norm = feature_norm.reshape([feature_norm.shape[0], 1])
        batch_feature = torch.div(batch_feature, feature_norm)

        return batch_feature

    @torch.no_grad()
    def en_generate(self, batch_img):
        """
        将一组字体图片(batch_img)转为字体 embedding 向量
        输入:
            batch_img: list 或 numpy array，字体图片列表，形状 [B, H, W, C] 或 [B, H, W]
        输出:
            font_embedding: 字体 embedding 向量，形状 [D]，已归一化
        """
        # 1. 图片转 tensor 并送入设备
        batch_input = self.cv2tensor(batch_img).to(self.device)  # [B, C, H, W]

        # 2. 网络前向，获取每张图片的特征向量
        batch_feature = self.en_backbone(batch_input)            # [B, D]

        # 3. 每张图片向量 L2 归一化
        batch_feature = batch_feature / batch_feature.norm(p=2, dim=1, keepdim=True)  # [B, D]

        # 4. 对 batch 内所有图片特征取均值
        font_embedding = batch_feature.mean(dim=0)               # [D]

        # 5. 最终向量再 L2 归一化
        font_embedding = font_embedding / font_embedding.norm()  # [D]

        return font_embedding

    @torch.no_grad()
    def generate_character_embeddings(self, batch_img, embedding_type):
        batch_input = self.cv2tensor(batch_img).to(self.device)
        if embedding_type == 'zh':
            batch_feature = self.backbone(batch_input)
        else:
            batch_feature = self.en_backbone(batch_input)
        batch_feature = batch_feature / batch_feature.norm(p=2, dim=1, keepdim=True)
        return batch_feature
    '''
    def en_generate(self, batch_img):
        batch_input = self.cv2tensor(batch_img)
        batch_input = batch_input.to(self.device)
        batch_feature = self.en_backbone(batch_input)
        # feature l2 norm
        feature_norm = torch.sqrt(
            torch.sum(torch.square(batch_feature), dim=1))
        feature_norm = feature_norm.reshape([feature_norm.shape[0], 1])
        batch_feature = torch.div(batch_feature, feature_norm)


        return batch_feature
    '''


class VectorManage:
    def __init__(self, load_collection=True):
        if not os.path.exists(args.font_save_dir):
            os.makedirs(args.font_save_dir)
        if load_collection:
            self.connect_alias = 'manage'
            connections.connect(
                alias=self.connect_alias,
                user='xxx',
                password='xxx',
                host=args.milvus_hostname,
                port=args.milvus_port
            )
            logger.info(f"init INFO: Milvus connect success!")
            self.collection_name = args.collection_name
            self.collection = Collection(name=self.collection_name, using=self.connect_alias)
            self.collection.load()  # 字体特征库用于验证font_id是否重复，所以加载进来
            logger.info(f"init INFO: get collections success: {self.collection_name}")
        self.vector_generator = VectorGenerator()
        # 打印当前版本号
        logger.info(f"init INFO: current dataset version: {self._get_version()}")


        # 获取6763字符集
        self.sort6763_dict = {}  # {'永': [永, ...]， '和': [和, ...]， '惠': [惠, ...]， '风': [风, ...]}  # 相似度排序
        for char in args.use_string:
            self.sort6763_dict[char] = []
        with open(args.sort6763_path, 'r') as f:
            for line in f.readlines():
                string_list = line.strip().split('\t')
                for i in range(len(args.use_string)):
                    self.sort6763_dict[args.use_string[i]].append(string_list[i])
    def get_support_string(self, local_font_path, mode):
        """
        检查字体是否支持'永和惠风'四个字，将不支持的替换成6763中支持的,替换规则按'永和惠风'相似度排序
        :param string:
        :param local_font_path:
        :param mode:
        :return:
        """
        # 支持字符集抽取
        support_charset = get_charset(local_font_path)
        logger.info(f"{mode} INFO: "
                    f"'{os.path.basename(local_font_path)}' support char num in 6763: {len(support_charset)}")
        # 字符太少，不做插入和更新
        if len(support_charset) < 100:
            return ''
        # 挑选渲染字符
        string_list = []
        for key, sort_values in self.sort6763_dict.items():  # 四个字
            for char in sort_values:  # 6763备选
                if char in support_charset:
                    string_list.append(char)
                    break

        logger.info(f"temp INFO: using string: {''.join(string_list)}")

        return ''.join(string_list)

    def get_embedding(self, font_file_url):
        file_name = font_file_url.split('/')[-1]
        if os.path.splitext(file_name)[-1].lower() not in ['.ttf', '.otf']:
            return {"status": 3,
                    "message": f"Un support font format: {os.path.splitext(file_name)[-1]},"
                               f"only support '.ttf' or '.otf'"}

        string = self.get_support_string(font_file_url, mode='getEmbedding')
        if len(string) < len(args.use_string):
            return {"status": 3,
                    "message": f"Font file support Chinese charset not enough: {font_file_url}"}

        img_list = []
        for char in string:
            img = font2img(font_path=font_file_url, char=char)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

        embedding = self.vector_generator.generate(img_list).cpu().numpy()
        embedding = np.concatenate(embedding, axis=0)
        return {"status": 0, "embedding": embedding.tolist()}

    def get_character_embeddings(self, font_file_url, characters, embedding_type):
        support_charset = get_charset(font_file_url)
        unsupported_characters = [char for char in characters if char not in support_charset]
        if unsupported_characters:
            return {"status": 3,
                    "message": "Font does not support all requested characters",
                    "unsupported_characters": unsupported_characters}

        img_list = []
        for char in characters:
            img = font2img(font_path=font_file_url, char=char)
            if img.shape[0] == 0 or img.shape[1] == 0:
                return {"status": 3,
                        "message": f"Character can not be rendered: {char}"}
            img_list.append(cv2.resize(img, (224, 224)))

        embeddings = self.vector_generator.generate_character_embeddings(
            img_list, embedding_type
        ).cpu().numpy()
        data = []
        for char, embedding in zip(characters, embeddings):
            data.append({"character": char, "embedding": embedding.tolist()})
        return {"status": 0,
                "embedding_type": embedding_type,
                "embedding_dimension": embeddings.shape[1],
                "data": data}

    def get_version(self):
        version = self._get_version()
        result_dict = {"status": 0,
                       "data": [{'version': version}],
                       "message": "Get Version success!"}

        return result_dict

    @staticmethod
    def _get_version():
        '''
        with open(args.version_save_path, 'r') as f:
            version = json.load(f)['version']
        return version
        '''
        return 'Version 1.00'
    @staticmethod
    def _update_version():

        #with open(args.version_save_path, 'w') as f:
        #    json_file = {'version': time.time()}
        #    f.write(json.dumps(json_file, ensure_ascii=False, indent=True))
        return True

    def _query(self, font_id):
        """
        内部私有查询接口，用于内部处理
        :param font_id:
        :return:
        """
        query_result = self.collection.query(
            expr=f"font_id == {font_id}",
            offset=0,
            limit=1,
            output_fields=['font_id', 'font_name', 'font_family', 'ttf_version', 'ttf_url', 'otf_version', 'otf_url']
        )

        return query_result

    def query(self, font_id):
        """
        对外查询接口
        :param font_id:
        :return:
        """
        query_result = self._query(font_id)
        if len(query_result) == 0:  # 字体不存在
            result_dict = {"status": 6,
                           "message": f"Query 'font_id' not exist: {font_id}"}
            return result_dict
        query_result = query_result[0]
        font_id = query_result['font_id']
        font_name = query_result['font_name']
        font_family = query_result['font_family']
        ttf_version = query_result['ttf_version']
        otf_version = query_result['otf_version']
        result_dict = {"status": 0, "data": [{'font_id': font_id, 'font_name': font_name, 'font_family': font_family,
                                              'ttf_version': ttf_version, 'otf_version': otf_version}],
                           "message": f"Query success!"}
        return result_dict


    def insert(self, font_id, font_name, font_family, is_fz_font, ttf_version, ttf_url, otf_version, otf_url):
        """
        :param font_id:
        :param font_name:
        :param font_family:
        :param is_fz_font:
        :param ttf_version: 字体版本号
        :param ttf_url: 字体文件下载地址
        :param otf_version:字体版本号
        :param otf_url: 字体文件下载地址， ttf和otf二选一
        :return:
        """
        # 检查font_id是否存在
        query_result = self._query(font_id)
        if len(query_result) != 0:  # True表示字体已经存在
            result_dict = {"status": 3,
                           "message": f"Insert 'font_id' is  exist: {font_id}"}
            return result_dict
        # 下载字体文件
        if ttf_url != '':
            font_file_url = ttf_url
        else:
            font_file_url = otf_url
        file_name = font_file_url.split('/')[-1]
        if os.path.splitext(file_name)[-1].lower() not in ['.ttf', '.otf']:  # 检查字体文件格式
            result_dict = {"status": 3,
                           "message": f"Un support font format: {os.path.splitext(file_name)[-1]},"
                                      f"only support '.ttf' or '.otf'"}
            return result_dict
        download_status = download_font_file(font_file_url, args.font_save_dir)
        if download_status == 1:  # 1表示下载不成功
            result_dict = {"status": 3,
                           "message": f"Download font file error: {font_file_url}"}
            return result_dict
        # 检查支持字符集
        local_font_path = os.path.join(args.font_save_dir, font_file_url.split('/')[-1])
        string = self.get_support_string(local_font_path, mode='insert')
        if len(string) < len(args.use_string):
            result_dict = {"status": 3,
                           "message": f"Insert font file support charset not enough: {font_file_url}"}
            return result_dict
        # 特征提取
        img_list = []

        for char in string:
            img = font2img(font_path=local_font_path, char=char)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

        embedding = self.vector_generator.generate(img_list)
        embedding = embedding.cpu()
        embedding = np.array(embedding)
        embedding = np.concatenate(embedding, axis=0)
        
        # 插入方正字体特征库
        if is_fz_font:
            self.collection.insert([{"font_id": font_id, "embedding": embedding, "font_name": font_name, "font_family": font_family,
     "ttf_version": ttf_version, "ttf_url": ttf_url, "otf_version": otf_version, "otf_url": otf_url}], partition_name='fz_font')
            logger.info(f"成功")
            logger.info(f"insert INFO: insert to '{self.collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: fz_font")
        # 插入第三方字体特征库
        else:
            self.collection.insert([[font_id], [embedding], [font_name], [font_family],
                                    [ttf_version], [ttf_url], [otf_version], [otf_url]], partition_name='other_font')
            logger.info(f"insert INFO: insert to '{self.collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: other_font")
        # 插入成功，更新版本号
        self._update_version()
        logger.info(f"insert INFO: current dataset version: {self._get_version()}")
        # 插入完成，返回
        result_dict = {"status": 0,
                       "message": f"Success insert, font id: {font_id}"}
        return result_dict

    def update(self, font_id, font_name, font_family, is_fz_font, ttf_version, ttf_url, otf_version, otf_url):
        """
        :param font_id:
        :param font_name:
        :param font_family:
        :param is_fz_font:
        :param ttf_version: 字体版本号
        :param ttf_url: 字体文件下载地址
        :param otf_version:字体版本号
        :param otf_url: 字体文件下载地址， ttf和otf二选一
        :return:
        """
        # 检查font_id是否存在
        query_result = self._query(font_id)  # 数据库中的信息，用来核对更新信息情况
        if len(query_result) == 0:  # 字体不存在
            result_dict = {"status": 4,
                           "message": f"Update 'font_id' not exist: {font_id}"}
            return result_dict
        query_result = query_result[0]
        # 下载字体文件
        if ttf_url != '':
            font_file_url = ttf_url
        else:
            font_file_url = otf_url
        file_name = font_file_url.split('/')[-1]
        if os.path.splitext(file_name)[-1].lower() not in ['.ttf', '.otf']:  # 检查字体文件格式
            result_dict = {"status": 3,
                           "message": f"Un support font format: {os.path.splitext(file_name)[-1]},"
                                      f"only support '.ttf' or '.otf'"}
            return result_dict
        download_status = download_font_file(font_file_url, args.font_save_dir)
        if download_status == 1:  # 1表示下载不成功
            result_dict = {"status": 3,
                           "message": f"Download font file error: {font_file_url}"}
            return result_dict
        # 检查支持字符集
        local_font_path = os.path.join(args.font_save_dir, font_file_url.split('/')[-1])
        string = self.get_support_string(local_font_path, mode='insert')
        if len(string) < len(args.use_string):
            result_dict = {"status": 3,
                           "message": f"Update font file support charset not enough: {font_file_url}"}
            return result_dict
        # 特征提取
        img_list = []

        for char in string:
            img = font2img(font_path=local_font_path, char=char)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

        embedding = self.vector_generator.generate(img_list)
        embedding = embedding.cpu()
        embedding = np.array(embedding)
        embedding = np.concatenate(embedding, axis=0)
        # 注意，update操作这里先做upsert再判断分区是否发生变化，分区发生变化则删除原分区数据。
        # 因为Milvus不检测主键重复，upsert分区变更，导致旧数据依旧存在旧分区，新分区存在新数据，但是主键相同。
        # 更新方正字体分区
        if is_fz_font:
            self.collection.upsert([[font_id], [embedding], [font_name], [font_family],
                                    [ttf_version], [ttf_url], [otf_version], [otf_url]], partition_name='fz_font')
            logger.info(f"insert INFO: Update to '{self.collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: fz_font")
        # 更新第三方字体分区
        else:
            self.collection.upsert([[font_id], [embedding], [font_name], [font_family],
                                    [ttf_version], [ttf_url], [otf_version], [otf_url]], partition_name='other_font')
            logger.info(f"insert INFO: Update to '{self.collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: other_font")
        # 判定分区是否存在变更，分区变更则删除原分区中的数据，防止数据重复
        if query_result['font_family'] == 'None' and font_family != 'None':  # 第三方字体 -> 方正字体
            self.collection.delete(expr=f"font_id == {font_id}", partition_name='other_font')
            logger.info(f"update INFO: partition change success, "
                        f"'{query_result['font_family']}' to '{font_family}'")
        elif query_result['font_family'] != 'None' and font_family == 'None':  # 方正字体 -> 第三方字体
            self.collection.delete(expr=f"font_id == {font_id}", partition_name='fz_font')
            logger.info(f"update INFO: partition change success, "
                        f"'{query_result['font_family']}' to '{font_family}'")
        # 修改成功，更新版本号
        self._update_version()
        logger.info(f"update INFO: current dataset version: {self._get_version()}")
        # 修改完成，返回
        result_dict = {"status": 0,
                       "message": f"Success update, font_id: {font_id}"}
        return result_dict

    def delete(self, font_id):
        # 检查font_id是否存在
        query_result = self._query(font_id)  # 数据库中的信息，删除记录埋点
        if len(query_result) == 0:  # False表示字体不存在
            result_dict = {"status": 1,
                           "message": f"Delete 'font_id' not exist: {font_id}"}
            return result_dict
        query_result = query_result[0]
        # 删除表达式
        expr = f"font_id == {font_id}"
        self.collection.delete(expr=expr)
        logger.info(f"delete INFO: delete '{self.collection_name}' collection, "
                    f"font_id: {font_id}, font_name: {query_result['font_name']}, "
                    f"font_family: {query_result['font_family']}")
        # 删除成功，更新版本号
        self._update_version()
        logger.info(f"delete INFO: current dataset version: {self._get_version()}")
        # 删除完成，返回
        result_dict = {"status": 0,
                       "message": f"Success delete font file: {font_id}"}
        return result_dict


class enVectorManage:
    def __init__(self):
        if not os.path.exists(args.en_font_save_dir):
            os.makedirs(args.en_font_save_dir)
        self.connect_alias = 'manage'
        connections.connect(
            alias=self.connect_alias,
            host=args.milvus_hostname,
            port=args.milvus_port
        )
        en_logger.info(f"init INFO: Milvus connect success!")
        self.en_collection_name = args.en_collection_name
        self.en_collection = Collection(name=self.en_collection_name, using=self.connect_alias)
        self.en_collection.load()  # 字体特征库用于验证font_id是否重复，所以加载进来
        en_logger.info(f"init INFO: get en collections success: {self.en_collection_name}")
        self.vector_generator = VectorGenerator()
        # 打印当前版本号
        en_logger.info(f"init INFO: current en dataset version: {self._get_version()}")

    def get_support_string(self, local_font_path, mode):
        """
        检查字体是否支持'永和惠风'四个字，将不支持的替换成6763中支持的,替换规则按'永和惠风'相似度排序
        :param string:
        :param local_font_path:
        :param mode:
        :return:
        """
        # 字母表备选
        alternate_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        # 支持字符集抽取
        support_charset = get_charset(local_font_path)
        # 字符太少，不做插入和更新
        if len(support_charset) < 10:
            return ''
        # 挑选渲染字符
        string_list = []
        for char in args.en_use_string:
            if char in support_charset:
                string_list.append(char)
            else:
                for alternate in alternate_charset:
                    if alternate in support_charset and alternate not in string_list:
                        string_list.append(alternate)
                        break

        en_logger.info(f"en temp INFO: using string: {''.join(string_list)}")

        return ''.join(string_list)

    def get_version(self):
        version = self._get_version()
        result_dict = {"status": 0,
                       "data": [{'version': version}],
                       "message": "Get Version success!"}

        return result_dict

    @staticmethod
    def _get_version():
        #with open(args.en_version_save_path, 'r') as f:
        #    version = json.load(f)['version']
        return 'Version 1.00'

    @staticmethod
    def _update_version():
        #with open(args.en_version_save_path, 'w') as f:
        #    json_file = {'version': time.time()}
        #    f.write(json.dumps(json_file, ensure_ascii=False, indent=True))
        return True

    def get_embedding(self,font_file_url):
        # 下载字体文件
        file_name = font_file_url.split('/')[-1]
        if os.path.splitext(file_name)[-1].lower() not in ['.ttf', '.otf']:  # 检查字体文件格式
            result_dict = {"status": 3,
                           "message": f"Un support font format: {os.path.splitext(file_name)[-1]},"
                                      f"only support '.ttf' or '.otf'"}
            return result_dict
        # 检查支持字符集
        local_font_path = font_file_url
        string = self.get_support_string(local_font_path, mode='enInsert')
        if len(string) < len(args.en_use_string):
            result_dict = {"status": 3,
                         "message": f"Insert font file support charset not enough: {font_file_url}"}
            return result_dict
        # 特征提取
        img_list = []
        for char in string:
            img = font2img(font_path=local_font_path, char=char)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

        embedding = self.vector_generator.en_generate(img_list)
        embedding = embedding.cpu().numpy()
        result_dict = {"status": 0,
                       "embedding": embedding.tolist()}  
        return result_dict

    def _query(self, font_id):
        """
        内部私有查询接口，用于内部处理
        :param font_id:
        :return:
        """
        query_result = self.en_collection.query(
            expr=f"font_id == {font_id}",
            offset=0,
            limit=1,
            output_fields=['font_id', 'font_name', 'ttf_version', 'ttf_url', 'otf_version', 'otf_url']
        )

        return query_result

    def is_same_embedding(self,embedding):
        # 用以判断库中是否有相同的embedding
        embedding = [embedding]
        search_params = {
            "metric_type": 'COSINE',
            "offset": 0,  # offset是跳过最相似的n个结果, limit是输出几个相似结果
            "ignore_growing": False,
            "params": {
                "radius": 0,
                "range_filter": 1  # 检索结果的分数区间为threshold~1.0
            }
        }
        # 开始检索
        res = self.en_collection.search(
            data=embedding,
            anns_field="embedding",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=3,
            expr='',  # 过滤信息
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['font_id'],
            consistency_level="Bounded"
        )[0]
        #en_logger.info(f"embedding结果为:{embedding}")
        if len(res)>0 and res[0].score >= 0.99:
            logger.info(f"存在相同符号：{res}")
            return False
        
        return True
    
    def insert(self, font_id, font_name, font_family, is_fz_font, ttf_version, ttf_url, otf_version, otf_url,GB):
        """
        :param font_id:
        :param font_name:
        :param font_family:
        :param is_fz_font:
        :param ttf_version: 字体版本号
        :param ttf_url: 字体文件下载地址
        :param otf_version:字体版本号
        :param otf_url: 字体文件下载地址， ttf和otf二选一
        :return:
        """
        # 检查font_id是否存在
        query_result = self._query(font_id)
        if len(query_result) != 0:  # True表示字体已经存在
            result_dict = {"status": 3,
                           "message": f"Insert 'font_id' is  exist: {font_id}"}
            return result_dict
        # 下载字体文件
        if ttf_url != '':
            font_file_url = ttf_url
        else:
            font_file_url = otf_url
        file_name = font_file_url.split('/')[-1]
        if os.path.splitext(file_name)[-1].lower() not in ['.ttf', '.otf']:  # 检查字体文件格式
            result_dict = {"status": 3,
                           "message": f"Un support font format: {os.path.splitext(file_name)[-1]},"
                                      f"only support '.ttf' or '.otf'"}
            return result_dict
        # 检查支持字符集
        local_font_path = font_file_url
        string = self.get_support_string(local_font_path, mode='enInsert')
        if len(string) < len(args.en_use_string):
            result_dict = {"status": 3,
                         "message": f"Insert font file support charset not enough: {font_file_url}"}
            return result_dict
        # 特征提取
        img_list = []
        for char in string:
            img = font2img(font_path=local_font_path, char=char)
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

        embedding = self.vector_generator.en_generate(img_list)
        embedding = embedding.cpu().numpy()  
        #embedding = np.array(embedding)
        #embedding = np.concatenate(embedding, axis=0)
        is_same = self.is_same_embedding(embedding)
        if not is_same:
            result_dict = {"status": 1,
                       "message": f"insert error, font id: {font_id} embedding excit"}
            return result_dict
        # 插入方正字体特征库
        if is_fz_font:
            self.en_collection.insert([{"font_id": font_id, "embedding": embedding, "font_name": font_name, "font_family": font_family,
     "ttf_version": ttf_version, "ttf_url": ttf_url, "otf_version": otf_version, "otf_url": otf_url,"GB_GBK":GB}], partition_name='fz_font')
            logger.info(f"insert INFO: insert to '{self.en_collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: fz_font")
        # 插入第三方字体特征库
        else:
            self.en_collection.insert([[font_id], [embedding], [font_name], [font_family],
                                      [ttf_version], [ttf_url], [otf_version], [otf_url]], partition_name='other_font')
            logger.info(f"insert INFO: insert to '{self.en_collection_name}' collection, "
                        f"font_id: {font_id}, font_name: {font_name}, font_family: {font_family}, "
                        f"ttf_version: {ttf_version}, ttf_url: {ttf_url}, "
                        f"otf_version: {otf_version}, otf_url: {otf_url}, "
                        f"partition_name: other_font")
        # 插入成功，更新版本号
        self._update_version()
        logger.info(f"insert INFO: current dataset version: {self._get_version()}")
        # 插入完成，返回
        result_dict = {"status": 0,
                       "message": f"Success insert, font id: {font_id}"}
        return result_dict


if __name__ == "__main__":
    pass
