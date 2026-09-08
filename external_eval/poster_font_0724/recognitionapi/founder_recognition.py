# -*- coding: utf-8 -*-
import cv2
import codecs
import settings
import numpy as np
from pythonapi import preprocess
from collections import defaultdict
from recognitionapi import coatnet_infer
import threshold_filter_font
import torch
import torch.nn.functional as F
import json
from pythonapi import recog_process



series_font_dict = threshold_filter_font.series_font_dict

with codecs.open(settings.char_8636_path, mode = 'r', encoding = "utf-8") as f:
    charlist = f.readlines()
f.close()
charlist = [info.strip() for info in charlist]

char8636 = charlist

with codecs.open(settings.similar_font_dict,mode='r',encoding="utf-8")  as f:
    similar_font_dict = json.load(f)
f.close()


with codecs.open(settings.LT_font_path,mode='r',encoding="utf-8")  as f:
    LT_list = f.readlines()
f.close()
LT_list = [info.strip() for info in LT_list]

def add_border_function(image_list):

    new_image_list = list()
    for image in image_list:
        histb = cv2.calcHist([image], [0], None, [256], [0, 255])
        histg = cv2.calcHist([image], [1], None, [256], [0, 255])
        histr = cv2.calcHist([image], [2], None, [256], [0, 255])

        histb = histb.tolist()
        histg = histg.tolist()
        histr = histr.tolist()

        value_b = histb.index(np.max(histb, axis=0)[0])
        value_g = histg.index(np.max(histg, axis=0)[0])
        value_r = histr.index(np.max(histr, axis=0)[0])

        height, width = image.shape[:2]
        value = [value_b, value_g, value_r]
        borderType = cv2.BORDER_CONSTANT

        if height > width:
            padding_left = int((height - width) / 2.0)
            padding_right = int((height - width) / 2.0)
            padding_top = 0
            padding_bottom = 0
            dst = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, borderType, None,
                                     value)
        elif height < width:
            padding_top = int((width - height) / 2.0)
            padding_bottom = int((width - height) / 2.0)
            padding_left = 0
            padding_right = 0
            dst = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, borderType, None,
                                     value)
        else:
            padding_top = 0
            padding_bottom = 0
            padding_left = 0
            padding_right = 0
            dst = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, borderType, None,
                                     value)

        dst = cv2.resize(dst, (224, 224))

        new_image_list.append(dst)

    return new_image_list

def area_split(font_recognition_result_list,ocr_image_list_oir):

    arcface_featrue = coatnet_infer.head_state_dict["weight"]
    split_list = list()

    for i in range(1,len(font_recognition_result_list)):

        # if int(font_recognition_result_list[i]["id"]) > 2000000:
        #     continue
        position_list = font_recognition_result_list[i]["position_list"]
        series_flag = 0
        for k, v in series_font_dict.items():
            if font_recognition_result_list[0]["font_name_ch"] in v and font_recognition_result_list[i]["font_name_ch"] in v:
                series_flag = 1
        if series_flag == 1:
            continue
        dif = 0
        if font_recognition_result_list[i]["count"] > 1:
            for j in range(len(position_list)-1):
                dif += (int(position_list[j+1])-int(position_list[j]))
            if dif == len(position_list)-1:
                split_list.append(font_recognition_result_list[i]["index"])
        else:

            target_height = ocr_image_list_oir[int(position_list[0])]["char_height"]
            char_height_extra = 0
            for j in range(len(ocr_image_list_oir)):
                if j != int(position_list[0]):
                    temp_height = ocr_image_list_oir[j]["char_height"]
                    char_height_extra += temp_height
            char_height_extra = char_height_extra/(len(ocr_image_list_oir)-1)

            if target_height > char_height_extra*1.2:
                split_list.append(font_recognition_result_list[i]["index"])

    if len(split_list) == 0:
        return [], 0

    need_split_list = list()

    top1_feature = arcface_featrue[int(font_recognition_result_list[0]["index"])]
    top1_feature = top1_feature.unsqueeze(0)
    top1_feature = F.normalize(top1_feature)
    top1_feature.cuda()

    for temp in split_list:
        temp_feature = arcface_featrue[int(temp)]
        temp_feature = temp_feature.unsqueeze(0)
        temp_feature = F.normalize(temp_feature)
        temp_feature.cuda()
        mx = torch.mm(top1_feature, temp_feature.T)
        if mx < 0.5:
            need_split_list.append(temp)

    if len(need_split_list) == 0:
        return [], 0

    need_split_list.append(font_recognition_result_list[0]["index"])

    font_recognition_result_list_temp = list()
    for temp in need_split_list:
        for info in font_recognition_result_list:
            if temp == info["index"]:
                font_recognition_result_list_temp.append(info)

    series_flag = 1
    font_recognition_result_list_new = list()
    for temp in font_recognition_result_list_temp:
        position_temp = temp["position_list"]
        result = list()
        font_list = list()
        for info in position_temp:
            result.append(ocr_image_list_oir[int(info)])

        if len(result) > 1:
            top = result[0]["top"]
            left = result[0]["left"]
            right = result[0]["right"]
            bottom = result[0]["bottom"]
            for mm in result[1:]:
                top = min(top,mm["top"])
                left = min(left, mm["left"])
                right = max(right, mm["right"])
                bottom = max(bottom, mm["bottom"])
        else:
            top = result[0]["top"]
            left = result[0]["left"]
            right = result[0]["right"]
            bottom = result[0]["bottom"]
        char_new = ""
        for p in position_temp:
            if int(p) <= len(font_recognition_result_list[0]["char_result"])-1:
                char_new += font_recognition_result_list[0]["char_result"][int(p)]

        if char_new == "":
            char_new = font_recognition_result_list[0]["char_result"]

        image_list_new = list()
        for p in position_temp:
            image_list_new.append(ocr_image_list_oir[int(p)]["image"])

        font_list.append(
            {"font_name_ch": temp["font_name_ch"], "id": temp["id"], "ttfname": temp["ttfname"],
             "score":temp["score"],"count":temp["count"]})

        font_recognition_result_list_new.append({"info":temp,"char_result":char_new,"image_list":image_list_new,"font_list":font_list, "left":left,"right":right,"top":top,"bottom":bottom})


    return  font_recognition_result_list_new,series_flag



def font_makelist(font_dict,char_result, single_font_threshold):

    font_recognition_result_dict = defaultdict(list)
    font_recognition_result_list = list()

    for i in range(len(font_dict)):
        if font_dict[i]["probably"] < single_font_threshold:
            continue
        font_recognition_result_dict[font_dict[i]["label"] + "#" + font_dict[i]["font_id"]+"#"+font_dict[i]["ttfname"]+"#"+str(font_dict[i]["index"])].append({"probably":font_dict[i]["probably"],"position":str(i)})


    for k,v in font_recognition_result_dict.items():
        sum_score = 0
        position_list = list()
        for info in v:
            sum_score += info["probably"]
            position_list.append(info["position"])
        avg_score = sum_score/float(len(v))
        font_recognition_result_list.append({"font_name_ch":k.split("#")[0],"id":int(k.split("#")[1]),"ttfname":k.split("#")[2],"index":k.split("#")[3],"position_list":position_list,"char_result":char_result, "score":avg_score,"count":len(v)})

    font_recognition_result_list.sort(key=lambda o: (-o["count"], -o["score"]))

    return font_recognition_result_list

def recognition(img, char_result, Bboxes):



    ocr_image_list_oir = list()
    ocr_image_list = list()
    char_height_sum = 0


    x_list = list()
    y_list = list()
    tmp = list()
    for Bbox in Bboxes:
        top = min(Bbox[:, 1])
        bottom = max(Bbox[:, 1])
        left = min(Bbox[:, 0])
        right = max(Bbox[:, 0])
        x_list.append(int((left+right)/2))
        y_list.append(int((top+bottom)/2))
        tmp.append([int((left+right)/2), int((top+bottom)/2)])
    if len(Bboxes) > 1:
        ####拟合直线####
        fitline = cv2.fitLine(np.array(tmp), cv2.DIST_L2, 0, 0.01, 0.01)
        grad = fitline[1] / fitline[0]
        ####拟合直线####
    else:
        grad = [0]


    for Bbox in Bboxes:

        if int(max(Bbox[:, 1])) - int(min(Bbox[:, 1])) <= settings.ver_recog_height:
            ver_border = 1
            hor_border = 1
        else:
            ver_border = int((int(max(Bbox[:, 1])) - int(min(Bbox[:, 1])) + int(max(Bbox[:, 0])) - int(min(Bbox[:, 0]))) / (2*10.0))
            hor_border = max(1, int((ver_border + 1)/2.0))
        top = max(0, int(min(Bbox[:, 1])) - ver_border)
        bottom = min(img.shape[0], int(max(Bbox[:, 1])) + ver_border)
        left = max(0, int(min(Bbox[:, 0])) - hor_border)
        right = min(img.shape[1], int(max(Bbox[:, 0])) + hor_border)
        crop_img = img[top:bottom, left:right]

        ####字符倾斜矫正模块####
        if abs(grad[0]) > 0.15 and abs(grad[0]) < 1.732:
            pixelValue = preprocess.extracting_boundary_pixels(crop_img)
            crop_img = preprocess.rotateImage(crop_img, grad[0] / np.pi * 180, pixelValue)

        cropImg,binary_img, char_width,char_height = preprocess.preprocession(crop_img)


        # #######过滤小于20的图片######
        if char_height < settings.max_area_size:
            continue

        char_height_sum += char_height
        ocr_image_list_oir.append({"image":cropImg,"sort_index":top+left,"char_height":char_height,"left":left,"right":right,"top":top,"bottom":bottom})

    if len(ocr_image_list_oir) == 0:
        return [], [], 0, []

    ocr_image_list_oir.sort(key=lambda o: (o["sort_index"]))

    for temp in ocr_image_list_oir:
        ocr_image_list.append(temp["image"])

    char_height_avg = int(char_height_sum/float(len(ocr_image_list)))

    font_image_list = ocr_image_list


    #######添加于底色相近的边缘,将图片修正为正方形######
    font_image_list = add_border_function(font_image_list)

     #######识别模块#####

    font_dict,font_dict1 = coatnet_infer.evaluate(font_image_list, settings.topk)

    font_recognition_result_list = font_makelist(font_dict, char_result, settings.single_font_threshold)
    font_recognition_result_all_list = font_makelist(font_dict, char_result, 0)

    if len(font_recognition_result_list) > 0:
        if len(font_recognition_result_list) >= 2 and len(ocr_image_list_oir) >= 2:
            font_recognition_result_list_split, series_flag = area_split(font_recognition_result_list,
                                                                         ocr_image_list_oir)

        else:
            font_recognition_result_list_split = list()

    else:
        font_recognition_result_list_split = list()

    font_recognition_result_list_final = []

    for temp in font_recognition_result_list:
        if len(char_result) / temp["count"] <= 2:
            font_recognition_result_list_final.append(temp)

    font_recognition_result_list_split_final = []
    for temp1 in font_recognition_result_list_split:
        temp_list = list()
        for temp2 in temp1:
            if len(char_result) / temp2["count"] <= 2:
                temp_list.append(temp2)
        if len(temp_list) > 0:
            font_recognition_result_list_split_final.append(temp_list)

    return font_recognition_result_all_list, font_recognition_result_list_final,font_recognition_result_list_split_final,char_height_avg,font_image_list


# results_all_information[jpgfile].append(
#     {"labelid": font_list[0]["id"], "ttflabel": font_list[0]["ttfname"], "probably": font_list[0]["score"],
#      "count": font_list[0]["count"], "char_result": char_result, "char_result_oir": char_result_oir, 'left': left,
#      'top': top, 'right': right, 'bottom': bottom, "label": font_list[0]["font_name_ch"],
#      "string_height": char_height_avg, "sort_index": sort_index, "image_height": oirimg_src.shape[0]})


def recognition_other_levels(results_information,results_copyright_information):

    results_high_information = defaultdict(list)
    results_medium_information = defaultdict(list)
    results_low_information = defaultdict(list)

    for k, v in results_information.items():
        for info in v:
            if info["probably"] > 0.7 and len(info["char_result"])/info["count"] < 2:
                if info["label"] in LT_list:
                    if info["probably"] > 0.95 and len(info["char_result"]) / info["count"] < 2:
                        results_high_information[k].append({"info":info,"mode_index":"0111"})
                    else:
                        results_medium_information[k].append({"info": info, "mode_index": "0011"})
                else:
                    results_high_information[k].append({"info": info, "mode_index": "0111"})

            elif info["probably"] > 0.5 and len(info["char_result"])/info["count"] <= 2:
                results_medium_information[k].append({"info":info,"mode_index":"0011"})
            else:
                results_low_information[k].append({"info":info,"mode_index":"0001"})

    results_all_information = defaultdict(list)

    for k,v in results_high_information.items():
        for temp in v:
            results_all_information[k].append(
                {"labelid": temp["info"]["labelid"],
                 "ttflabel": temp["info"]["ttflabel"],
                 "label": temp["info"]["label"],
                 "probably": temp["info"]["probably"],
                 'chinesestr': temp["info"]["char_result_oir"], 'left': temp["info"]["left"],
                 'top': temp["info"]["top"], 'right': temp["info"]["right"], 'bottom': temp["info"]["bottom"],"string_height":temp["info"]["string_height"], "sort_index":0.0,"mode_index":temp["mode_index"],"area_index":"normal"})

    for k,v in results_medium_information.items():
        for temp in v:
            results_all_information[k].append(
                {"labelid": temp["info"]["labelid"],
                 "ttflabel": temp["info"]["ttflabel"],
                 "label": temp["info"]["label"],
                 "probably": temp["info"]["probably"],
                 'chinesestr': temp["info"]["char_result_oir"], 'left': temp["info"]["left"],
                 'top': temp["info"]["top"], 'right': temp["info"]["right"], 'bottom': temp["info"]["bottom"],"string_height":temp["info"]["string_height"],"sort_index":0.0,
                 "mode_index": temp["mode_index"],"area_index":"normal"})

    for k,v in results_low_information.items():
        for temp in v:
            results_all_information[k].append(
                {"labelid": temp["info"]["labelid"],
                 "ttflabel": temp["info"]["ttflabel"],
                 "label": temp["info"]["label"],
                 "probably": temp["info"]["probably"],
                 'chinesestr': temp["info"]["char_result_oir"], 'left': temp["info"]["left"],
                 'top': temp["info"]["top"], 'right': temp["info"]["right"], 'bottom': temp["info"]["bottom"],"string_height":temp["info"]["string_height"],"sort_index":0.0,
                 "mode_index": temp["mode_index"],"area_index":"normal"})

    results_all_information_final = defaultdict(list)

    for k,v in results_all_information.items():
        temp = recog_process.combine_recog(v)
        results_all_information_final[k] = temp


    for k,v in results_copyright_information.items():
        results_all_information_final_temp =  results_all_information_final[k].copy()
        for temp in v:
            for info in results_all_information_final_temp:
                if temp["chinesestr"] == info["chinesestr"] and temp["string_height"] == info["string_height"]:
                    if info in results_all_information_final[k]:
                        results_all_information_final[k].remove(info)
                        results_all_information_final[k].append(temp)

    return results_all_information_final
