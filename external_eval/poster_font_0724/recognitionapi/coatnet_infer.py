
import torch
from torchvision import transforms
from recognitionapi.backbone.networks import coatnet_1, ArcFace
import settings
from collections import defaultdict
from PIL import Image
import codecs

with codecs.open(settings.font_recognition_model_label_path, mode="r", encoding="utf-8") as f:
    data = f.readlines()
f.close()
labellist_ch = [info.strip().split(",,,")[1] for info in data]

labellist = [info.strip().split(",,,")[2] for info in data]

font_id_list = [info.strip().split(",,,")[0] for info in data]

label_dict = defaultdict(list)

for i in range(len(labellist_ch)):
    label_dict[labellist_ch[i]].append({"ttfname":labellist[i],"font_id":font_id_list[i]})


with codecs.open(settings.cy_font_path,mode='r',encoding="utf-8")  as f:
    font107_list = f.readlines()
f.close()

font107_list = [info.strip() for info in font107_list]


with codecs.open(settings.cy_font_threshold_path,mode='r',encoding="utf-8")  as f:
    cy_font_threshold = f.readlines()
f.close()

cy_font_threshold = [info.strip().split("*")[1] for info in cy_font_threshold]

GPU_ID = [0]

train_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

# 组网，优化器构建
backbone = coatnet_1(input_shape=[224, 224], num_classes=5330)
head = ArcFace(in_features=768, out_features=5330, device_id=GPU_ID)
backbone_state_dict = torch.load(settings.font_recognition_model_backbone_path)["model_dict"]
head_state_dict = torch.load(settings.font_recognition_model_head_path)["model_dict"]
print('Loading backbone...')
backbone.load_state_dict(backbone_state_dict)  # strict为False，模型缺失的权重可不加载
print('Loading head...')
head.load_state_dict(head_state_dict)  # strict为False，模型缺失的权重可不加载

backbone.cuda()
head.cuda()

backbone.eval()
head.eval()

backbone.half()
head.half()




def cv2tensor(inputs):
    inputs_tendor = list()
    for inp in inputs:
        img = Image.fromarray(inp).convert('RGB')
        inputs_tendor.append(train_transform(img))
    return torch.stack(inputs_tendor)


def evaluate(img_list, top_k):
    input = cv2tensor(img_list)
    with torch.no_grad():
        input = input.to(settings.DEVICE)
        input = input.half()
        feature = backbone(input)
        output = head(feature, "")
        _, pred = output.data.topk(top_k, 1, True, True)

    font_dict1 = list()
    font_dict2 = list()


    for i in range(pred.size(0)):
        for j in range(pred.size(1)):
            font_name = labellist_ch[int(pred[i][j])]
            score = float(_[i][j])
            font_id = font_id_list[int(pred[i][j])]
            ttf_name = labellist[int(pred[i][j])]
            index = int(pred[i][j])

            font_dict2.append({"label": labellist_ch[int(pred[i][j])], "probably": float(_[i][j]),
                               "font_id": font_id_list[int(pred[i][j])], "ttfname": labellist[int(pred[i][j])],
                               "index": int(pred[i][j])})

            if font_name in font107_list:
                for i in range(len(font107_list)):
                    if font107_list[i] == font_name:
                        threshold = cy_font_threshold[i]
                if score > float(threshold):
                    font_dict1.append({"label": font_name, "probably": score,
                                         "font_id": font_id, "ttfname": ttf_name,
                                         "index": index})
            else:
                font_dict1.append({"label": labellist_ch[int(pred[i][j])], "probably": float(_[i][j]),
                                  "font_id": font_id_list[int(pred[i][j])], "ttfname": labellist[int(pred[i][j])],
                                  "index": int(pred[i][j])})


    return font_dict1,font_dict2