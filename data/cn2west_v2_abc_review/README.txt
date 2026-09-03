CN2WEST v2 · A–H 协议 Review

入口: http://127.0.0.1:8777/cn2west_v2_abc_review/
Hub:   http://127.0.0.1:8777/render_qa_hub.html
A墨量: http://127.0.0.1:8777/cn2west_v2_abc_review/proto_A_ink/
A vs H: http://127.0.0.1:8777/cn2west_v2_abc_review/proto_AH_compare/

重建 Review: python scripts/build_cn2west_v2_protocol_review.py
重建 A 墨量: python scripts/build_cn2west_v2_proto_a_ink_preview.py
服务: python -m http.server 8777 --directory data/

图片直接引用各协议数据集目录（不复制 PNG）。
筛选/色散基于 B 协议自动筛查；永字高度按 A/B/C/D/F/H 分别显示。
