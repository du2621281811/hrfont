# HR-Font mentor portal

只读入口。审查页**直接挂载**既有 `:8777` 目录，不再提供重写摘要：

| 路径 | 对应原页 |
|---|---|
| `/cn2west_v2_abc_review/` | `http://127.0.0.1:8777/cn2west_v2_abc_review/` |
| `/e1_formal_eval/` | `http://127.0.0.1:8777/e1_formal_eval/` |
| `/render_qa_hub.html` | `http://127.0.0.1:8777/render_qa_hub.html` |
| `/weekly.html` | 周报 HTML |
| `/assets/` | 周报插图 |

启动：

```bash
python scripts/serve_mentor_portal.py --host 0.0.0.0 --port 19001
```

本机优先打开 `:8777` 原地址；导师若只能访问 19000–19002，用本站同路径镜像。
