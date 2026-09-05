# HR-Font mentor portal

只读静态评审入口，公开内容被服务端路由限制为：

- `/`：整理后的项目入口；
- `/rendering.html`：六种渲染协议与边界框解释；
- `/bbox/`：原始边界框技术页；
- `/e1/`：E1 正式交互评测；
- `/assets/`：周报精选图片。

启动：

```bash
python scripts/serve_mentor_portal.py --host 0.0.0.0 --port 19001
```

本机检查：<http://127.0.0.1:19001/>

导师内网入口（与此前 overnight 网页同一主机）：

- 周报：<http://172.19.45.13:19001/weekly.html>
- 总入口：<http://172.19.45.13:19001/>

只暴露 `19000–19002`；不要把服务根目录改成项目根目录或 `data/`。
