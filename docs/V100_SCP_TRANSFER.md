# V100 非 Git 数据：映射表 + scp

Git 只传代码。协议 A PNG、F0 ckpt、Es/Ec、官方 P1、support bank **不进仓**。  
所里 V100（`172.18.41.x`）目前 **ping/ssh 不通** 3090（`172.19.45.13:2222`），所以不在 V100 上直接拉。

流程：

1. **3090 源机**填 [`manifests/v100_scp_map.json`](../manifests/v100_scp_map.json)（主机、可达性、每项是否存在、实际体积、TTF 路径）。
2. 把填好的 JSON **commit + push**（只有路径和体积，没有权重）。
3. 在 **同时能碰到源机和 V100** 的跳板上，按表 `scp`/`rsync`。
4. V100 解包：`bash scripts/unpack_newhost_migrate.sh /root/projects/hrfont/artifacts/migrate_v100`。

填表命令（只在 3090 / 有数据的那台跑）：

```bash
cd /root/projects/hrfont
python scripts/fill_v100_scp_map.py
git add manifests/v100_scp_map.json
git commit -m "[3090] fill v100 scp path map"
git push origin HEAD:main   # 或开 PR；禁止 force-push
```

---

## 机器（V100 侧已填；源机/跳板请补）

| 角色 | 已知 | 请源机补 |
|------|------|----------|
| **源 / 3090** | `root@172.19.45.13:2222`，项目 `/root/projects/hrfont`，4×RTX 3090 | hostname、`git rev-parse HEAD`、本机能否 ssh 到 V100 宿主机 |
| **目标 / V100** | 用户记的宿主机 `172.18.41.23`；容器 hostname `0ecccf9be2dd`，容器 IP `172.17.0.2`；项目 **必须** `/root/projects/hrfont`；conda `boogu` | 从 3090/跳板 ssh 的 **真实端口、用户、是否要进容器** |
| **跳板** | 隔壁 `172.18.41.24`（SSH config `24-8V100`，端口 17569）也 ping 不通 `172.19.45.13` | 一台能两边通的机器：IP、端口、用户 |

V100 磁盘（2026-09-11）：overlay 800G，剩约 **181G**。全套约 **103G**（含 Ec 94G）能放下；**不要**拷整份 `runs/`。

---

## 必须拷的内容（优先这条，不要用 TTF 重渲替代）

路径都相对 `/root/projects/hrfont`。目标机同样落这些路径。

| id | 源路径 | 约体积 | 必须 | 建议传输 | 解包后落点 |
|----|--------|--------|------|----------|------------|
| `01_small` | `artifacts/migrate_v100/01_small.tar` | ~0.7G | 是 | scp tar | unpack → `artifacts/f0/support_bank*.json` + `code/official/FontDiffuser/ckpt/*.pth` |
| `03_es` | `artifacts/migrate_v100/03_es_cache.tar` | ~1.7G | 是 | scp tar | unpack → `artifacts/f0/es_spatial_f0/`（**F0 Es，禁止用 E1**） |
| `04_f0` | `artifacts/migrate_v100/04_f0_best.tar` | ~1.1G | 是 | scp tar | unpack → `runs/F0-RSIFREE-FT-A-S3407/best/` |
| `05_eval` | `artifacts/migrate_v100/05_eval_ckpts.tar` | ~4G | 评旧臂才要 | scp tar | unpack → F1/F2/F3 `global_step_*` |
| `proto_A` | `data/fontdiffuser-p253-t295-s338-cn2west-v2/` | ~0.7G / 16 万 PNG | 是 | **rsync/scp -r**，不打 tar | 同相对路径 |
| `ec_f0` | `artifacts/f0/ec_multiscale_f0/` | **~94G** | 训/评 F 臂必须 | **rsync**，不打 tar | 同相对路径 |

源机若 tar 尚未打好：

```bash
cd /root/projects/hrfont
bash scripts/pack_newhost_migrate.sh
# 产出 artifacts/migrate_v100/{01_small,03_es_cache,04_f0_best,05_eval_ckpts}.tar
```

**不要拷：** 整份 `runs/`（单臂 18–23G）、`data/fontdiffuser`、`data/font`、E1 的 Es/Ec、E12 cache（除非做 E12）、训练 log。

---

## 跳板上 scp / rsync（填完 JSON 后改主机）

下面 `SRC` / `DST` / 端口换成 `v100_scp_map.json` 里的值。先拷 tar 和小目录，Ec 最后拷。

```bash
SRC=root@172.19.45.13
SRC_PORT=2222
DST=root@172.18.41.23
DST_PORT=22
ROOT=/root/projects/hrfont

# 1) tar（若跳板磁盘紧，可以不落地，ssh 管道直送）
ssh -p "$SRC_PORT" "$SRC" "ls -lh $ROOT/artifacts/migrate_v100/*.tar"
scp -P "$SRC_PORT" "$SRC:$ROOT/artifacts/migrate_v100/"*.tar \
    "$ROOT/artifacts/migrate_v100/"          # 跳板中转时改成本地目录
scp -P "$DST_PORT" "$ROOT/artifacts/migrate_v100/"*.tar \
    "$DST:$ROOT/artifacts/migrate_v100/"

# 管道直送（跳板不落盘）
ssh -p "$SRC_PORT" "$SRC" "tar -C $ROOT/artifacts/migrate_v100 -cf - ." \
  | ssh -p "$DST_PORT" "$DST" "mkdir -p $ROOT/artifacts/migrate_v100 && tar -C $ROOT/artifacts/migrate_v100 -xf -"

# 2) 协议 A PNG
ssh -p "$SRC_PORT" "$SRC" "tar -C $ROOT/data -cf - fontdiffuser-p253-t295-s338-cn2west-v2" \
  | ssh -p "$DST_PORT" "$DST" "mkdir -p $ROOT/data && tar -C $ROOT/data -xf -"

# 3) Ec 94G（优先 rsync；断点可续）
rsync -aH --info=progress2 -e "ssh -p $SRC_PORT" \
  "$SRC:$ROOT/artifacts/f0/ec_multiscale_f0/" \
  /tmp/ec_multiscale_f0/
rsync -aH --info=progress2 -e "ssh -p $DST_PORT" \
  /tmp/ec_multiscale_f0/ \
  "$DST:$ROOT/artifacts/f0/ec_multiscale_f0/"
```

V100 上：

```bash
bash /root/projects/hrfont/scripts/unpack_newhost_migrate.sh \
  /root/projects/hrfont/artifacts/migrate_v100
# 确认：
#   data/fontdiffuser-p253-t295-s338-cn2west-v2/train/TargetImage
#   artifacts/f0/es_spatial_f0/manifest.json
#   artifacts/f0/ec_multiscale_f0/manifest.json
#   artifacts/f0/support_bank.json
#   runs/F0-RSIFREE-FT-A-S3407/best/unet.pth
#   code/official/FontDiffuser/ckpt/unet.pth
```

---

## 可选：TTF 映射（只为协议 A 重渲 PNG）

**不能**从 TTF 得到 ckpt / Es / Ec / support_bank。与 3090 **不能保证逐像素一致**（Pillow / FreeType / 字体文件版本）。V100 现缺：

- `/root/data/font_50`
- `data/suiti_fonts_probe/随体字体集`
- `data/fontdiffuser-p253-t295-s338-cn2west-v2b-hfit/summary.json`（脚本靠它读 stem→ttf）

JSON 里 `ttf_render_optional.stems` 已列出 228+16+16 个 stem。源机跑 `fill_v100_scp_map.py` 会写入每条 `src_ttf`。  
若仍要走重渲：把对应 TTF **连同** `summary.json` 拷到 V100（字体文件仍不进 Git），再：

```bash
python scripts/build_cn2west_v2_proto_abc.py --proto A --workers 16
# 仓内估计 15–30 分钟（字体齐的前提下）
```

Content 字体 V100 已有：`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。

---

## 不要做的事

- 不要 force-push `main`。
- 不要 `git add` 权重、PNG、cache、整棵 `runs/`。
- 不要把 E1 cache 接到 F 臂。
- 不要在 V100 上 bf16；capability 必须是 `(7, 0)`，fp16。
- 不要开训，直到 unpack 清单齐、且 `python scripts/pm_preflight.py` 通过。
