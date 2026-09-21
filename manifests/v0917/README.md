# v0917 Train / Val / Test 划分归档

- dataset_id: `v0917_split`
- seed: `3407`
- source: `/root/data1/hrfont_dataset_v2_20260917/manifests/v0917`
- `required_pair_filter`: `true`
- donor filter: `donor_train_by_cp.json`

## 划分概览

| split | fonts | pairs | latin | kana | bopomofo |
|---|---:|---:|---:|---:|---:|
| train | 200 | 39547 | 17012 | 20870 | 1665 |
| val | 16 | 3208 | 1370 | 1690 | 148 |
| test | 16 | 3208 | 1370 | 1690 | 148 |

## 字体划分

- `train`: 200 fonts
  - `4314d6a5`, `5f132397`, `FZAiSJW`, `FZAiSJW_Cu`, `FZAiSJW_Xian`, `FZBGDT`, `FZBITMAPFENGJW`, `FZBaDSRXKJW`, `FZBaiZDZ113JW`, `FZBaiZJDTJW`, `FZBaiZJHTJW-L`, `FZBaoCTJW`, `FZBingMYTJW`, `FZBoRDKJW`, `FZBoYTJW-B`, `FZBoYTJW-L`, `FZChaoLTJW`, `FZDaLTJW`, `FZDengHLJW`, `FZDiHTJ_Te`, `FZDiHTJ_Xian`, `FZDiHTJ_Zhun`, `FZDunHJSJW`, `FZFeiYTJW`, `FZFenSTXJW`, `FZFengBPYTJW`, `FZFengRSTJW-B`, `FZFengRSTJW-R`, `FZFengYSJW-EL`, `FZFengZTJW-B`, `FZFengZTJW-EB`, `FZFengZTJW-M`, `FZFuGCSJW`, `FZGaoHeiJ`, `FZHanSTJW`, `FZHuaSKXTJW-T`, `FZHuoLTJW`, `FZJiHXTJW`, `FZJianQTJW`, `FZJinCTJW`, `FZJingWTJW`, `FZJingWTJW_Cu`, `FZJingWTJW_Xian`, `FZKuaiSTJW`, `FZLaGBTTJW`, `FZLangKCTJW2`, `FZLangYJW-L`, `FZLangYJW-R`, `FZLiangHJW-EB`, `FZLiangHJW-L`, `FZLingDTJW-B`, `FZLingDTJW-L`, `FZLingFJXKJW`, `FZLiuCTJW`, `FZLouLTJW-EB`, `FZLouLTJW-R`, `FZLuoDTJW`, `FZLuoMTJW-L`, `FZLuoMTJW-R`, `FZLuoMXTJW-EB`, `FZLuoMXTJW-M`, `FZMaKTJW`, `FZMaLTJW`, `FZManYTJW-T`, `FZMengHHXJW-EB`, `FZMengHHXJW-EL`, `FZMengYTJW`, `FZMingZTJW-EB`, `FZOuYHGXSJW`, `FZPANGPBJW`, `FZPangPHJW`, `FZPiaoTJW`, `FZPiaoTJW_Cu`, `FZPiaoTJW_Xi`, `FZPiaoYSJW`, `FZQINGSTJW`, `FZQiMTJW`, `FZQiMTJW_Xian`, `FZQiMTJW_Zhun`, `FZQianWTJW`, `FZQianWTJW_Te`, `FZQianWTJW_Xian`, `FZQiangKTJW`, `FZSJ-CHAOJXHN`, `FZSJ-KEADKA`, `FZSJ-NUANNXWC`, `FZSJ-QIYXRYY`, `FZSJ-WOSJJ`, `FZSJ-WULXEM`, `FZSJ-XIAOHBZLL`, `FZSJ-XIDPB`, `FZShaoLGFTJW`, `FZShengSKSJW_Zhun`, `FZShiTSJW`, `FZShuYTJW-L`, `FZSiBTJW`, `FZSiLTJW_Cu`, `FZSiLTJW_Xian`, `FZSuXSGYSJW`, `FZSuXSMCTJW`, `FZSuXSMZTJW`, `FZTanHTJW`, `FZTieXHJW_Xi`, `FZTieXHJW_Zhun`, `FZVDLJingQSJ-H`, `FZVDLJingQSJ-L`, `FZVDLJingQSJ-T`, `FZWanBTJW`, `FZWanBTJW_Xian`, `FZXLB`, `FZXianWTJW_Zhun`, `FZXianYSTJW`, `FZXiangPZTJW`, `FZXiangPZTJW_Da`, `FZXiangPZTJW_Xian`, `FZXinGHJW-UL`, `FZYNJW`, `FZYaZTJ_Te`, `FZYaZTJ_Xian`, `FZYinSJW-DB`, `FZYinSJW-R`, `FZYingCTJW`, `FZYingCTJW_Xian`, `FZZH-AXTJW`, `FZZH-DDYLTJW`, `FZZH-DFHBJW`, `FZZH-KTMHTJW`, `FZZH-KTPPTJW`, `FZZH-LTFKTJW`, `FZZH-LTLMMTJW`, `FZZH-LTXXTJW`, `FZZH-MXQYTJW`, `FZZH-NYXFTJ`, `FZZH-QHXKJW`, `FZZH-SJDFTJW`, `FZZH-TTTJW`, `FZZH-YMTJW`, `FZZH-YueXSTJW-EB`, `FZZH-ZRKTJW`, `FZZJ-BCSXJW`, `FZZJ-BXMRTJW`, `FZZJ-CZYKSJW`, `FZZJ-DDFDTJW`, `FZZJ-FOJW`, `FZZJ-FXGGLJW`, `FZZJ-GYJGTJW`, `FZZJ-HLYHXLJW`, `FZZJ-JKTJW`, `FZZJ-LSHSSJW`, `FZZJ-LXBYTJW`, `FZZJ-LXCTJW`, `FZZJ-LXXSJW`, `FZZJ-LongYTJW`, `FZZJ-MGSCTJW`, `FZZJ-MMXJJW`, `FZZJ-MSTJW`, `FZZJ-PYQYTJW`, `FZZJ-SDXKJW`, `FZZJ-SLJMTJW`, `FZZJ-TBPYTJW`, `FZZJ-WFSXTJ`, `FZZJ-XHFTJW`, `FZZJ-YYXSTJW`, `FZZJ-ZCJW`, `FZZJ-ZDFCSJW`, `FZZJ-ZDXKJW`, `FZZJ-ZQXKJW`, `FZZhuMTJW`, `FZZhuoYTJW`, `FZZuanSTJW`, `FZZuoZJHJW-T`, `FZZuoZYXTJW-B`, `FZZuoZYXTJW-R`, `STFZRTXSJ`, `ZJZhuangSHJ-T`, `ZKTBanQTFU`, `ZKTMingXTFU`, `fc4a8715`, `fzsj_1275625`, `fzsj_1275627`, `fzsj_1275635`, `fzsj_1522777`, `fzsj_1928053`, `fzsj_1966669`, `fzsj_1966674`, `fzsj_1966717`, `fzsj_1966721`, `fzsj_1966722`, `fzsj_1966797`, `fzsj_1967025`, `fzsj_2017088`, `fzsj_2127126`, `fzsj_2135224`, `fzsj_2173041`, `fzsj_2187650`, `fzsj_2187909`, `fzsj_2187928`, `fzsj_2261418`, `fzsj_2923481`, `fzsj_2923966`

- `val`: 16 fonts
  - `FZHuangTJXSJW`, `FZKuaiHTJW`, `FZSUXSFBCJW`, `FZShouHCGTJW`, `FZShouHJW`, `FZShouHJW_Xi`, `FZVDLQianZHJW-B`, `FZVDLQianZHJW-L`, `FZXianZTJW`, `FZZJ-XTJW`, `FZZanMTJW`, `FZZhiQCTJW-B`, `FZZhiQCTJW-L`, `FZZhongYSJW`, `FZZiZSFSJW`, `fzsj_1966700`

- `test`: 16 fonts
  - `FZBaiZHXTJW`, `FZCYJW`, `FZDouNTJW`, `FZFeiHHJW-T`, `FZFuXTJW`, `FZHanBTJW`, `FZHanBTJW_Da`, `FZHanBTJW_Xi`, `FZHuoYYJW-L`, `FZJianLTJW`, `FZXSHJW`, `FZZJ-HLYHXWJW`, `FZZJ-KFTXKJW`, `FZZJ-MJZCFU`, `ee347e17`, `fzsj_1275648`

## 文件校验

| file | sha256 |
|---|---|
| `donor_train.json` | `3a37d07882e4c180fd3be08d9fcdfa0a2eb12039ab51472f4cde32fda7e61f27` |
| `donor_train_by_cp.json` | `b2e904ea2d84f1f5809c6eebaad50d316db48add322e8622f925243650021c5a` |
| `fonts.tsv` | `62c904f5021014763e04c3da4e83415973cb1e70c12c98e2efb5b63a477b4322` |
| `pairs_test.tsv` | `6797749c1ca97e7831db11c55fdf0745e6aef491fc9d6ac747179026e298641f` |
| `pairs_train.tsv` | `4e5432d9721b63af3f3b9c2750db9ec202d6aa8992d2fbe9579d7999e3341ce2` |
| `pairs_val.tsv` | `fc11f5ae106a6bca1e3b103fcbfd6b39133087bfae3c9c43008645bb138d8d38` |
| `reference_compatibility.json` | `8e6ebaab25207f32020ca35a7b5625bc99e2aee6497a110a7adeca3e79af9f4f` |
| `sample_weights.json` | `f8164fc816c76a1e6894c88e7bab262ba14b02e19170a2ef8f4c616fc16df4b3` |
| `split.json` | `1d897c0a6f061059be013734ac8ab34bae67d96746af29fe9160b21f0a1097c2` |
| `style_pool.json` | `9a1c2607c3754a6aeb9d71b597405bfdf5d18f7d5a8134aba7c03669a0ddce1a` |

该目录保存完整 v0917 manifest；图片数据不复制进 Git，使用 manifest 中的 split/font/cp 记录定位原始数据。
