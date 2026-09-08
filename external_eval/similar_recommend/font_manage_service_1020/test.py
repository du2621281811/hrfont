import pandas as pd
import matplotlib.pyplot as plt

# 读取文件
df = pd.read_excel("files/字体符号数量(1).xlsx")

# 提取符号数量列（假设列名为“符号数量”）
symbol_counts = df["符号数量"]

# 绘制频率直方图
plt.figure(figsize=(10, 6))
plt.hist(symbol_counts, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("符号数量")
plt.ylabel("频率")
plt.title("符号数量分布直方图")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
