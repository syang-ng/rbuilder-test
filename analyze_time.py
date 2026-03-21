import os
import re
import pandas as pd

path25n7 = "/home/senyang/rbuilder-test/outputs-v6"
# path25 = "/home/senyang/rbuilder-test/outputs-n8"
# path25 = "/home/senyang/rbuilder-test/outputs-v6"
path50n8 = "/home/senyang/rbuilder-test/outputs-t50v2"
# path80 = "/home/senyang/rbuilder-test/outputs-t80v2"
path50n7 = "/home/senyang/rbuilder-test/outputs-t50n7"
path80n7 = "/home/senyang/rbuilder-test/outputs-t80n7"

pattern = re.compile(r'with builder: "([^"]+)".*?took\s+([0-9]*\.?[0-9]+)\s*(ms|s)\b')

def count_data(path):
    files = os.listdir(path)
    data = []

    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    builder = match.group(1)
                    value = float(match.group(2))
                    unit = match.group(3).lower()
                    ms = value if unit == "ms" else value * 1000
                    data.append((file, lineno, builder, ms))
    return data


# data25 = count_data(path25)
# data50 = count_data(path50)
# data80 = count_data(path80)
data25n7 = count_data(path25n7)
data50n8 = count_data(path50n8)
data50n7 = count_data(path50n7)
data80n7 = count_data(path80n7)

df25 = pd.DataFrame(data25n7, columns=["file", "lineno", "builder", "time_ms"])
df25["builders"] = df25["builder"].apply(lambda x: f"{x} (rbuilder)" if x != "default" else "default (threads=25, $k_{cutoff}$=8)")
# df25["threads"] = "25"
df50n8 = pd.DataFrame(data50n8, columns=["file", "lineno", "builder", "time_ms"])
df50n8 = df50n8[df50n8["builder"] == "default"].reindex()
df50n8["builders"] = "default (threads=50, $k_{cutoff}$=9)"
df50 = pd.DataFrame(data50n7, columns=["file", "lineno", "builder", "time_ms"])
df50 = df50[df50["builder"] == "default"].reindex()
df50["builders"] = "default (threads=50, $k_{cutoff}$=8)"
# df50["threads"] = "50"
df80 = pd.DataFrame(data80n7, columns=["file", "lineno", "builder", "time_ms"])
df80 = df80[df80["builder"] == "default"].reindex()
df80["builders"] = "default (threads=80)"
# df80["threads"] = "80"
# df25n7 = pd.DataFrame(data25n7, columns=["file", "lineno", "builder", "time_ms"])
# df25n7["threads"] = "25 (n=7)"

df = pd.concat([df25, df50, df50n8], ignore_index=True)

# time_df = df[df["builder"] == ]

import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
plt.figure(figsize=(16, 10))
# line_styles = ["-", "--", "-.", ":", "--"]
ax = sns.ecdfplot(data=df, x="time_ms", stat="proportion", linewidth=5, hue="builders")
sns.move_legend(ax, "lower right", title="building algorithms", fontsize=24, title_fontsize=24)
plt.ylabel('CDF', fontsize=28)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(0, 6000)
plt.xlabel("Time (ms)", fontsize=28)
plt.savefig("time_ecdf_v3.pdf", bbox_inches='tight')
# plt.savefig("time_ecdf_final.png", bbox_inches='tight')

# plt.clf()
# plt.figure(figsize=(12, 6))
# sns.ecdfplot(data=df25n7,  x="time_ms", stat="proportion", linewidth=3, hue="builder")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlim(0, 30000)
# plt.xlabel("Time (ms)", fontsize=24)
# plt.ylabel("CDF", fontsize=24)
# plt.savefig("time_ecdf_new5.png", bbox_inches='tight')




# plt.clf()
# plt.figure(figsize=(12, 6))
# sns.ecdfplot(data=df25,  x="time_ms", stat="proportion", linewidth=3, hue="builder")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlim(0, 30000)
# plt.xlabel("Time (ms)", fontsize=24)
# plt.ylabel("CDF", fontsize=24)
# plt.savefig("time_ecdf_new6.png", bbox_inches='tight')