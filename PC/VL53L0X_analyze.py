import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, zscore, skew, kurtosis, anderson, probplot, kstest, jarque_bera
import pandas as pd
import seaborn as sns
import platform
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

file_name = "sensor_data(400)mm(V1).xlsx"
df = pd.read_excel(file_name)

col_name = df['Index']
data = df['Value']

base_name = os.path.splitext(file_name)[0]
log_file_name = base_name + "분석결과.txt"

sys.stdout = Logger(log_file_name)

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

z_scores = zscore(data)
mask = np.abs(z_scores) < 6
filtered_data = data[mask]

removed_count = len(data) - len(filtered_data)
removed_ratio = removed_count / len(data) * 100

def analyze_normality(arr, label):
    print(f"\n--- {label} ---")
    print(f"데이터 개수: {len(arr)}")
    
    mu, std = norm.fit(arr)
    print(f"평균(μ): {mu:.4f}, 표준편차(σ): {std:.4f}")
    
    skewness = skew(arr)
    kurt = kurtosis(arr, fisher=False)  # 일반 첨도 (정규=3)
    print(f"왜도(Skewness): {skewness:.4f}")
    print(f"첨도(Kurtosis): {kurt:.4f}")

    # Jarque-Bera 테스트
    jb_stat, jb_p = jarque_bera(arr)
    print(f"Jarque-Bera 통계량: {jb_stat:.4f}, p-value: {jb_p:.4g}")
    if jb_p > 0.05:
        print("  -> 귀무가설 채택 (정규분포 따름)")
    else:
        print("  -> 귀무가설 기각 (정규분포 따르지 않음)")

fig, axs = plt.subplots(1, 2, figsize=(12,6))

probplot(data, dist="norm", plot=axs[0])
axs[0].set_title("① 이상치 제거 전 Q-Q Plot")
axs[0].grid(True)

probplot(filtered_data, dist="norm", plot=axs[1])
axs[1].set_title("② 이상치 제거 후 Q-Q Plot")
axs[1].grid(True)

plt.tight_layout()
plt.show()

print(f"총 데이터 수: {len(data)}")
print(f"이상치 제거 수: {removed_count} ({removed_ratio:.2f}%)")

analyze_normality(data, "이상치 제거 전 데이터")
analyze_normality(filtered_data, "이상치 제거 후 데이터")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
