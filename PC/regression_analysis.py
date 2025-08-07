import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import platform

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 정의
true_distances = np.array([400, 500, 600, 700, 800, 900, 1000])
means = np.array([421.7013, 529.8565, 640.7314, 745.0980, 849.7795, 951.6294, 1043.3470])
stds = np.array([3.1969, 4.2684, 5.1780, 6.4103, 7.6950, 8.8500, 10.4828])
errors = means - true_distances  # 오차 계산

# 2D로 reshape (scikit-learn 요구사항)
X = true_distances.reshape(-1, 1)

# ------------------------------------
# 📌 오차 회귀 분석
# ------------------------------------
error_model = LinearRegression().fit(X, errors)
error_pred = error_model.predict(X)
r2_error = r2_score(errors, error_pred)
a1 = error_model.coef_[0]
b1 = error_model.intercept_

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(true_distances, errors, color='blue', label='오차 데이터')
plt.plot(true_distances, error_pred, color='red', label=f'y = {a1:.4f}x + {b1:.2f}\n$R^2$ = {r2_error:.4f}')
plt.xlabel('실제 거리 (mm)')
plt.ylabel('오차 (mm)')
plt.title('오차 vs 실제 거리 (회귀 분석)')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------
# 📌 표준편차 회귀 분석
# ------------------------------------
std_model = LinearRegression().fit(X, stds)
std_pred = std_model.predict(X)
r2_std = r2_score(stds, std_pred)
a2 = std_model.coef_[0]
b2 = std_model.intercept_

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(true_distances, stds, color='green', label='표준편차 데이터')
plt.plot(true_distances, std_pred, color='orange', label=f'y = {a2:.4f}x + {b2:.2f}\n$R^2$ = {r2_std:.4f}')
plt.xlabel('실제 거리 (mm)')
plt.ylabel('표준편차 (mm)')
plt.title('표준편차 vs 실제 거리 (회귀 분석)')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------
# 📌 수식 및 R² 결과 출력
# ------------------------------------
print("오차 회귀식:      오차 = {:.4f} * 거리 + {:.2f}".format(a1, b1))
print("오차 R² 값:        {:.4f}".format(r2_error))

print("표준편차 회귀식:  표준편차 = {:.4f} * 거리 + {:.2f}".format(a2, b2))
print("표준편차 R² 값:    {:.4f}".format(r2_std))
