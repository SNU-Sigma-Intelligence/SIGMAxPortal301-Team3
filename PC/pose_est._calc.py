import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 센서 수 및 위치 (dy, dz)
dy = np.array([0, 10, 0, 10])
dz = np.array([0, 10, 10, 10])
N = 3

# 시리얼 포트 연결
ser = serial.Serial('COM15', 115200, timeout=1)  # ← 포트 이름 확인 필요
ser.flush()

# 평면 회귀 → 법선 벡터 계산 + 평면 계수 반환
def compute_normal(d, dy, dz):
    dy = dy[:N]
    dz = dz[:N]
    A = np.vstack([dy, dz, np.ones(N)]).T
    x = d
    coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    a, b, c = coeffs
    n = np.array([1, -a, -b])
    return n / np.linalg.norm(n), (a, b, c)

def update(frame):
    global arrow
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        print(line)
        try:
            values = list(map(float, line.split(',')))
            if len(values) != N:
                return 
            d = np.array(values)
            normal, (a, b, c) = compute_normal(d, dy, dz)

            # 평면 그리드 생성
            y_range = np.linspace(-10, 20, 10)
            z_range = np.linspace(-10, 20, 10)
            Y, Z = np.meshgrid(y_range, z_range)
            X = a * Y + b * Z + c

            # 중심점 계산
            mid_idx_y = Y.shape[0] // 2
            mid_idx_z = Y.shape[1] // 2
            x0 = X[mid_idx_y, mid_idx_z]
            y0 = Y[mid_idx_y, mid_idx_z]
            z0 = Z[mid_idx_y, mid_idx_z]

            # 그래프 초기화
            ax.cla()
            ax.set_xlim([0, 200])
            ax.set_ylim([-50, 50])
            ax.set_zlim([-50, 50])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

           # 평면 그리기
            ax.plot_surface(X, Y, Z, alpha=0.4, color='gray', rstride=1, cstride=1, edgecolor='none')

            # ▶️ 법선 벡터: 평면 중앙에서부터 그리기
            ax.quiver(x0, y0, z0, normal[0], normal[1], normal[2], length=20, color='blue')

            # 센서가 측정한 점 시각화
            for i in range(N):
                ax.scatter(d[i], dy[i], dz[i], color='red', s=30)

        except Exception as e:
            print(f"Error: {e}")

# matplotlib 초기 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
arrow = None

ani = FuncAnimation(fig, update, interval=1)
plt.show()
