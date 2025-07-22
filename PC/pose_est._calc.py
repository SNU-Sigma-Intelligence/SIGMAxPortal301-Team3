import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# 센서 수 및 위치 (dy, dz)
dy = np.array([0, 4.5, 4.5, 4.5])
dz = np.array([0, 0, 3.5, -3.5])
N = 4

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
    global timer
    line = None
    while ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            values = list(map(float, line.split(',')))
            if len(values) != N:
                return
            print(values)
            d = np.array(values)
            normal, (a, b, c) = compute_normal(d, dy, dz)

            # 평면 계산
            y_range = np.linspace(-10, 20, 10)
            z_range = np.linspace(-10, 20, 10)
            Y, Z = np.meshgrid(y_range, z_range)
            X = a * Y + b * Z + c
            x0 = np.mean(X)
            y0 = np.mean(Y)
            z0 = np.mean(Z)

            # ------------------------------
            # 1. 정면 뷰 (x vs y)
            # ax_front.cla()
            # ax_front.set_title("Front View (X vs Y)")
            # ax_front.set_xlabel("X")
            # ax_front.set_ylabel("Y")
            # ax_front.grid(True)
            # ax_front.plot(a * y_range + b * 0 + c, y_range, color='gray')  # z=0 slice
            # for i in range(N):
            #     ax_front.scatter(d[i], dy[i], color='red')

            # # ------------------------------
            # # 2. 측면 뷰 (x vs z)
            # ax_side.cla()
            # ax_side.set_title("Side View (X vs Z)")
            # ax_side.set_xlabel("X")
            # ax_side.set_ylabel("Z")
            # ax_side.grid(True)
            # ax_side.plot(a * 0 + b * z_range + c, z_range, color='gray')  # y=0 slice
            # for i in range(N):
            #     ax_side.scatter(d[i], dz[i], color='red')

            # ------------------------------
            # 3. 3D 뷰
            ax_3d.cla()
            ax_3d.set_title("3D View")
            # ax_3d.set_xlim([np.min(X), np.max(X)])
            # ax_3d.set_ylim([np.min(Y), np.max(Y)])
            # ax_3d.set_zlim([np.min(Z), np.max(Z)])
            ax_3d.set_xlim([0, 200])
            ax_3d.set_ylim([-50, 50])
            ax_3d.set_zlim([-50, 50])
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.plot_surface(X, Y, Z, alpha=0.4, color='gray')
            ax_3d.quiver(x0, y0, z0, normal[0], normal[1], normal[2], length=20, color='blue')
            for i in range(N):
                ax_3d.scatter(d[i], dy[i], dz[i], color='red', s=30)

        except Exception as e:
            print(f"Error: {e}")


# matplotlib 초기 설정
fig = plt.figure(figsize=(12, 8))
ax_front = fig.add_subplot(221)
ax_side = fig.add_subplot(222)
ax_3d = fig.add_subplot(223, projection='3d')
ani = FuncAnimation(fig, update, interval=1)
plt.tight_layout()
plt.show()