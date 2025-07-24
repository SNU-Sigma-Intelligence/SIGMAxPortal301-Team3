import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

N = 4

# 시리얼 포트 연결
ser = serial.Serial('COM15', 115200, timeout=1)  # ← 포트 이름 확인 필요
ser.flush()

class PointManager:
    points = {}
    def __init__(self, n, dx, dy, dz, yaw, pitch):
        # for i in range(n):
        #     self.points[i] = Point(i, self.dy[i], self.dz[i])
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.ptch = pitch
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
    
    def draw_surface(self, frame):
        update_ret = self.update()
        if not update_ret:
            return
        if update_ret == 0:
            return
        normal, X, Y, Z, x0, y0, z0, loc_d, loc_dx, loc_dy, loc_dz = update_ret
        self.ax_3d.cla()
        self.ax_3d.set_title("3D View")
        self.ax_3d.set_xlim([0, 200])
        self.ax_3d.set_ylim([-50, 50])
        self.ax_3d.set_zlim([-50, 50])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.plot_surface(X, Y, Z, alpha=0.4, color='gray')
        self.ax_3d.quiver(x0, y0, z0, normal[0], normal[1], normal[2], length=20, color='blue')

        for i in range(len(loc_d)):
            self.ax_3d.scatter(loc_d[i][0]+loc_dx[i], loc_d[i][1]+loc_dy[i], loc_d[i][2]+loc_dz[i], color='red', s=30)
    
    def update(self):
        line = None
        while ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                #센서 측정 값 받아오기
                values = list(map(float, line.split(',')))
                if len(values) != N:
                    return
                d = np.array(values)

                #find indexes of abnormal data 
                inds = []
                for i in range(len(d)):
                    if d[i] > 8000:
                        inds.append(i)
                inds.reverse()

                #법선 벡터 계산
                normal, (a, b, c), (new_d, new_dx, new_dy, new_dz) = self.compute_normal(d, inds)
                # 평면 계산
                y_range = np.linspace(-10, 20, 10)
                z_range = np.linspace(-10, 20, 10)
                Y, Z = np.meshgrid(y_range, z_range)
                X = a * Y + b * Z + c
                x0 = np.mean(X)
                y0 = np.mean(Y)
                z0 = np.mean(Z)
                return normal, X, Y, Z, x0, y0, z0, new_d, new_dx, new_dy, new_dz
            except Exception as e:
                print(f"Error: {e}")
                return 0
   
    def compute_normal(self, d, inds):
        loc_d = d
        loc_dx = self.dx
        loc_dy = self.dy
        loc_dz = self.dz
        for i in inds:
            loc_d = np.delete(loc_d, i)
            loc_dx = np.delete(loc_dx, i)
            loc_dy = np.delete(loc_dy, i)
            loc_dz = np.delete(loc_dz, i)
        A = np.vstack([loc_dy, loc_dz, np.ones(len(loc_d))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, loc_d, rcond=None)
        a, b, c = coeffs
        n = np.array([1, -a, -b])
        return n / np.linalg.norm(n), (a, b, c), (loc_d, loc_dx, loc_dy, loc_dz)

class Point:
    def __init__(self, index, dy, dz):
        self.index = index
        self.x = 0
        self.y = dy
        self.z = dz

#PointManager 설정
amt = 4
dx = np.array([0, 0, 0, 0])
dy = np.array([0, 14.1, 14.1, 0])
dz = np.array([0, 0, 12.2, 12.2])
yaw = np.array([0, 0, 0, 0])
pitch = np.array([0, 0, 0, 0])
point_manager = PointManager(amt, dx, dy, dz, yaw, pitch)

#그림그리기
ani = FuncAnimation(point_manager.fig, point_manager.draw_surface, interval=1)
plt.tight_layout()
plt.show()