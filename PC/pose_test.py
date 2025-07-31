import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime

# 센서 수
N = 4
max_len = 10000  # 각 센서별로 저장할 최대 데이터 수

# 시리얼 포트 연결
ser = serial.Serial('COM15', 115200, timeout=1)
ser.flush()

# 센서별 데이터 저장 리스트 초기화
data_history = [[] for _ in range(N)]

# matplotlib 초기 설정
fig, axes = plt.subplots(1, N, figsize=(5 * N, 4))
if N == 1:
    axes = [axes]

# 업데이트 함수
def update(frame):
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        try:
            values = list(map(float, line.split(',')))
            if len(values) != N:
                return

            for i in range(N):
                data_history[i].append(values[i])
                if len(data_history[i]) > max_len:
                    data_history[i].pop(0)

            for i in range(N):
                axes[i].cla()
                if data_history[i]:  # 데이터가 존재할 때만 처리
                    d = data_history[i]
                    bin_min = int(np.floor(min(d)))
                    bin_max = int(np.ceil(max(d)))
                    bins = np.arange(bin_min, bin_max + 1, 1)  # 간격 최소 1

                    axes[i].hist(d, bins=bins, color='skyblue', edgecolor='black')
                    axes[i].set_xlim([bin_min - 1, bin_max + 1])  # 여유 공간 추가
                axes[i].set_title(f'Sensor {i+1} Histogram')
                axes[i].set_xlabel('Distance')
                axes[i].set_ylabel('Frequency')


        except Exception as e:
            print(f"Error: {e}")

# 데이터 저장 함수
def save_data_and_fig():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_excel = f"sensor_data_{timestamp}.xlsx"
    filename_fig = f"histogram_plot_{timestamp}.png"

    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(dict(
        (f'Sensor_{i+1}', data_history[i]) for i in range(N)
    ))
    df.to_excel(filename_excel, index=False)
    print(f"✅ Data saved to {filename_excel}")

    # 현재 히스토그램 그래프 저장
    fig.savefig(filename_fig)
    print(f"✅ Plot saved to {filename_fig}")

# 애니메이션 시작 및 종료 처리
ani = FuncAnimation(fig, update, interval=1)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    save_data_and_fig()
    ser.close()
