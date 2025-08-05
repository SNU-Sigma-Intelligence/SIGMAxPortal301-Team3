import numpy as np
import pandas as pd

# 기본 설정
np.random.seed(42)

# 1. 평면 생성 조건
num_planes = 100  # 테스트용 평면 수
max_distance = 2.0  # 원점으로부터 최대 거리
max_angle_deg = 30  # Z축과의 최대 기울기 각도
max_angle_rad = np.deg2rad(max_angle_deg)

# 2. 평면 정의: n · (x - p0) = 0
# n: 단위 법선 벡터, p0: 평면 위 한 점 (거리 제한)

def random_plane(max_dist, max_angle):
    # 임의의 법선벡터 생성 (Z축과의 각도 제한)
    while True:
        n = np.random.randn(3)
        n = n / np.linalg.norm(n)
        angle = np.arccos(n[2])
        if angle <= max_angle:
            break
    # 거리 제한을 만족하는 평면 위의 점 선택
    d = np.random.uniform(0.1, max_dist)
    p0 = n * d
    return {'normal': n, 'point': p0}

# 여러 개의 평면 생성
planes = [random_plane(max_distance, max_angle_rad) for _ in range(num_planes)]

# DataFrame 형태로 정리
plane_data = pd.DataFrame({
    'normal_x': [p['normal'][0] for p in planes],
    'normal_y': [p['normal'][1] for p in planes],
    'normal_z': [p['normal'][2] for p in planes],
    'point_x': [p['point'][0] for p in planes],
    'point_y': [p['point'][1] for p in planes],
    'point_z': [p['point'][2] for p in planes],
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Generated Plane Dataset", dataframe=plane_data)
