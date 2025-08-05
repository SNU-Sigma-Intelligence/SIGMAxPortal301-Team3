import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# torch.autograd.set_detect_anomaly(True)

# 거리 기반 센서 오차 모델
def sensor_noise_std(dist, a=0.01, b=0.002):
    return a * dist + b

# 센서 클래스 정의
class Sensor(nn.Module):
    def __init__(self, num_sensors, radius = 0.3):
        super().__init__()
        self.num = num_sensors
        # z 좌표를 제외한 xy 좌표만 학습 파라미터로 설정
        self.pos_xy = []
        for i in range(num_sensors):
            self.pos_xy.append([radius * math.cos(2 * math.pi * i / num_sensors),
                                radius * math.sin(2 * math.pi * i / num_sensors)])
        self.pos_xy = torch.tensor(self.pos_xy, dtype=torch.float32)
        self.pos_xy = nn.Parameter(self.pos_xy)
        self.dir_angles = nn.Parameter(torch.zeros(num_sensors, 2))  # theta, phi

    def get_rays(self):
        # 학습된 pos_xy에 z=0 좌표를 추가하여 3D 위치를 동적으로 생성
        z_coord = torch.zeros(self.num, 1, device=self.pos_xy.device, dtype=self.pos_xy.dtype)
        pos_3d = torch.cat([self.pos_xy, z_coord], dim=1)

        theta = self.dir_angles[:, 0]  # 위도
        phi = self.dir_angles[:, 1]    # 방위각
        dx = torch.sin(theta) * torch.cos(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(theta)
        dirs = torch.stack([dx, dy, dz], dim=1)
        dirs = F.normalize(dirs, dim=1)
        
        # 생성된 3D 위치와 방향을 반환
        return pos_3d, dirs

# ray-plane 교점 계산
def ray_plane_intersect(ray_o, ray_d, plane_n, plane_p):
    denom = ray_d @ plane_n
    t = ((plane_p - ray_o) @ plane_n) / denom
    point = ray_o + t.unsqueeze(-1) * ray_d
    return point, t

# 법선벡터 추정 (공분산 행렬 및 고유값 분해 사용)
def estimate_normal(points):
    centroid = points.mean(dim=0)
    centered = points - centroid
    
    # NaN/Inf 제거
    mask = ~(torch.isnan(centered).any(dim=1) | torch.isinf(centered).any(dim=1))
    centered = centered[mask]
    
    if centered.shape[0] < 3:
        raise ValueError("Too few valid points to estimate normal.")

    # 공분산 행렬 계산
    covariance_matrix = centered.T @ centered

    # 역전파 시 수치적 안정성을 위해 작은 jitter 값을 더함
    # 고유값이 0이 되는 것을 방지하여 NaN 그래디언트 발생을 막음
    jitter = 1e-6 * torch.eye(covariance_matrix.shape[0], device=covariance_matrix.device)
    
    # torch.linalg.eigh는 대칭 행렬에 대한 고유값과 고유벡터를 계산
    # 고유값은 오름차순으로 정렬되어 반환됨
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix + jitter)
    
    # 법선 벡터는 가장 작은 고유값에 해당하는 고유벡터
    normal = eigenvectors[:, 0]
    
    # 법선 벡터의 방향을 일관성 있게 Z축의 양수 방향으로 조정
    if normal[-1] < 0:
        normal = -normal
        
    return normal

# 각도 오차 계산
def angle_error(n_est, n_true):
    cos_sim = F.cosine_similarity(n_est, n_true, dim=0).clamp(-0.9999999, 0.9999999)
    return torch.acos(cos_sim) * 180 / torch.pi

# 평가 함수 (평균 각도 오차)
def evaluate_model(sensor: Sensor, plane_dataset, noise_sample_per_plane=5):
    torch.set_grad_enabled(True)
    ray_o, ray_d = sensor.get_rays()
    total_error = torch.tensor(0.0, device=ray_o.device)

    theta_min, theta_max = torch.tensor(torch.deg2rad(torch.tensor(-90.0))), torch.tensor(torch.deg2rad(torch.tensor(90.0)))
    angle_penalty_weight = 100
    angle_penalty = F.relu(sensor.dir_angles[:, 0] - theta_max).sum() + \
                     F.relu(theta_min - sensor.dir_angles[:, 0]).sum()

    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    position_penalty_weight = 100
    position_penalty = F.relu(sensor.pos_xy[:, 0] - x_max).sum() + \
                        F.relu(x_min - sensor.pos_xy[:, 0]).sum() + \
                        F.relu(sensor.pos_xy[:, 1] - y_max).sum() + \
                        F.relu(y_min - sensor.pos_xy[:, 1]).sum()
    
    range_penalty_weight = 10
    max_range = 3
    min_range = 0.1
    range_penalty = torch.tensor(0.0, device=ray_o.device)

    valid_count = 0
    for plane in plane_dataset:
        p_n = torch.tensor(plane['normal'], dtype=torch.float32, device=ray_o.device)
        p_p = torch.tensor(plane['point'], dtype=torch.float32, device=ray_o.device)

        inter_pts, dists = ray_plane_intersect(ray_o, ray_d, p_n, p_p)
        range_penalty += F.relu(dists - max_range).sum() + \
                            F.relu(min_range - dists).sum()
            
        # 유효하지 않은 거리나 NaN 탐지
        invalid_mask = (dists < 0) | torch.isnan(dists) | torch.isinf(dists)
        if invalid_mask.any():
            continue
        
        error_sum = torch.tensor(0.0, device=ray_o.device)
        for _ in range(noise_sample_per_plane):
            noise_std = sensor_noise_std(dists)
            # noise 항은 gradient가 끊기지 않도록 detaching
            noise = (torch.randn_like(inter_pts) * noise_std.unsqueeze(-1)).detach()
            noisy_pts = inter_pts + noise

            n_est = estimate_normal(noisy_pts)
            if torch.isnan(n_est).any() or torch.isinf(n_est).any():
                print(n_est, p_n)
            err = angle_error(n_est, p_n)
            error_sum += err

        total_error += error_sum / noise_sample_per_plane
        valid_count += 1

    if valid_count == 0:
        error_loss = torch.tensor(0.0, device=ray_o.device)
    else:
        error_loss = total_error / valid_count
    
    range_penalty /= len(plane_dataset)
    print(error_loss.data, angle_penalty.data, position_penalty.data, range_penalty.data)
    loss = error_loss + \
           angle_penalty_weight * angle_penalty + \
           position_penalty_weight * position_penalty + \
           range_penalty_weight * range_penalty
    return loss.requires_grad_()

# 평면 데이터셋 CSV 로드 함수
def load_plane_dataset(csv_path):
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            'normal': [row['normal_x'], row['normal_y'], row['normal_z']],
            'point':  [row['point_x'],  row['point_y'],  row['point_z']]
        })
    return dataset

# 학습 루프 실행 함수
def train_sensor_model(csv_path, num_sensors=5, epochs=200, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_plane_dataset(csv_path)
    sensor = Sensor(num_sensors).to(device)
    visualize_sensor(sensor, save_path="sensor_snapshots/initial.png")
    optimizer = torch.optim.Adam(sensor.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = evaluate_model(sensor, dataset)
        if torch.isnan(loss):
            raise ValueError(f"NaN detected in loss at epoch {epoch + 1}. Training aborted.")
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        if (epoch + 1) % 20 == 0:
            visualize_sensor(sensor, save_path=f"sensor_snapshots/epoch_{epoch+1}.png")

    return sensor

# 최종 결과 시각화
def visualize_sensor(sensor, save_path=None):
    with torch.no_grad():
        pos, dirs = sensor.get_rays()
        pos = pos.cpu().numpy()
        dirs = dirs.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                dirs[:, 0], dirs[:, 1], dirs[:, 2],
                length=0.2, normalize=True, color='blue')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimized Sensor Positions and Directions')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 사용 예시:
sensor = train_sensor_model("plane_data.csv", num_sensors=20, epochs=200, lr=0.01)
visualize_sensor(sensor)
