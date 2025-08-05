import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#torch.autograd.set_detect_anomaly(True)

# 거리 기반 센서 오차 모델
def sensor_noise_std(dist, a=0.01, b=0.002):
    return a * dist + b

# 센서 클래스 정의
class Sensor(nn.Module):
    def __init__(self, num_sensors):
        super().__init__()
        self.num = num_sensors
        init_pos = torch.randn(num_sensors, 3) * 0.1
        init_pos[:, 2] = 0.0  # z=0에 고정
        self.pos = nn.Parameter(init_pos)
        self.dir_angles = nn.Parameter(torch.randn(num_sensors, 2))  # theta, phi

    def get_rays(self):
        # z=0에 고정
        self.pos.data[:, 2] = 0.0
        theta = self.dir_angles[:, 0]  # 위도
        phi = self.dir_angles[:, 1]    # 방위각
        dx = torch.sin(theta) * torch.cos(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(theta)
        dirs = torch.stack([dx, dy, dz], dim=1)
        dirs = F.normalize(dirs, dim=1)
        return self.pos, dirs

# ray-plane 교점 계산
def ray_plane_intersect(ray_o, ray_d, plane_n, plane_p):
    denom = ray_d @ plane_n
    t = ((plane_p - ray_o) @ plane_n) / denom
    point = ray_o + t.unsqueeze(-1) * ray_d
    return point, t

# 법선벡터 추정 (최소제곱 SVD)
def estimate_normal(points):
    centroid = points.mean(dim=0)
    centered = points - centroid
    mask = ~(torch.isnan(centered).any(dim=1) | torch.isinf(centered).any(dim=1))
    centered = centered[mask]

    if centered.shape[0] < 3 or centered.std() < 1e-6:
        # fallback normal: upward z-axis
        return torch.tensor([0., 0., 1.], device=points.device)

    try:
        _, _, V = torch.linalg.svd(centered)
        normal = V[-1, :]
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    except RuntimeError:
        normal = torch.tensor([0., 0., 1.], device=points.device)

    if normal[-1] < 0:
        normal = -normal
    return normal


# 각도 오차 계산
def angle_error(n_est, n_true):
    cos_sim = F.cosine_similarity(n_est, n_true, dim=0).clamp(-0.999999, 0.999999)
    return torch.acos(cos_sim) * 180 / torch.pi

# 평가 함수 (평균 각도 오차 + 레귤레이션 항)
def evaluate_model(sensor: Sensor, plane_dataset, noise_sample_per_plane=5):
    torch.set_grad_enabled(True)
    ray_o, ray_d = sensor.get_rays()
    # 센서 위치 제약: 직사각형 크기를 넘지 않도록 ReLU 규제 적용
    regulation_penalty = torch.tensor(0.0, device=ray_o.device)
    dist_penalty = torch.tensor(0.0, device=ray_o.device)
    negdist_penalty = torch.tensor(0.0, device=ray_o.device)
    boundary_penalty = torch.tensor(0.0, device=ray_o.device)
    angle_penalty = torch.tensor(0.0, device=ray_o.device)
    boundary = 0.5  # 허용된 최대 절댓값 거리 (x, y 각각)
    boundary_penalty += F.relu(torch.abs(ray_o[:, 0]) - boundary).sum()
    boundary_penalty += F.relu(torch.abs(ray_o[:, 1]) - boundary).sum()
    total_error = torch.tensor(0.0, device=ray_o.device)
    valid_count = 0

    for plane in plane_dataset:
        p_n = torch.tensor(plane['normal'], dtype=torch.float32, device=ray_o.device)
        p_p = torch.tensor(plane['point'], dtype=torch.float32, device=ray_o.device)

        error_sum = torch.tensor(0.0, device=ray_o.device)
        for _ in range(noise_sample_per_plane):
            inter_pts, dists = ray_plane_intersect(ray_o, ray_d, p_n, p_p)
            boundary = 3.0  # 허용된 최대 절댓값 거리 (x, y 각각)
            dist_penalty += F.relu(dists - boundary).sum()
            negdist_penalty += F.relu(-dists).sum()
            
            # 유효하지 않은 거리나 NaN 탐지
            invalid_mask = (dists < 0) | torch.isnan(dists) | torch.isinf(dists)
            if invalid_mask.any():
                continue

            # 방향 패널티: 센서 방향이 평면을 향하지 않을 경우
            cos_angle = (ray_d @ p_n)  # ray가 평면을 얼마나 잘 향하는가
            angle_thresh = torch.cos(torch.tensor(torch.deg2rad(torch.tensor(90.0)), device=ray_o.device))
            penalty_a = F.relu((angle_thresh - cos_angle))
            angle_penalty += penalty_a.sum()
            noise_std = sensor_noise_std(dists)
            # noise 항은 gradient가 끊기지 않도록 detaching
            noise = (torch.randn_like(inter_pts) * noise_std.unsqueeze(-1)).detach()
            noisy_pts = inter_pts + noise

            n_est = estimate_normal(noisy_pts)
            err = angle_error(n_est, p_n)
            error_sum += err

        total_error += error_sum / noise_sample_per_plane
        valid_count += 1
    print(dist_penalty, negdist_penalty, boundary_penalty, angle_penalty)
    regulation_penalty = dist_penalty * 50.0 + negdist_penalty * 50.0 + boundary_penalty * 50.0 + angle_penalty * 50.0
    if valid_count == 0:
        raise ValueError("No valid plane measurements during evaluation.")
    return ((total_error / valid_count) + regulation_penalty).requires_grad_()

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
    optimizer = torch.optim.Adam(sensor.parameters(), lr=lr)

    import os
    os.makedirs("sensor_snapshots", exist_ok=True)

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
sensor = train_sensor_model("plane_data.csv", num_sensors=5, epochs=200, lr=0.01)
visualize_sensor(sensor)