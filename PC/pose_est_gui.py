from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import sys
import pandas as pd
import serial

class Sensor:
    def __init__(self, dx, dy, dz, yaw=0, pitch=0):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.pitch = pitch
        self.dist = 0.0
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0

    def update_distance(self, dist):
        self.dist = dist
        if dist < 8000:
            self.px = self.dx + dist * np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
            self.py = self.dy + dist * np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
            self.pz = self.dz + dist * np.sin(np.radians(self.pitch))
        else:
            self.px = self.py = self.pz = np.nan

class SensorManager:
    def __init__(self):
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def remove_sensor(self, index):
        if 0 <= index < len(self.sensors):
            del self.sensors[index]

    def update_from_serial(self, line):
        try:
            dists = list(map(float, line.strip().split(',')))
            for i, dist in enumerate(dists):
                if i < len(self.sensors):
                    self.sensors[i].update_distance(dist)
        except Exception as e:
            print(f"Failed to parse serial line: {line}, Error: {e}")

    def compute_normal(self):
        points = np.array([[s.px, s.py, s.pz] for s in self.sensors if s.dist < 8000])
        print(points)
        if len(points) < 3:
            return None, None, points
        centroid = np.mean(points, axis=0)
        u, s, vh = np.linalg.svd(points - centroid)
        normal = vh[-1]
        return normal / np.linalg.norm(normal), centroid, points

    def to_dataframe(self):
        data = [
            [s.dx, s.dy, s.dz, s.yaw, s.pitch, s.dist]
            for s in self.sensors
        ]
        return pd.DataFrame(data, columns=['dx', 'dy', 'dz', 'yaw', 'pitch', 'dist'])

    def from_dataframe(self, df):
        self.sensors.clear()
        for _, row in df.iterrows():
            sensor = Sensor(row['dx'], row['dy'], row['dz'], row['yaw'], row['pitch'])
            sensor.dist = row['dist']
            self.sensors.append(sensor)

class MatplotManager(FigureCanvas):
    def __init__(self, sensor_manager):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.sensor_manager = sensor_manager
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(500)

    def update_plot(self):
        self.ax.cla()
        normal, center, points = self.sensor_manager.compute_normal()
        if points is not None and points.ndim == 2 and points.shape[0] > 0:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', label='Sensor Points')
        

        if normal is not None and center is not None:
            self.ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=50, color='b')
            self.ax.text2D(0.05, 0.95, f"Normal: {normal.round(2)}", transform=self.ax.transAxes)
        self.ax.set_xlim(0, 300)
        self.ax.set_ylim(-100, 100)
        self.ax.set_zlim(-100, 100)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.draw()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Normal Vector Viewer")
        self.resize(1000, 600)

        self.sensor_manager = SensorManager()
        self.sensor_manager.add_sensor(Sensor(0, 0, 0))
        self.sensor_manager.add_sensor(Sensor(0, 10, 0))
        self.sensor_manager.add_sensor(Sensor(0, 0, 10))
        self.sensor_manager.add_sensor(Sensor(0, 10, 10))

        self.plot_canvas = MatplotManager(self.sensor_manager)
        self.sensor_table = QtWidgets.QTableWidget(0, 6)
        self.sensor_table.setHorizontalHeaderLabels(['dx', 'dy', 'dz', 'yaw', 'pitch', 'dist'])

        self.add_button = QtWidgets.QPushButton("Add Sensor")
        self.add_button.clicked.connect(self.add_sensor_row)

        self.update_button = QtWidgets.QPushButton("Update Sensor Values")
        self.update_button.clicked.connect(self.update_sensors_from_table)

        self.save_button = QtWidgets.QPushButton("Save to Excel")
        self.save_button.clicked.connect(self.save_to_excel)

        self.load_button = QtWidgets.QPushButton("Load from Excel")
        self.load_button.clicked.connect(self.load_from_excel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_canvas)
        layout.addWidget(self.sensor_table)
        layout.addWidget(self.add_button)
        layout.addWidget(self.update_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.load_button)

        self.update_table()

        self.serial = serial.Serial('COM15', 115200, timeout=1)
        self.serial_timer = QtCore.QTimer()
        self.serial_timer.timeout.connect(self.read_serial_input)
        self.serial_timer.start(100)

    def update_table(self):
        self.sensor_table.setRowCount(len(self.sensor_manager.sensors))
        for i, sensor in enumerate(self.sensor_manager.sensors):
            for j, value in enumerate([sensor.dx, sensor.dy, sensor.dz, sensor.yaw, sensor.pitch, sensor.dist]):
                item = QtWidgets.QTableWidgetItem(str(round(value, 2)))
                self.sensor_table.setItem(i, j, item)

    def add_sensor_row(self):
        self.sensor_manager.add_sensor(Sensor(0, 0, 0))
        self.update_table()

    def update_sensors_from_table(self):
        for i in range(self.sensor_table.rowCount()):
            try:
                dx = float(self.sensor_table.item(i, 0).text())
                dy = float(self.sensor_table.item(i, 1).text())
                dz = float(self.sensor_table.item(i, 2).text())
                yaw = float(self.sensor_table.item(i, 3).text())
                pitch = float(self.sensor_table.item(i, 4).text())
                self.sensor_manager.sensors[i].dx = dx
                self.sensor_manager.sensors[i].dy = dy
                self.sensor_manager.sensors[i].dz = dz
                self.sensor_manager.sensors[i].yaw = yaw
                self.sensor_manager.sensors[i].pitch = pitch
            except:
                continue

    def save_to_excel(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Sensor Data", "sensors.xlsx", "Excel Files (*.xlsx)")
        if path:
            df = self.sensor_manager.to_dataframe()
            df.to_excel(path, index=False)

    def load_from_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Sensor Data", "", "Excel Files (*.xlsx)")
        if path:
            df = pd.read_excel(path)
            self.sensor_manager.from_dataframe(df)
            self.update_table()

    def read_serial_input(self):
        if self.serial.in_waiting:
            line = self.serial.readline().decode(errors='ignore').strip()
            if line:
                self.sensor_manager.update_from_serial(line)
                self.update_table()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())