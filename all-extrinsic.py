import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 文件路径
extrinsic_save_file = r'C:\Users\14168\1\Python\pythonProject\homework\lab03-sfm-ba-HM-ZC-main\predictions\temple\results\no-bundle-adjustment\all-extrinsic.json'

# 读取 JSON 文件
with open(extrinsic_save_file, 'r') as f:
    all_extrinsics = json.load(f)

# 将数据转换为 numpy 数组
for image_id, extrinsic in all_extrinsics.items():
    all_extrinsics[image_id] = np.array(extrinsic)

# 初始化 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取相机的平移向量（tvec）
camera_positions = []
for image_id, extrinsic in all_extrinsics.items():
    # 提取平移向量 (tvec)
    tvec = extrinsic[:, 3]
    camera_positions.append(tvec)

camera_positions = np.array(camera_positions)

# 绘制相机路径
ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], '-o', label='Camera Path')

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Path in 3D Space')
ax.legend()

plt.show()