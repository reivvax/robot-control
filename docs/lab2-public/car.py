import mujoco
import matplotlib.pylab as plt
import numpy as np

DURATION = 3
DT = 1/30
ANGLE = np.pi

with open('car_xml_template.xml', mode='r') as f:
    model = mujoco.MjModel.from_xml_string(f.read())

data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 1280, 1280)
car_id = model.body("car").mocapid[0]


def translation(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T

def rotation_z(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

def pose_to_quat(T):
    pos = T[:3, 3].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    return pos, quat


current_time = 0
start_pos = np.array([-1, -1, 0.1])
rot_center = np.array([-1., 0., 0.1])
T_start = translation(start_pos)

n_steps = int(DURATION / DT)

for i in range(n_steps + 1):
    theta = ANGLE * i / n_steps

    T = translation(rot_center) @ rotation_z(theta) @ translation(-rot_center) @ T_start

    pos, quat = pose_to_quat(T)

    data.mocap_pos[car_id] = pos
    data.mocap_quat[car_id] = quat

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    plt.imsave(f"frames/frame_{i:03d}.png", renderer.render())

renderer.close()
