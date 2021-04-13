import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--skeleton', default='skeleton_keep_walk.csv', type=str, metavar='NAME', help='target dataset')
args = parser.parse_args()


# Fixing random state for reproducibility
np.random.seed(19680801)

joint_names=[
    # 0-4
    'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    # 5-9
    'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
    # 10-14
    'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
    # 15-16
    'spine', 'head',
]

joint_tree=[
    1, 15, 1, 2, 3,
    1, 5, 6, 14, 8,
    9, 14, 11, 12, 14,
    14, 1
]

hflip_indices=[
    0, 1, 5, 6, 7,
    2, 3, 4, 11, 12,
    13, 8, 9, 10, 14,
    15, 16
]


p_pos = [
    (0.267, 0.832), # 0
    (0.181, 0.880), # 1
    (0.131, 0.886), # 2
    (0.066, 0.881), # 3
    (0.292, 0.655), # 4
    (0.209, 0.680), # 5
    (0.133, 0.689), # 6
    (0.058, 0.705), # 7
    (0.279, 0.554), # 8
    (0.255, 0.389), # 12
    (0.185, 0.438), # 13
    (0.118, 0.463), # 14
    (0.217, 0.242), # 17
    (0.119, 0.251), # 18
    (0.194, 0.079), # 21
    (0.107, 0.094), # 22
]



class Skel:
    def __init__(self, row):
        self.row = row
    
    def joint(self, i):
        # print("type = {} at joint".format(type(i)))
        ii = 84 + i * 6
        r = self.row
        return [r[ii], r[ii + 1], -r[ii + 2]]

    def bone(self, i):
        b = [self.joint(i), self.joint(joint_tree[i])]
        return np.array(b).swapaxes(0, -1)


def update_bones(frame_num, skels, bone_lines):
    skel = skels[frame_num]
    for i, bone_line in enumerate(bone_lines):
        b = skel.bone(i)
        # print("b = {}".format(b))
        bone_line.set_data(b[:2])
        bone_line.set_3d_properties(b[2])

    return bone_lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)


filedir = args.skeleton
df = pd.read_csv(filedir)
print("read file {}".format(filedir))
skels = [Skel(row) for i, row in df.iterrows()]

bone_lines = []
skel = skels[0]
for i in range(17):
    b = skel.bone(i)
    bone_lines.append(ax.plot(b[0], b[1], b[2])[0])


# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_bones, len(df.index), fargs=(skels, bone_lines),
                                   interval=1000/30, blit=False)

plt.show()