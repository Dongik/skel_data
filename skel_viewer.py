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

# parent of joint
poj = joint_tree

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


def update_bones(frame_num, skel_data, bone_lines):
    skel = skel_data[frame_num]
    for i, bone_line in enumerate(bone_lines):
        b = bone(i, skel)
        # print("b = {}".format(b))
        bone_line.set_data(b[:2])
        bone_line.set_3d_properties(b[2])

    return bone_lines


def bone(i, s):
    j0, j1 = i * 3, joint_tree[i] * 3
    # return np.asarray([[s[j0 + k], s[j1 + k]] for k in range(3)]) 
    return np.asarray([[s[i][k], s[poj[i]][k]] for k in range(3)])



default_skel_data = "skeleton_data/skeleton_swagging.csv"

    

def play_skeleton(skel_dir=default_skel_data):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    df = pd.read_csv(skel_dir)
    s = df.loc[:, "0.x":"16.z"].values
    
    s = s.reshape(s.shape[0], 17, 3)
    s[:,:,2] *= -1

    bone_lines = []
    
    for i in range(17):
        b = bone(i, s[0])
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
    line_ani = animation.FuncAnimation(fig, update_bones, s.shape[0], fargs=(s, bone_lines),
                                    interval=1000/30, blit=False)

    plt.show()


if __name__ == "__main__":
# Attaching 3D axis to the figure
    # show_skel()
    play_skeleton()
    