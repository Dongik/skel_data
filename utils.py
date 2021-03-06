
# from torchvision import read_image
# from torchvision.transforms import Grayscale
import pandas as pd
import glob
from multiprocessing import Process, Pool
import os
source_prefix = "original"
import numpy as np
from tqdm import tqdm

joint_index = [
    1, 15, 1, 2, 3,
    1, 5, 6, 14, 8,
    9, 14, 11, 12, 14,
    14, 1
]

class Skel:
    def __init__(self, s_row):
        self.s_row = s_row
    
    def joint(self, i):
        # print("type = {} at joint".format(type(i)))
        
        ii = i * 3
        if self.s_row:
            r = self.s_row
            return [r[ii], r[ii + 1], -r[ii + 2]]
        

    def bone(self, i):
        b = [self.joint(i), self.joint(joint_index[i])]
        return np.array(b).swapaxes(0, -1)



def png2csv(input_dir, output_size=(10, 30)):
    import cv2

    img = cv2.imread(input_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(img.shape)
    print("r = {}".format(img.shape[0]/img.shape[1]))

    img = cv2.resize(img, dsize=output_size)
    
    # maxh = img.amax()
    # minh = img.amin()

    # cv2.imshow("show", img)
    # cv2.waitKey(0)


    a = 5
    img = cv2.resize(img, dsize=(10 * a, 30 * a), interpolation=cv2.INTER_LINEAR)
    
    # cv2.imshow("show", img)
    # cv2.waitKey(0)

    cv2.imwrite('interpolated.png', img)

    # img = cv2.resize(img, dsize=())
    
    output_dir = input_dir.replace(".png", ".csv")
    df = pd.DataFrame(img)
    
    df.to_csv(output_dir)
    

def png2csv_all():
    import cv2

    for target_dir in glob.glob("orthotics_data/**/*_orthotics.png"):
        png2csv(target_dir)
        break


def slice_gyro(df):
    return df[:,pd.np.r_[1:7, 26:42, 44:66]]

def slice_pressure(df):
    return df[:,pd.np.r_[1:7, 26:42, 44:66]]

def slice_skeleton(df):
    return df[:,pd.np.r_[1:7, 26:42, 44:66]]


def rename_files():
    for old_dir in glob.glob("orthotics_data/**/aligned_*"):
        print("rename {}".format(old_dir))
        new_dir = old_dir.replace("aligned_", "").replace(".", "_orthotics.")
        os.rename(old_dir, new_dir)





p_cols = list(range(9, 25)) + list(range(50, 66))
g_cols = list(range(3, 10)) + list(range(44, 51))
cols = list(range(3))

# def slice_old_skeleton_csv():

#     cols = pd.np.r_[3:25, 44:66, 84:185]
#     print(cols)
    
#     for old_csv_dir in glob.glob("skeleton_data/old_*.csv"):
#         new_csv_dir = old_csv_dir.replace("old_", "")
#         o_df = pd.read_csv(old_csv_dir)
#         print(print("col n = {}".format(o_df.shape)))
#         n_df = pd.DataFrame(o_df.iloc[:,cols])
#         n_df.to_csv(new_csv_dir)



def r_(s, e):
    return list(range(s, e))

def slice_old_pressure_to_skel():
    # cols = pd.np.r_[3:25, 44:66]
    cols = np.r_[3: 25, 44: 66, 84 : 84 + 3 * 17]
    # s_cols = []
    # for i in range(17):
    #     ii = i * 6
    #     ss = 84
    #     s_cols.append(ss + ii)
    #     s_cols.append(ss + ii + 1)
    #     s_cols.append(ss + ii + 2)
    
    # cols += s_cols

    print(cols)
    for old_csv_dir in glob.glob("legacy_skeleton_data/skeleton_*.csv"):
        new_csv_dir = old_csv_dir.replace("legacy_skeleton_data", "skeleton_data")
        o_df = pd.read_csv(old_csv_dir)
        print("col n = {}".format(o_df.shape))
        n_df = pd.DataFrame(o_df.iloc[:,cols])
        n_df.to_csv(new_csv_dir)

def slice_old_pressure():
    cols = r_(2, 24) + r_(43, 65)
    cols_2 = r_(3, 25) + r_(44, 66)

    print(cols)
    for old_csv_dir in glob.glob("orthotics_data/**/foot_pressures.csv"):
        new_csv_dir = old_csv_dir.replace("foot_pressures", "gyro_pressure")
        if "subject_5" in old_csv_dir:
            c = cols_2
        else:
            c = cols

        o_df = pd.read_csv(old_csv_dir)
        n_df = pd.DataFrame(o_df.iloc[:,c])
        print("col n = {}".format(o_df.shape))
        n_df.to_csv(new_csv_dir)


def transform():
    inputPoly = read_stl(filename="original_right.stl")
    transform = vtk.vtkTransform()

    transform.RotateWXYZ(45,0,1,0)


def read_stl(filename="original_right.stl"):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(reader.GetOutput())
    return inputPoly

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


# normalize by pelvis position
# normalize by pelvis width

def normalize_skel():
    cols = []
    axises = ['x', 'y', 'z']
    gyro_types = ['an', 'ac']
    sides = ['l', 'r']
    fsr_num = 16
    joint_num = 17

    for side in sides:
        for gyro_type in gyro_types:
            for axis in axises:
                cols.append("{}.{}.{}".format(side, gyro_type, axis))
        for fsr_index in range(fsr_num):
            cols.append("p{}".format(fsr_index))
    
    for ji in range(joint_num):
        for a in axises:
            cols.append("s.{}.{}".format(ji, a))
    
    new_skel_folder_name = "skeleton_data_v2"
    if not os.path.isdir(new_skel_folder_name):
        os.mkdir(new_skel_folder_name)

    for skel_dir in tqdm(glob.glob("skeleton_data/*.csv")):
        
        df = pd.read_csv(skel_dir)
        f = df.iloc[:, 1:45].values  # foot data
        s = df.iloc[:, 45:].values   # skel data

        ln = s.shape[0]
        jn = 17
        cn = 3

        # print("s.shape = {}".format(s.shape))
        s = np.reshape(s, (ln, jn, cn))
        
        p = s[:,joint_names.index("pelvis")] # pelvis
        sp = s[:,joint_names.index("spine")]

        lf = s[:, joint_names.index("left_ankle")]
        rf = s[:, joint_names.index("right_ankle")]
        
        g = np.minimum(lf[:,2:], rf[:,2:]) - 0.5
        # c = (lf[:,:2] + rf[:,:2]) / 2
        # print("g.s = {}, c.s = {}".format(g.shape, c.shape))
        

        gc = np.hstack([p[:,:2], g]) 
        
        sl = np.sqrt(np.sum((p - sp) ** 2, axis=1))

        # sl = sl.T 
        print("sl.shape = {}, ls = {}".format(sl.shape, sl[ln // 2]))
        sl = 0.2 / sl

        s = np.reshape(s, (ln, jn * cn))
        # s = s * sl.T
        
        

        # print("p.shape = {}".format(p.shape))
        scale = np.tile(sl, (jn * cn, 1))
        # pivot = np.tile(p, (1, jn))
        pivot = np.tile(gc, (1, jn))

        print("scale.s = {}, pivot.shape = {}".format(scale.shape, pivot.shape))
        # .reshape(ln, jn, cn)
        # print("pr.shape = {}".format(pr.shape))
        # print("pr = {}".format(pr[ln // 2]))
        

        # s = np.sum((s, pr), axis=2)
        s -= pivot
        s = scale.T * s

        
        
        # print("s.shape = {}".format(s.shape))

        # s = np.reshape(s, (ln, jn * cn))

        f2s = np.hstack([f, s])

        ndf = pd.DataFrame(f2s, columns=cols)
        
        n_skel_dir = skel_dir.replace("skeleton_data", "skeleton_data_v2")

        ndf.to_csv(n_skel_dir)



def decimation(input_dir, max_points=5000):

    # base_dir = "/data/balance/insole_walk_pairs/subject_3/"
    # base_dir = ""
    # InputFilename = "{}right_aligned.stl".format(base_dir)
    # OutputFilename = "{}right_decimated.stl".format(base_dir)
    
    output_dir = input_dir.replace(source_prefix, "decimated_aligned")

    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_dir)
    
    print("decimate {}".format(input_dir))
    reader.Update()

    # triangles = vtk.vtkTriangleFilter()
    # triangles.SetInputData(reader.GetOutput())
    # triangles.Update()
    # inputPoly = triangles.GetOutput()
    
    # sphereS = vtkSphereSource()
    # sphereS.Update()

    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(reader.GetOutput())

    input_pn = inputPoly.GetNumberOfPoints()

    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(inputPoly.GetNumberOfPoints()) + " points.\n"
          "There are " + str(inputPoly.GetNumberOfPolys()) + " polygons.\n")
    
    if input_pn < max_points:
        return
    
    
    reduction_rate = 1 - max_points / input_pn
    
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(reduction_rate)
    decimate.Update()

    print("Read decimated")

    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())


    # centerFilter = vtk.vtkCenterOfMass()
    # centerFilter.SetInputData(decimatedPoly)
    # centerFilter.SetUseScalarsAsWeights(False)
    # centerFilter.Update()
    # center = centerFilter.GetCenter()


    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimatedPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(decimatedPoly.GetNumberOfPolys()) + "polygons.\n")


    # ren = vtk.vtkRenderer()
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(decimatedPoly)
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)


    transformFilter = vtk.vtkTransformPolyDataFilter()
    transform = vtk.vtkTransform()
    # transform.Translate(-center[0], -center[1], -center[2])
    
    dg = 180
    if '_right' in input_dir:
        rx, ry, rz = 132, -11.9, -88.6
        wa, xa, ya, za = 0.124, 0.378, 0.802, 0.445
    elif '_left' in input_dir:
        rx, ry, rz = -40.5, 9.01, -86.9
        wa, xa, ya, za = 0.356, 0.614, -0.639, -0.296
    
    transform.RotateWXYZ(wa, xa, ya, za)
    # transform.RotateX(rx)
    # transform.RotateY(ry)
    # transform.RotateZ(rz)

    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(decimatedPoly)
    transformFilter.Update()


    # triangleTrans = vtk.vtkTriangleFilter()
    # triangleTrans.SetInputData(transformFilter.GetOutputPort())
    # triangleTrans.Update()

    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(output_dir)
    stlWriter.SetFileTypeToBinary()
    stlWriter.SetInputConnection(transformFilter.GetOutputPort())
    stlWriter.Write()

    print("saved as {}".format(output_dir))
    return True

def decimate(workers=12):

    import vtk
    with Pool(processes=workers) as pool:
        targets = glob.glob("subject_datasets/**/{}_*.stl".format(source_prefix))
        print(pool.map(decimation, targets))

if __name__ == "__main__":
    # rename_files()
    # png2csv_all()
    # slice_old_pressure_to_skel()
    normalize_skel()
    pass

