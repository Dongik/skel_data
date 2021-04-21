
# from torchvision import read_image
# from torchvision.transforms import Grayscale
import pandas as pd
import glob
from multiprocessing import Process, Pool
import os
source_prefix = "original"


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


def png2csv(input_dir, output_size=(10, 30)):
    img = cv2.imread(input_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, dsize=output_size)
    
    maxh = img.amax()
    minh = img.amin()

    cv2.imshow("show", img)
    cv2.waitKey(0)
    
    output_dir = input_dir.replace(".png", ".csv")
    df = pd.DataFrame(img)
    
    df.to_csv(output_dir)
    

def png2csv_all():
    import cv2

    for target_dir in glob.glob("orthotics_data/**/aligned_*.png"):
        png2csv(target_dir)


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

def slice_old_skeleton_csv():

    cols = pd.np.r_[3:25, 44:66, 84:185]
    print(cols)
    
    for old_csv_dir in glob.glob("skeleton_data/old_*.csv"):
        new_csv_dir = old_csv_dir.replace("old_", "")
        o_df = pd.read_csv(old_csv_dir)
        print(print("col n = {}".format(o_df.shape)))
        n_df = pd.DataFrame(o_df.iloc[:,cols])
        n_df.to_csv(new_csv_dir)



def slice_old_pressure():
    cols = pd.np.r_[3:25, 44:66]
    print(cols)
    for old_csv_dir in glob.glob("orthotics_data/**/foot_pressures.csv"):
        new_csv_dir = old_csv_dir.replace("foot_pressures", "insole_sensor_data")
        o_df = pd.read_csv(old_csv_dir)
        print(print("col n = {}".format(o_df.shape)))
        n_df = pd.DataFrame(o_df.iloc[:,cols])
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
    slice_old_pressure()
    pass

