
# from torchvision import read_image
# from torchvision.transforms import Grayscale
import cv2
import pandas as pd
import glob
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
    for target_dir in glob.glob("orthotics_data/**/aligned_*.png"):
        png2csv(target_dir)


if __name__ == "__main__":
    png2csv_all()

