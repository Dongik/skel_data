

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
