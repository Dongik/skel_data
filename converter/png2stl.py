import subprocess as sp
import cv2

def test_converting(input_dir):
    img = cv2.imread(input_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    in_size = 100, 300
    out_size = 10, 30

    img = cv2.resize(img, dsize=in_size)

    ret, mask = cv2.threshold(img,1,255,cv2.THRESH_BINARY)

    cv2.imshow("show", mask)
    cv2.waitKey(0)

    print(img.shape)

    cv2.imshow("show", img)
    cv2.waitKey(0)

    # print("r = {}".format(img.shape[0]/img.shape[1]))

    img = cv2.resize(img, dsize=out_size)
    
    # maxh = img.amax()
    # minh = img.amin()

    cv2.imshow("show", img)
    cv2.waitKey(0)


    
    img = cv2.resize(img, dsize=in_size, interpolation=cv2.INTER_LINEAR)
    ks = 15
    
    img = cv2.GaussianBlur(img,(ks,ks),0)

    cv2.imshow("show", img)
    cv2.waitKey(0)

    img = cv2.bitwise_and(img, mask)


    cv2.imshow("show", img)
    cv2.waitKey(0)

    cv2.imwrite('interpolated.png', img)


    # img = cv2.resize(img, dsize=())
    
    # output_dir = input_dir.replace(".png", ".csv")
    # df = pd.DataFrame(img)
    
    # df.to_csv(output_dir)

    sp.call(["./hmstl", "-z", "0.1", "-i", "interpolated.png", "-o", "interpolated.stl"])
    

if __name__ == "__main__":
    test_converting("left_orthotics.png")
