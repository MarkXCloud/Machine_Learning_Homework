import numpy as np
import cv2
img_lst = [ '280',  '1560', '3287',  '100015','133391','203099','208365']
fore_ave = []
back_ave = []
for img in img_lst:
    GT = cv2.imread("./MSRA10K_Imgs_GT/Imgs/"+img+".png")
    GT = cv2.resize(GT, (288, 288), interpolation=cv2.INTER_CUBIC)
    GT = np.float32(GT)
    GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    GT /= 255
    GT[GT > 0.5] = 1
    GT[GT < 0.5] = 0
    pre = cv2.imread("./result/"+img+".png")
    pre = np.float32(pre)
    pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    pre /= 255
    pre[pre > 0.5] = 1
    pre[pre < 0.5] = 0
    #foreground
    intersection =GT*pre
    union = GT+pre
    union[union > 0.5] = 1
    union[union < 0.5] = 0
    fore_ave.append(np.sum(intersection)/np.sum(union))
    #background
    GT = 1-GT#reverse the pixel, so we get background
    pre = 1-pre
    intersection = GT * pre
    union = GT + pre
    union[union > 0.5] = 1
    union[union < 0.5] = 0
    back_ave.append(np.sum(intersection) / np.sum(union))

fore_ave = np.array(fore_ave)
back_ave = np.array(back_ave)
print(np.mean(fore_ave),np.mean(back_ave))