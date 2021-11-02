import cv2
import numpy as np
from PCA import PCA
np.random.seed(234)

def random_points(w, h, lst, n):
    randw = np.random.randint(low=w//3, high=w//3*2, size=n).reshape(-1, 1)
    randh = np.random.randint(low=h//3, high=h//3*2, size=n).reshape(-1, 1)
    rand_point = np.hstack((randw, randh))
    result = []
    for point in rand_point:
        result.append([*point, *lst[point[0]][point[1]]])  # [x,y,L,a,b] position and Lab color
    #print("init centers: ",temp_result)
    return np.array(result)


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_pq(x, y):
    return euclidean_distance(x[0:2], y[0:2]) / (1 + 3 * euclidean_distance(x[2:], y[2:]))  # d_color/(1+c*d_position)

def divide_cluster(w,h,data,n,center):
    cluster = []
    for _ in range(n):
        cluster.append([])
    for i in range(w):
        for j in range(h):
            # fench every point to judge
            dis_max = -1  # initiate min distance, actually, the word "likelihood" is more proper
            cluster_tag = None  # to remeber which cluster it belongs to
            cur_point = [i, j, *data[i][j]]
            for k in range(n):
                dis = distance_pq(cur_point, center[k])
                if dis > dis_max:
                    dis_max = dis
                    cluster_tag = k
            cluster[cluster_tag].append(cur_point)
    return cluster

def cal_new_center(lst):
    return np.mean(lst,axis=0)


class K_means:
    def __init__(self, n):
        """
        K-means algorithm for pictures, so there mightbe some differences.
        :param n:number of clusters
        """
        self._n = n
        self._center = None

    def fit(self, data,epoch):
        w, h, _ = data.shape
        #print(w, h)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)#convert dim

        self._center = random_points(w, h, data, self._n)
        #print(self._center)
        ini_cluster = divide_cluster(w,h,data,self._n,self._center)
        old_cluster = ini_cluster
        for ep in range(epoch):
            new_centers = []
            for point in old_cluster:#calculate new cluster center
                temp_center = cal_new_center(point)
                new_centers.append(temp_center)
            #print("epoch ",ep," New centers: ",new_centers)
            print("epoch: ",ep)
            #print(new_centers)
            new_centers = np.array(new_centers)
            new_cluster = divide_cluster(w,h,data,self._n,new_centers)
            old_cluster = new_cluster
        final_center = []
        for point in old_cluster:#get final center after all epochs
            temp_center = cal_new_center(point)
            final_center.append(temp_center)
        self._center = np.array(final_center)

    def predict(self,data):
        w, h, _ = data.shape
        data = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)
        pre_cluster = divide_cluster(w,h,data,self._n,self._center)
        for k in range(self._n):
            for point in pre_cluster[k]:
                i,j = point[0],point[1]
                data[i][j] = self._center[k][2:]

        return cv2.cvtColor(data,cv2.COLOR_Lab2BGR)




if __name__ == '__main__':
    img_lst = [ '280',  '1560', '3287',  '100015','133391','203099','208365']
    for img_name in img_lst:
        print(img_name)
        path = './MSRA10K_Imgs_GT/Imgs/'+img_name+'.jpg'
        img = cv2.imread(path)
        #cv2.imshow("prime pic",img)
        img = cv2.resize(img,(288,288),interpolation=cv2.INTER_CUBIC)
        img_clster = np.float32(img)
        kmeans = K_means(n=5*5)
        kmeans.fit(img,epoch=20)
        new_img = kmeans.predict(img).astype(np.uint8)
        #cv2.imshow("cluster pic",new_img)
        cv2.imwrite("./clus_Imgs/"+img_name+".jpg",new_img)
        #cv2.waitKey(0)
