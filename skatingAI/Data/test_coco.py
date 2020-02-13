from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

coco=COCO('coco/annotations/person_keypoints_train2017.json')

imgIds = coco.getImgIds(catIds=[1])
print(f"Coco dataset contains {len(imgIds)} annotated images.")

for i in range(10):
    randomImg = imgIds[np.random.randint(0,len(imgIds))] #40954 #450559
    coco_img = coco.loadImgs([randomImg])[0]
    print(coco_img)



    annIds = coco.getAnnIds(imgIds=coco_img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        kps = np.array(ann.get('keypoints')).reshape((-1,3))[:,:2]
        print(kps)
        print('-'*100)

    I = io.imread(coco_img['coco_url'])
    plt.imshow(I)
    plt.show()


for kp in kps:
    I[kp[1],kp[0]]=[255,0,0]
    I[kp[1]+1,kp[0]]=[255,0,0]
    I[kp[1],kp[0]+1]=[255,0,0]
    I[kp[1]+1,kp[0]+1]=[255,0,0]
print(kps)
print(I.shape)
print(I[186,98])

plt.imshow(I)
plt.show()