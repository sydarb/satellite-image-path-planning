import os
import json
import numpy as np
import skimage.morphology as skm
import skimage.filters as skf
from predict import *
from route_finder import *
#import argparse

ROOT_DIR = os.getcwd()
SaveImages = True

filepath = ROOT_DIR+'/Files/'
outpath = ROOT_DIR+'/Outputs/'
out_imgpath = outpath+'images/'
json_filename = "submission_HiveMind"

def main():
    imagenames = []
    filenames = os.listdir(filepath)
    imagenames=[]
    for file in filenames:
        if file.split('.')[-1] == 'png':
            imagenames.append(file)
    #print(imagenames)
    n = len(imagenames)
    i=0
    lengths = []
    ans = {}
    for imagename in imagenames:
        i=i+1
        print('working on image', i, 'out of', n, 'images')
        image_path = os.path.join(filepath, imagename)
        raw_image = cv2.imread(image_path)
        json_name= imagename.split(".")[0]+'.json'
        f = open(filepath + json_name)
        data = json.load(f)
        f.close()
        start = data['Start']
        end = data['End']
  
        mask = get_mask(image_path, imagename.split(".")[0], out_imgpath=out_imgpath, SaveImages=True)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=6)
        gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        (th, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        skell = skm.thin(bw)
        skell = skell.astype("uint8")*255
        dil = cv2.dilate(skell, kernel, 16)
        thinned = cv2.cvtColor(skell, cv2.COLOR_GRAY2RGB)
        thinned = cv2.dilate(thinned, kernel, iterations=1)

        top_left = (0,0)
        bott_right = skell.shape[::-1]
        color = (0,0,0)
        thickness = 5
        roads = cv2.rectangle(thinned, top_left, bott_right, color, thickness)
        if SaveImages:
            cv2.imwrite(out_imgpath+imagename.split(".")[0]+"_skeleton.png", roads)

        road_start = find_nearest_white(dil, start)[0]
        road_end = find_nearest_white(dil, end)[0]

        p1= []
        p3 = []
        p1, p3 = off_road_paths(start, road_start, end, road_end)
        p2 = find_shortest_path(roads, road_end, road_start)
        p = p1 + p2[1:-1] + p3

        image = raw_image.copy()
        back = np.zeros(image.shape, dtype = np.uint8)
        drawPath(back,p)
  
        im1 = skf.unsharp_mask(back, radius=2.0, amount=-2.0, preserve_range=True).astype("uint8")
        im2 = cv2.dilate(back, kernel, iterations=5)
        gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        (th, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        im3 = skm.thin(bw)
        im3 = im3.astype("uint8")*255
        im4 = cv2.GaussianBlur(im3, (15, 15), 0)
        (th, bw2) = cv2.threshold(im4, 8, 255, cv2.THRESH_BINARY)
        im5 = cv2.cvtColor(bw2, cv2.COLOR_GRAY2RGB)
        im5 = cv2.erode(im5, kernel, iterations=1)

        im6 = skm.skeletonize(im5)
        im6 = im6.astype("uint8")
        cv2.circle(im6, (start[0],start[1]), 20, (0,255,0), -1)
        cv2.circle(im6, (end[0],end[1]), 20, (0,255,0), -1)
        im6 = cv2.dilate(im6, kernel, iterations=1)
        if SaveImages:
            cv2.imwrite(out_imgpath+imagename.split(".")[0]+"_path.png", im6)

        path = find_shortest_path(im6, end, start)
        colorPath(image,path)
        if SaveImages:
            cv2.imwrite(out_imgpath+imagename.split(".")[0]+"_route.png", image)
        ans[imagename]= path
        lengths.append(len(path))
        print('image', i, 'completed')

    json_object = json.dumps(ans, indent = 4)
    with open(outpath + json_filename + ".json", "w") as outfile:
        outfile.write(json_object)
    #print(lengths)

if __name__ == '__main__':
    main()

