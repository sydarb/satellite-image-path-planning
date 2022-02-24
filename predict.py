
import os
import cv2
import torch
import segmentation_models_pytorch as smp
from utils import *

preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
select_class_rgb_values = [[0, 0, 0], [255, 255, 255]]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.getcwd()
MODEL_2_PATH = ROOT_DIR+'/TrainedModels/DeepLabV3_ResNet50_DeepGlobe.pth'
MODEL_1_PATH = ROOT_DIR+'/TrainedModels/UNet_ResNet34_Massachusetts.pth'

if os.path.exists(MODEL_1_PATH):
    my_model_1 = torch.load(MODEL_1_PATH, map_location=DEVICE)
if os.path.exists(MODEL_2_PATH):
    my_model_2 = torch.load(MODEL_2_PATH, map_location=DEVICE)

def get_submasks(img_path, crop_size):
    t = crop_size
    img = cv2.imread(img_path)
    image= cv2.resize(img,(2048,1024))
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    crop1 = image[0:1024, 0:1024]
    crop2 = image[0:1024, 1024:2048]
    crop1 = cv2.resize(crop1, (t,t))
    crop2 = cv2.resize(crop2, (t,t))
    crop1 = preprocessing_fn(crop1)
    crop2 = preprocessing_fn(crop2)

    x1_tensor  = to_tensor(crop1)
    x1_tensor = torch.from_numpy(x1_tensor).to(DEVICE).unsqueeze(0)
    pred_mask1 = my_model_1(x1_tensor)
    pred_mask1 = pred_mask1.detach().squeeze().cpu().numpy()
    pred_mask1 = np.transpose(pred_mask1,(1,2,0))
    pred_mask1 = colour_code_segmentation(reverse_one_hot(pred_mask1), select_class_rgb_values)

    x2_tensor  = to_tensor(crop2)
    x2_tensor = torch.from_numpy(x2_tensor).to(DEVICE).unsqueeze(0)
    pred_mask2 = my_model_1(x2_tensor)
    pred_mask2 = pred_mask2.detach().squeeze().cpu().numpy()
    pred_mask2 = np.transpose(pred_mask2,(1,2,0))
    pred_mask2 = colour_code_segmentation(reverse_one_hot(pred_mask2), select_class_rgb_values)

    final = cv2.hconcat([pred_mask1, pred_mask2])
    
    size=img.shape[:2]
    size = size[::-1]
    final = final.astype('float32')
    submask = cv2.resize(final,size)
    
    return submask

def mask1(image_path):
    img_path = image_path
    sizes = [224, 288, 352, 384, 480, 512, 736]
    masks = []
    for size in sizes:
        masks.append(get_submasks(img_path,size).astype("float32"))
    result = masks[0]
    for mask in masks:
        result = cv2.add(result, mask)
    return result

def mask2(image_path):
    img = cv2.imread(image_path)
    image= cv2.resize(img,(2048,1024))
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop1 = image[0:1024, 0:1024]
    crop2 = image[0:1024, 1024:2048]
    crop1 = preprocessing_fn(crop1)
    crop2 = preprocessing_fn(crop2)

    x1_tensor  = to_tensor(crop1)
    x1_tensor = torch.from_numpy(x1_tensor).to(DEVICE).unsqueeze(0)
    pred_mask1 = my_model_2(x1_tensor)
    pred_mask1 = pred_mask1.detach().squeeze().cpu().numpy()
    pred_mask1 = np.transpose(pred_mask1,(1,2,0))
    pred_mask1 = colour_code_segmentation(reverse_one_hot(pred_mask1), select_class_rgb_values)

    x2_tensor  = to_tensor(crop2)
    x2_tensor = torch.from_numpy(x2_tensor).to(DEVICE).unsqueeze(0)
    pred_mask2 = my_model_2(x2_tensor)
    pred_mask2 = pred_mask2.detach().squeeze().cpu().numpy()
    pred_mask2 = np.transpose(pred_mask2,(1,2,0))
    pred_mask2 = colour_code_segmentation(reverse_one_hot(pred_mask2), select_class_rgb_values)

    final = cv2.hconcat([pred_mask1, pred_mask2])

    size=img.shape[:2]
    size = size[::-1]
    #print(size)
    final = final.astype('float32')
    mask= cv2.resize(final,size)
    #plt.figure(figsize=(8,5))
    #plt.imshow(mask)
    #print(mask.shape)
    #cv2.imwrite("image6_mask.png", mask)
    return mask

def makeborder(im, bs): 
  bordersize = bs
  border = cv2.copyMakeBorder(
      im,
      top=bordersize,
      bottom=bordersize,
      left=bordersize,
      right=bordersize,
      borderType=cv2.BORDER_CONSTANT,
      value=[115, 147, 179]
  )
  return border

def get_mask(image_path, image_name, out_imgpath, SaveImages = True):
    msk1, msk2 = mask1(image_path), mask2(image_path)
    res = cv2.add(msk1, msk2)
    res1 = makeborder((np.clip(res, 0, 1)*255).astype("uint8"), 10)
    m1 = makeborder((np.clip(msk1, 0, 1)*255).astype("uint8"), 10)
    m2 = makeborder((np.clip(msk2, 0, 1)*255).astype("uint8"), 10)
    img1 = makeborder(cv2.imread(image_path), 10)
    v1 = cv2.vconcat([m1,m2])
    v2 = cv2.vconcat([img1,res1])
    coll = makeborder(cv2.hconcat([v1, v2]), 10)
    if SaveImages:
        cv2.imwrite(out_imgpath+image_name+"_mask.png", coll)
    return res