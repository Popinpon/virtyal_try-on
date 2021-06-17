import time
#from options.test_options import TestOptions
#from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator
from models.afwm import AFWM
import torch.nn as nn
import os
if os.name=='posix':#linux or Mac
    import pyfakewebcam
if os.name =='nt':#Windows
    import pyvirtualcam
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image


def load_checkpoint(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]

    model.load_state_dict(checkpoint_new)
def background_seg(rimage):
    with torch.no_grad():
        output=segmentation(rimage)['out'][0]
    predict=output.argmax(0)
# plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(predict.byte().cpu().numpy()).resize((192,256))
    seg=np.array(r.convert('RGB'))
    #print(seg==13)
    mask=cv2.inRange(seg,(15,15,15),(15,15,15))/255
    mask=np.stack([mask]*3, axis=2)
    seg= resize*mask
    #real_background=rimage.byte().cpu().numpy()*mask
    background=np.zeros(resize.size)
    background=background[:,:,0:3]=[241,240,238]*(1-mask)
    seg=(background+seg).astype(np.uint8)
    
    real_image = transform(seg).unsqueeze(0).cuda()

    return real_image#,mask


cap = cv2.VideoCapture(0)
if os.name=='nt':
        cam=pyvirtualcam.Camera(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=30)
  
segmentation=models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)#torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)

#opt = TestOptions().parse()

segmentation.eval()
segmentation.cuda()

# start_epoch, epoch_iter = 1, 0

# data_loader = CreateDataLoader(opt)

# dataset = data_loader.load_data()
# dataset_size = len(data_loader)
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

clothes=[]
cloth_id=["014396","010567","003434","000005","000003"]
for id in cloth_id:
    cloth=transform(Image.open("dataset/test_clothes/"+id+"_1.jpg")).unsqueeze(0)
    edge=transforms.functional.to_tensor(Image.open("dataset/test_edge/"+id+"_1.jpg").convert("L")).unsqueeze(0)
    clothes.append((cloth,edge))
input_cloth,input_edge = clothes[0]
print(input_cloth.shape,input_edge.shape)
#cloth2=transform(Image.open("dataset/test_img/010567_1.jpg")).unsqueeze(0)
#cloth3=transform(Image.open("dataset/test_img/003434_1.jpg")).unsqueeze(0)


#cloth_id="014396"
#input_edge=transforms.functional.to_tensor(Image.open("dataset/test_edge/"+cloth_id+"_1.jpg")).unsqueeze(0)
#input_cloth=transform(Image.open("dataset/test_clothes/"+cloth_id+"_1.jpg")).unsqueeze(0)


warp_model = AFWM(3)
#print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, "checkpoints/PFAFN/warp_model_final.pth")

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
#print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, 'checkpoints/PFAFN/gen_model_final.pth')

#total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
#step_per_batch = dataset_size / opt.batchSize
print("test")



#for i, data in enumerate(dataset, start=epoch_iter):
t=0
is_whiteback=False
clothes_num=0
while(1):
    key = cv2.waitKey(1)
    if key == ord('c'):
        
        clothes_num=(clothes_num+1)%len(clothes)
        
        input_cloth,input_edge = clothes[clothes_num]

    if key == ord('b') :
        if  (not is_whiteback):
            is_whiteback=True
        else:
            is_whiteback=False
    if key == ord('z'):
        model = None
    if key == ord('q'):
        break


    t+=1 
    iter_start_time = time.time()
    #total_steps += opt.batchSize
    #epoch_iter += opt.batchSize
    #frame=cv2.imread("dataset/test_img/015794_0.jpg")
    ret,frame=cap.read()
    #print(frame.shape)
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    center = frame.shape
    #print(center)
    scale=center[0]/256.
    #print(scale)
    w=192*scale
    
    x = center[1]/2 - w/2
    y = center[0]
    crop_img = rgb[0:center[1], int(x):int(x+w)]
    #print(w)
    resize=cv2.resize(crop_img,(192,256))
    
    real_image = transform(resize).unsqueeze(0).cuda()
    #print(real_image.shape)
    
   


    
    ##edge is extracted from the clothes image with the built-in function in python
    
    if is_whiteback:
        real_image= background_seg(real_image)

    edge = torch.FloatTensor((input_edge.detach().numpy() > 0.5).astype(np.int))
    cloth = input_cloth * edge        
    rimage=real_image.cuda()
    cloth=cloth.cuda()

 
    flow_out = warp_model(rimage, cloth)
    warped_cloth, last_flow, = flow_out
    warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
    end=time.time()
    #print(int(1/(end-iter_start_time)))
    #b= clothes.cuda()
    
    c = p_tryon
    #print(torch.cat((c[0], cloth[0]),2).shape)
    cv_img=(torch.cat((c[0], cloth[0],real_image[0]),2) .permute(1,2,0).detach().cpu().numpy()+1)/2
    #show_cloth=(.permute(1,2,0).detach().cpu().numpy()+1)/2
    #cv_img=seg
    image=(cv_img*255).astype(np.uint8)
    #rgb=seg.astype(np.uint8)
    #print(seg.shape,resize.shape)
    h=int(round(640*image.shape[0]/image.shape[1],0))
    image=cv2.resize(image,(640, h))
    padding=int((480-h)/2)
    frame= cv2.copyMakeBorder(image,padding ,padding, 0, 0, cv2.BORDER_CONSTANT)
    if os.name=='nt':
                    cam.send(frame)
    bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


    cv2.imshow('virtual try on',bgr)
    



