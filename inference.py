import os

import sys

from options.train_options import TrainOptions
from torch.utils.data import DataLoader
import importlib
from torch.utils.data.dataset import Subset
#from torch.utils.data.distributed import DistributedSampler
from data.custom_dataset_data_loader import CreateDataset
import torchvision.models as models
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F

if os.name=='posix':#linux or Mac
    import pyfakewebcam
if os.name =='nt':#Windows
    import pyvirtualcam
import cv2

# import models.networks
# importlib.reload(models.networks)
from models.networks import ResUnetGenerator
from models.networks import save_checkpoint, load_checkpoint_part_parallel, load_checkpoint_parallel

# import models.afwm
# importlib.reload(models.afwm)
from models.afwm import AFWM 
import yaml
import os.path as osp
import glob

import numpy as np

def concat_prev(prev, now,save_frame_n):
    if type(prev) == list:
        return [concat_prev(p, n,save_frame_n) for p, n in zip(prev, now)]
    if prev is None:
        prev = now.unsqueeze(1).repeat(1, save_frame_n, 1,1, 1)#first generaited frame repeat
    else:
        prev = torch.cat([prev[:, 1:], now.unsqueeze(1)], dim=1)#idx 0 is the oldest frame 
    return prev.detach()
frame_list=[]
def denormalize(tensor):
    return (tensor+1)/2
from torchvision import transforms


def half_crop(image):

    c,h,w=image.shape
    if h==256:
        return transforms.functional.crop(image,0,0,128,192)
    return image

transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(half_crop),
    transforms.Resize([128,192]),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(half_crop),
    transforms.Resize([128,192]),
])

def get_data_t(data,  t):
    if data is None: return None
    if type(data)==list:
        return data
    if type(data) == dict:
        return {k:get_data_t(v, t) for k,v in data.items()}
    if len(data.shape)==5:
        return data[:,t] 
    else:
        return data


dict_param={"args":""}
args=f"--resize_or_crop upper_half --verbose --tf_log --batchSize 5 --num_gpus 2 --label_nc 14  --PBAFN_warp_checkpoint 'checkpoints/video/PBAFN/stage1/baseline2/PBAFN_warp_epoch_022.pth' \
--PBAFN_gen_checkpoint 'checkpoints/video/PBAFN/half_PBAFN_e2e/PBAFN_gen_epoch_101.pth' \
--max_t_step 1 --n_frames_total 1 --scene_mode zoom "


args=args.split()
opt = TrainOptions().parse()


# dataset = CreateDataset(opt)
# print(opt.PBAFN_warp_checkpoint)

# val_loader = DataLoader(dataset['val'], batch_size=opt.batchSize,num_workers=3,shuffle=False)#, pin_memory=True)
# dataset_size = len(val_loader)
# print(dataset_size)
# itr=iter(val_loader)
# data=next(itr)
import torch.nn as nn


def load_model(model_types,warp_paths,gen_paths):
    model_dict={}
    for i,  model_type in enumerate(model_types):
        warp_path=warp_paths[i]
        gen_path=gen_paths[i]

        condition_ch=3
        gen_ch=3+3+1
        prev=None
        if model_type=="temporal":
            condition_ch+=(3+1)*(2-1)
            if not is_viton_gen:
                gen_ch+=(3+1)*(2-1)
            prev=[]
        PF_warp = AFWM( opt,condition_ch)

        PF_gen =  ResUnetGenerator(gen_ch, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)

        load_checkpoint_parallel(PF_warp,warp_path)
        load_checkpoint_parallel(PF_gen,gen_path)
        PF_warp.eval()
        PF_warp.cuda()
        PF_gen.eval()
        PF_gen.cuda()
        p,mb=calc_param(PF_warp)
        p2,mb2=calc_param(PF_gen)
        print(p+p2,'model size: {:.3f}MB'.format(mb+mb2))

        model_dict.update( {model_type:{"model":[PF_warp,PF_gen],"path":[warp_path,gen_path],"prev":prev}})

    return model_dict







def matting_load_video(person_id):
#     movie_id="084"#"084"#"028"#029"
# import glob

# from PIL import Image
# person_id=["022","023","084","025","028"]
    clothes=[]
    edges=[]
    person_video=[]
    min_num=200
    for id in person_id:
        frames=[transform(Image.open( path )).unsqueeze(0) for path in glob.glob("../extract/crop_img/"+id+"/*.jpg")]
        if min_num >len(frames):
            frames.append(frames[-1]*(min-frames))
        person_video.append(torch.cat(frames[:200],0).unsqueeze(0))

    person_video=torch.cat(person_video,0)
    return person_video

def load_viton_cloth(cloth_id):
    test_dir="dataset"
    # all_paths=sorted(glob.glob(test_dir+"test_cloths/*"))
    clothes=[]
    edges=[]
    # paths=all_paths[19:21]
    for id in cloth_id:
        filename=f"{id}_1.jpg"
        path=osp.join(test_dir,"test_edge",filename)
        print(path)
        edge=mask_transform(Image.open(path).convert("L")).unsqueeze(0)

        cloth=(transform(Image.open(osp.join(test_dir,"test_clothes",filename)))).unsqueeze(0)*edge

    # for path in paths:
    #     filename=osp.basename(path)
    #     edge=transforms.functional.to_tensor(Image.open(osp.join(test_dir,"cloth-mask",filename)).convert("L")).unsqueeze(0)
    #     cloth=(transform(Image.open(path))).unsqueeze(0)*edge

        
        edges.append(edge[:,:,:128,:])
        clothes.append((cloth[:,:,:128,:]))
        # return torch.cat(clothes,0) ,torch.cat(edges,0) 

    cloth_b=torch.cat(clothes,0)
    edge_b=torch.cat(edges,0)
    return cloth_b,edge_b
def calc_param(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in  model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    param_all=sum(p.numel() for p in model.parameters())
    return param_all,size_all_mb
def image_forward(model_list,real_image,cloth,edge,prev_list=None):
        warp_model,gen_model=model_list
        warp_input_l=[real_image.cuda()]
        b,c,h,w=cloth.shape
        
        if prev_list is not None:
                
                prev_warp_clothes,prev_warp_edges=prev_list
                prev_warp=prev_warp_clothes.contiguous().view(b,-1,h,w)
                prev_edge=prev_warp_edges.contiguous().view(b,-1,h,w)
                prev_l=[prev_warp.cuda(),prev_edge.cuda()]
                warp_input_l.extend(prev_l)
        # print(real_image.shape,cloth.shape,prev_warp.shape,prev_warp.shape,edge.shape)
        warp_input=torch.cat(warp_input_l,1)

        PF_flow=warp_model(warp_input, cloth.cuda(),edge.cuda())
        PF_warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, PF_x_edge_all, delta_x_all, delta_y_all = PF_flow
        PF_warped_prod_edge = PF_x_edge_all[4]
        
        
        gen_cat=[real_image.cuda(), PF_warped_cloth.cuda(), PF_warped_prod_edge.cuda()]
        if prev_list is not None and  not is_viton_gen :

            gen_cat.extend(prev_l)
        gen_inputs = torch.cat(gen_cat, 1)    
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * PF_warped_prod_edge
        p_tryon = PF_warped_cloth * m_composite + p_rendered * (1 - m_composite)

        
        if prev_list is not None:
                new_prev_list=concat_prev(prev_list,[PF_warped_cloth,PF_warped_prod_edge],opt.n_frames_G-1)
                return p_tryon,PF_warped_cloth,PF_warped_prod_edge,new_prev_list
        return p_tryon,PF_warped_cloth ,PF_warped_prod_edge,None

def denormalize(tensor):
    return (tensor+1)/2




frame_transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize([128,192]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_test_cloth(cloth_names):

    test_dir=f"zoom_video/test/_resized"
    # cloth_id=["000192","014396","010567","003434","019119","006026"]

    clothes=[]
    edges=[]

    # paths=all_paths[19:21]
    for cloth_name in cloth_names:
        path=glob.glob(osp.join(test_dir,"edge",cloth_name+"_edge*",))[0]

        print(path)

        edge=mask_transform(Image.open(path).convert("L")).unsqueeze(0)
        path=glob.glob(osp.join(test_dir,"cloth",cloth_name+".*",))[0]
        cloth=(transform(Image.open(path))).unsqueeze(0)*edge

    # for path in paths:
    #     filename=osp.basename(path)
    #     edge=transforms.functional.to_tensor(Image.open(osp.join(test_dir,"cloth-mask",filename)).convert("L")).unsqueeze(0)
    #     cloth=(transform(Image.open(path))).unsqueeze(0)*edge

        print(cloth_name)
        edges.append(edge[:,:,:128,:])
        clothes.append((cloth[:,:,:128,:]))

    return torch.cat(clothes,0),torch.cat(edges,0) #,torch.cat(edges,0) 



def load_test_video(video_names):
# video_names=["test_video2.mp4","zoom_in2.mp4","zoom_in.mp4"]
#zoom_test_moving
    videos=[]
    max_vlen=0
    for name in video_names:
    #cap_video=torchvision.io.read_video(video_name[0])[0].permute(0,3,1,2)
            vid = torchvision.io.VideoReader(f"zoom_video/test/{name}", "video")
            vid.seek(0)
            frames=[frame_transform(frame['data']).unsqueeze(0).unsqueeze(0).to(float) for frame in vid]
            if max_vlen<len(frames):
                    max_vlen=len(frames)
            videos.append(frames)
    video=[]
    for i,v in enumerate(videos):
            vlen=len(v)

            print(len(v),v[0].shape)
            if vlen<max_vlen:
                    last_frames=[v[-1]]*(max_vlen-vlen)
                    v.extend(last_frames)
            t=torch.cat(v,0)
            video.append(t)
    videos=torch.cat(video,1)
    return videos





is_whiteback=False

# cap = cv2.VideoCapture(video_name[0])

# tmp=data["frame"]



# video=crop_video


# if "temporal" in loaded_model_names:
#     if "vvt_base" in loaded_model_names:
#         # output_dir="test/video/temporal/lambda/1/vs_vvt/"
#         output_dir=test/experiment/vvt
#     elif "viton_base" in loaded_model_names:
#         output_dir="test/video/temporal/lambda/1/vs_viton/"
# if "vvt_base" in loaded_model_names and "viton" in loaded_model_names:
#     output_dir="test/video/baseline/image_video/real_video/frame8"
#     output_dir="test/video/VITONvsVVT/zoom"

#output_dir="test/video/0.01vs_baseline/image_video/matting/"
#output_dir="test/video/0.01vs_baseline/image_video/VITON/"
#output_dir="test/FID/test"

# init_warped_cloth_PF=torch.zeros(b,1,3,h,w).cuda()#gt_cloth,warped_cloth
# init_warped_edge_PF=torch.zeros(b,1,1,h,w).cuda()
# prev_list=[init_warped_cloth_PF,init_warped_edge_PF]
#PF_temporal_prev_list=[init_warped_cloth_PF,init_warped_edge_PF]
import time
t_list=[]
frame_list=[]

video_type="zoom"

c_i=0
cloth_id=["000002","000007","000192","010567","003434"]
# cloth_id=["014396","010567","003434","019119","006026"]#"000192",
cloth,edge=load_viton_cloth(cloth_id)
test_cloth_name=["t_uec"]
# cloth_source,edge_source=load_test_cloth(test_cloth_name)

# cloth_a=cloth_main[c_i:c_i+1]
# edge_a =cloth_edge_main[c_i:c_i+1]

loaded_model_names= [ "viton_base","vvt_base","temporal"]#,"viton_base" ]# temporal "vvt_base" "vvt_viton_base" PF_viton_base"
warp_paths=[]
gen_paths=[]

is_viton_gen=True
output_texts={}
for model_name in loaded_model_names:
    if model_name=="temporal":
        base_dir="checkpoints"
        # lambda_t=1
        date="01290524"
        dir=osp.join(base_dir,model_name,date)
        stage="warp"
        filename=f"PFAFN_{stage}_latest.pth"
        warp_paths.append(osp.join(dir,stage,filename))
        
        stage="gen"
        filename=f"PFAFN_{stage}_latest.pth"
        if is_viton_gen:
            gen_paths.append(viton_gen_path)
        else:
            gen_paths.append(osp.join(dir,stage,filename))#viton_gen_path)
        output_texts.update({model_name:["ours"]})
    
    if model_name=="vvt_base":
        base_dir="checkpoints/"
        date="01181148"
        stage="warp"
        dir=osp.join(base_dir,model_name,date)
        filename=f"PFAFN_{stage}_epoch_035.pth.pth"
        warp_paths.append(osp.join(dir,stage,filename))
        stage="gen"
        filename=f"PFAFN_{stage}_epoch_027.pth.pth"

        gen_paths.append(osp.join(dir,stage,filename))#viton_gen_path)
        viton_gen_path=osp.join(dir,stage,filename)

        output_texts.update({model_name:["PF-AFN","(video dataset)"]})
    if model_name=="viton_base":
        base_dir="checkpoints"
        
        stage="warp"
        dir=osp.join(base_dir,model_name)
        # filename=f"{stage}_model_final.pth"
        filename=f"PFAFN_{stage}_latest.pth"
        warp_paths.append(osp.join(dir,stage,filename))
        stage="gen"
        # filename=f"{stage}_model_final.pth"
        filename=f"PFAFN_{stage}_latest.pth"
        gen_paths.append(osp.join(dir,stage,filename))
        output_texts.update({model_name:["PF-AFN","(image dataset)"]})

print(warp_paths)

print(gen_paths)

model_dict=load_model(loaded_model_names,warp_paths,gen_paths)
loaded_model_names.pop(1)
model_dict.pop("vvt_base")


output_dir=f"test/experiment/{loaded_model_names[0]}"


# if video_type=="vvt":
#     frames=None
#     #frames=data["frame"][:b].permute(1,0,2,3,4)
# elif video_type=="matting": 
#     person_video=matting_load_video()

#     frames=person_video.permute(1,0,2,3,4)
# elif video_type=="zoom":
#     video_names=["uec_t_2.mp4"]#,#uec_t.mp4","sweater_right.mp4","shirt.mp4",uec_t.mp4,white.mp4]"zoom_in2.mp4","zoom_in.mp4"]
#     frames=load_test_video(video_names)
#     t,b,c,h,w=frames.shape
#     frames=torch.cat([frames]*cloth.shape[0],1)
#     cloth=cloth.repeat_interleave(b,0)
#     edge=edge.repeat_interleave(b,0)

b,c,h,w=cloth.shape
b=1
opt.n_frames_G=2
init_warped_cloth=torch.zeros(b,opt.n_frames_G-1,3,h,w).cuda()#gt_cloth,warped_cloth
init_warped_edge=torch.zeros(b,opt.n_frames_G-1,1,h,w).cuda()
prev_list=[init_warped_cloth,init_warped_edge]


for k in model_dict.keys():
  if model_dict[k]["prev"] is not None:
     model_dict[k]["prev"]=prev_list

# num=frames[0].shape[0]
# cloth=cloth[c_i:c_i+num]

    

def background_seg(rimage,model_name="DEEPLAB_PLUS"):
    with torch.no_grad():
        if model_name=="MODNET":
            _, _, predict = modnet(rimage, True)
            mask_tensor = predict.repeat(1, 3, 1, 1)
            mask=mask_tensor[0].data.cpu().numpy().transpose(1, 2, 0)  
        if model_name=="DEEPLAB_PLUS":
            output=segmentation(rimage)['out'][0]
            predict=output.argmax(0)
            mask=predict.data.cpu().numpy()
            mask=cv2.inRange(mask,(15),(15))/255
            mask=np.stack([mask]*3, axis=2)
    #with torch.no_grad():
    #    output=segmentation(rimage)['out'][0]
    #predict=output.argmax(0)
# plot the semantic segmentation predictions of 21 classes in each color
    # r = Image.fromarray(predict.byte().cpu().numpy()).resize((192,256))
    # seg=np.array(r.convert('RGB'))
    # #print(seg==13)
    # mask=cv2.inRange(seg,(15,15,15),(15,15,15))/255
    # mask=np.stack([mask]*3, axis=2)


    np_image=denormalize(rimage.squeeze().to('cpu').detach().numpy().transpose(1, 2, 0))*255#

    seg= np_image*mask
    #real_background=rimage.byte().cpu().numpy()*mask
    #print(mask.shape)
    background=np.zeros(rimage.shape)
    background=background[:,:,0:3]=np.array([241,240,238])*(1-mask)
    seg=(background+seg).astype(np.uint8)
    
    real_image = transform(seg).unsqueeze(0).cuda()

    return real_image#,mask


cap = cv2.VideoCapture(1)
if os.name=='nt':
        camera_w,camera_h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam=pyvirtualcam.Camera(width=camera_w, height=camera_h, fps=30)
  
segmentation=models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)#torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)

#opt = TestOptions().parse()

segmentation.eval()
segmentation.cuda()



segnames=["DEEPLAB_PLUS"]

t=0
seg_num=0
clothes_num=0
is_whiteback=True
is_half=True
is_only_person=True
text_add=True
model_idx=0
model_names=loaded_model_names
average_time=[]
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
while(1):
        key = cv2.waitKey(1)
        if key == ord('c'):
            
            clothes_num=(clothes_num+1)%len(cloth)
        if key == ord('f'):
            if is_only_person:
                is_only_person=False
            else:
                is_only_person=True
        if key == ord('m'):
            model_idx+=1
            if model_idx==0 or model_idx > len(loaded_model_names):
                model_names=loaded_model_names
                model_idx=0
            elif model_idx<=len(loaded_model_names):
                model_names=loaded_model_names[model_idx-1:model_idx]
            average_time=[]

            
        if key == ord('b') :

            seg_num=(seg_num+1)%(1+len(segnames))
            if seg_num<len(segnames):
                is_whiteback=True
            else:
                is_whiteback=False
        if key == ord('h'):
            if is_half:
                is_half=False
            else:
                is_half=True
        if key == ord('z'):
            model = None
        if key == ord('t'):
            if text_add:
                text_add=False
            else:
                text_add=True
        if key == ord('q'):
            break
        
        
        input_cloth,input_edge = cloth[clothes_num:clothes_num+1],edge[clothes_num:clothes_num+1]

        t+=1 
        iter_start_time = time.time()
        #total_steps += opt.batchSize
        #epoch_iter += opt.batchSize
        #frame=cv2.imread("dataset/test_img/015794_0.jpg")
        ret,frame=cap.read()


        #print(frame.shape)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        final_h=128
        final_w=192
        
        h,w,c=frame.shape
        center=[h,w]
        scale=center[0]/final_h
        #print(scale)
        # if final_w/final_h==3/4:
        #     h=center[0]
        #     w=center[1]
            
        # else:
        
        if final_w/final_h<1:
            w=center[0]*final_w/final_h
            h=center[0]
        else:
            w=center[1]
            h=center[1]*final_h/final_w
        
        x = center[1]/2 - w/2
        
        y = 0

        frame=Image.fromarray(frame)
        frame=transform(frame).unsqueeze(0)
        real_image = frame[:,y:int(y+h), int(x):int(x+w)]

        real_image=F.interpolate(real_image, size=[final_h,final_w])

        # real_image=cv2.resize(real_image,(final_w,final_h))
        #real_image = transform(real_image.numpy()).unsqueeze(0).cuda()

        ##edge is extracted from the clothes image with the built-in function in python
        real_image=real_image.cuda()
        # real_image=torch.cat([real_image]*len(cloth),0)
        input_cloth=input_cloth.cuda()
        input_edge=input_edge.cuda()
        b,c,h,w=real_image.shape

        if is_whiteback:
            real_image= background_seg(real_image,segnames[seg_num])

        if is_only_person:
            person_list=[]
            cloth_list=[]
        else:
            person_list=[real_image]
            cloth_list=[input_cloth]


        for model_name in model_names:
          prev=model_dict[model_name]["prev"]
          with torch.no_grad():
            p_tryon,warped_cloth,mask,prev,=image_forward(model_dict[model_name]["model"],real_image,input_cloth,input_edge,prev)
          model_dict[model_name]["prev"]=prev
          
        #   print(p_tryon.shape,real_image.shape,input_cloth.shape)
          if not is_only_person:
            cloth_list.extend([warped_cloth ] )
          person_list.extend([p_tryon])

        cat=torch.cat(cloth_list+person_list)
        img_row=len(person_list)
        result_frame   = torchvision.utils.make_grid(denormalize(cat),nrow=img_row).squeeze()
        # result_frame=denormalize(img.cpu())
        end=time.time()
        # average_time.append(end-iter_start_time)
        # if len(average_time)>100:
        #     average_time.pop(0)
        #     print(sum(average_time)/len(average_time))
        fps=str(int(1/(end-iter_start_time)))
        cv_img=result_frame.permute(1,2,0).detach().cpu().numpy()
        #show_cloth=(.permute(1,2,0).detach().cpu().numpy()+1)/2
        #cv_img=seg
        image=(cv_img*255).astype(np.uint8)
        #rgb=seg.astype(np.uint8)
        #print(seg.shape,resize.shape)

        h=int(round(640*image.shape[0]/image.shape[1],0))
        
        padding=int((480-h)/2)
        if padding>0:
            frame= cv2.copyMakeBorder(image,padding ,padding, 0, 0, cv2.BORDER_CONSTANT)
        else:
            w=int(round(480*image.shape[1]/image.shape[0],0))
            int((640-w)/2)
            frame= cv2.copyMakeBorder(image,0,0,padding ,padding, cv2.BORDER_CONSTANT)
        frame=cv2.resize(frame,(camera_w, camera_h))

        unit_w=int(camera_w/img_row)
        # frame = cv2.flip(frame, 1)
        if text_add:
            for i , model_name in enumerate(model_names[::-1]):
                text_ln=output_texts[model_name]
                for t_i ,text in enumerate(text_ln):
                    scale=1
                    font_size=1.4-t_i*0.3
                    (text_w, text_h), baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN, font_size,scale)
                    # print(text_w)
                    cv2.putText(frame,text,(camera_w-(int(unit_w/2) + unit_w*i + int(text_w/2)) ,50 +text_h*t_i),cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), scale, cv2.LINE_AA)
        cam_frame=frame
        
        if os.name=='nt':
                        cam.send(cam_frame)
                        cam.sleep_until_next_frame()
        bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.putText(bgr,'FPS:'+fps,(10,30),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)


        # print(bgr.shape)
        cv2.imshow('window',bgr)
        # frmae=cv2.resize(frame,(640, h))





# video_ext="mp4"
# hypara_ext="txt"


# # temporal "vvt_base" "vvt_viton_base" PF_viton_base"
# #output_dir="test/video/pretrain/half_all"

# for i,id in enumerate(cloth_id):

#     for n, name in enumerate(loaded_model_names):
#         for v_name in video_names:
#             dict_param[name]=model_dict[name]["path"] 
#             dict_param["video_typ"]=video_type
#             output_name=f"{loaded_model_names[n]}_{v_name}_{id}"
#             output_name=f"{output_name}.{video_ext}"
#             hypara_name=f"{output_name}.{hypara_ext}"
#             # print(frame_list[0].shape)
#             videos_result=torch.cat(frame_list,0)
#             # print(videos_result.shape)
#             os.makedirs(output_dir,exist_ok=True)
#             video_result=(videos_result[:,i*len(video_names)+n].permute(0,2,3,1)*255).to(torch.uint8).cpu()
#             print(output_dir,output_name)
#             torchvision.io.write_video(f"{output_dir}/{output_name}",video_result,fps=30,video_codec='h264')#TxHxWBxC

#     # for k in dict_param.keys():
#     #     dict_param[k]=eval(k)
        
#         with open(osp.join(output_dir,hypara_name),"w") as f :
#             yaml.dump(dict_param,f)