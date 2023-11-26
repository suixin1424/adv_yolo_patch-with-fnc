from __future__ import print_function
import sys
from typing import Optional

if len(sys.argv) < 4:
    print('Usage:')
    print('python eval.py datacfg cfgfile weight1 weight2 ...')
    exit()

import time
import torch
from load_data import *
from utils import *
from cfg import parse_cfg
from darknet import Darknet
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]


gpus          = data_options['gpus']  # e.g. 0,1,2,3
num_worker   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])

use_cuda      = True
seed          = 22222
eps           = 1e-5

# Test parameters
conf_thresh   = 0.5
nms_thresh    = 0.45
iou_thresh    = 0.5
patch_size = 300

###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
model.print_network()

init_width        = model.width
init_height       = model.height

kwargs = {'num_workers': num_worker, 'pin_memory': True} if use_cuda else {}

map_meter = MeanAveragePrecision(iou_thresholds=[iou_thresh],box_format='cxcywh')

test_loader = torch.utils.data.DataLoader(InriaDataset("/home/zhuyiming/dataset/INRIAPerson/Test/pos", 
                                                       "/home/zhuyiming/dataset/INRIAPerson/custom_ann_yolov2/Test", 
                                                       14, 
                                                       init_width,
                                                        shuffle=True),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_worker)


#patch_img = Image.open('/home/zhuyiming/ssd2/zhuyiming/yolov2-patch/adversarial-yolo/saved_patches/patchnew0.jpg').convert('RGB')
patch_img=cv2.imread('/home/zhuyiming/ssd2/zhuyiming/yolov2-patch/adversarial-yolo/patch_attack/white.png')
patch_img = cv2.cvtColor(np.asarray(patch_img), cv2.COLOR_BGR2RGB)  
patch_img = PIL.Image.fromarray(patch_img)
tf = transforms.Resize((patch_size,patch_size))
patch_img = tf(patch_img)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)
adv_patch = adv_patch_cpu.cuda()

patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()
#print(adv_patch)

def test():
    model.eval()
    num_classes = model.num_classes
    anchors     = model.anchors
    num_anchors = model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    for i_batch, (img_batch, lab_batch, lens) in tqdm(enumerate(test_loader)):

        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        x = lab_batch[:, :, 1]
        y = lab_batch[:, :, 2]
        w = lab_batch[:, :, 3]
        h = lab_batch[:, :, 4]
        lab_adv = lab_batch.clone()
        lab_adv[:,:,1] = x + w/2
        lab_adv[:,:,2] = y + h/2

        adv_batch_t = patch_transformer(adv_patch, lab_adv, init_width, do_rotate=False, rand_loc=True)

        p_img_batch = patch_applier(img_batch, adv_batch_t)
        p_img_batch = p_img_batch.cuda()


        output = model(p_img_batch).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            local_target = []
            unpair_target = []
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            label = lab_batch[i,0:lens[i],:]
            total += lens[i]
            for j in range(len(boxes)):
                if boxes[j][4] > conf_thresh and boxes[j][6] == 0:
                    proposals = proposals+1
            for k in range(lens[i]):
                box_gt = [label[k][1]+label[k][3]/2, label[k][2]+label[k][4]/2, label[k][3], label[k][4]]
                
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == 0:
                    correct = correct+1
                    local_target.append(box_gt)
                else:
                    unpair_target.append(box_gt)
            local_target = local_target+unpair_target
            box_tensor = torch.Tensor(boxes).cuda()
            if box_tensor.shape[0]==0:
                continue
            t = box_tensor[box_tensor[:, 6] == 0]
            box_cal = t[:, :4]
            label_cal = t[:, 6].to(dtype=torch.int)
            scores = t[:, 4]
            
            local_target = torch.Tensor(local_target).cuda()
            target_labels = torch.zeros(local_target.shape[0]).to(dtype=torch.int).cuda()
            map_meter.update(
                [dict(boxes=box_cal, scores=scores, labels=label_cal)],
                [dict(boxes=local_target, labels=target_labels)],
            )
    results: dict[str, torch.Tensor] = map_meter.compute()
    mean_ap = results["map_50"].item() * 100.0
    logging("map50: %f" % mean_ap)   


    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))   
    #print(total)
    #print(proposals)
    #print(correct)

        #print(output)

for i in range(3, len(sys.argv)):
    weightfile = sys.argv[i]
    model.load_weights(weightfile)
    model = model.cuda()
    logging('evaluating ... %s' % (weightfile))
    test()
