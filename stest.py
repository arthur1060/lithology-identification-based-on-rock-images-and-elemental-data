import argparse
from cfg import Cfg
from easydict import EasyDict as edict
import cv2, glob,os
from general import *

def ele_modification(image_id, poss_max, x, y):
    path = "D:\\data\\lithology-english-backup\\append-ele\\*\\" + image_id + ".JPG"
    path_real = glob.glob(path)[0]
    dirs = glob.glob(os.path.dirname(path_real) + "\\*.txt")
    for dd in dirs:
        if dd.split("\\")[-1][0:-4][0]!="5":
            ele_path = dd
    with open(ele_path,"r") as f1:
        data =f1.readlines()
    ele = data[0].split(",")
    mean = [0.0999, 573.460, 26.6674, 1.6448, 61.9799, 12.4258, 35.4500, 0.3955, 268.2282, 2.0163, 10.7092]
    var = [0.3572, 75.504, 31.2970, 2.4437, 95.5864, 10.1709, 18.7760, 1.2563, 114.3540, 8.2844, 18.01466]
    mean_p = [0.1292, 0.0974, 0.1245, 0.0877, 0.0820, 0.1145, 0.1454, 0.0743, 0.1351]
    var_p = [0.3258, 0.2933, 0.3294, 0.2753, 0.2721, 0.3137, 0.3401, 0.2493, 0.3352]
    element = torch.tensor((np.array(ele, dtype= np.float32) - mean) / var, dtype=torch.float32).unsqueeze(0).cuda()
    poss = torch.tensor((np.array(poss_max) - mean_p) / var_p, dtype=torch.float32)
    possbility = poss.unsqueeze(0).cuda()
    poss2= torch.cat([possbility,possbility],0)
    elem2 = torch.cat([element,element],0)
    model1 = Fusion2()
    if torch.cuda.is_available():
        model1 = model1.cuda()
        model1.load_state_dict(torch.load(".//fusionweight//bset.pkl"))
        model1.eval()
        with torch.no_grad():
            out = model1(elem2,poss2)
        _, lithology = torch.max(out, 1)

    return lithology[0].item()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default="D:\\Users\\heng_\\PycharmProjects\\pytorch_study\\YOLOv4-study\\coins",
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=3, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train-v6.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    cfg.update(args)

    return edict(cfg)

@torch.no_grad()
def stest( eval_model, val_loader,config, device,conf_thresh = 0.4 ,nms_thresh = 0.5 ):

    model = eval_model
    model = nn.DataParallel(model)

    nc = 9
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    model.eval()
    seen = 0

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for images, targets in val_loader:
        model_input = [[cv2.resize(img, (config.w, config.h))] for img in images]
        model_width_height = [[img.shape[1],img.shape[0]] for img in images]

        target = torch.zeros(len(targets),6)
        image_ids = []
        for s in range(config.batch):
            boxes = targets[s]["boxes"]
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            target[s,0] = torch.tensor(s)
            target[s, 1] = targets[s]["labels"]
            target[s, 2:6] = targets[s]["boxes"]
            image_ids.append(targets[s]["image_id"])

        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        target = target.to(device)
        output1 = model(model_input)

        output = [None] * config.batch
        for j in range(config.batch):
            out = []
            out.append(output1[0][j].unsqueeze(0))
            out.append(output1[1][j].unsqueeze(0))
            boxes = spost_processing(conf_thresh, nms_thresh, out)
            if len(boxes[0]) == 0:
                output[j] == None
            else:
                boxes_acc = []
                width, height = model_width_height[j]

                for i in range(len(boxes[0])):
                    temp_box = []

                    x1, y1 = boxes[0][i][0] * width, boxes[0][i][1] * height
                    x2, y2 = boxes[0][i][2] * width, boxes[0][i][3] * height
                    poss = boxes[0][i][-2]
                    lith = boxes[0][i][-1]
                    temp_box.append(x1)
                    temp_box.append(y1)
                    temp_box.append(x2)
                    temp_box.append(y2)
                    temp_box.append(poss)
                    temp_box.append(lith)
                    boxes_acc.append(temp_box)

                h = torch.tensor(boxes_acc, dtype=torch.float32).to(device)
                output[j] = h


        for si, pred in enumerate(output):

            labels = target[target[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = labels[:, 1:5]

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    clsetcls = cls == tcls_tensor
                    ti = torch.nonzero(clsetcls, as_tuple=False).view(-1)  # prediction indices

                    clsepred = cls == pred[:, 5]
                    pi = torch.nonzero(clsepred, as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious

                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        #print(ious)
                        # Append detections
                        "  (ious > iouv[0]).nonzero()  "
                        ious_v = ious > iouv[0]
                        for j in torch.nonzero(ious_v, as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    # print(stats[0])
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)

        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]

        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return mp, mr, map50, map,  maps


if __name__=="__main__":
    cfg = get_args(**Cfg)
    n_classes = 9
    weightfile = "weight\\Yolov4_epoch48.pth "
    device = torch.device("cuda")


    mp, mr, map50, map,  maps = stest(config=cfg, device=device)
    print(type(mp))
    print("final",mp, mr, map50, map,  maps)

    print(mp.shape)