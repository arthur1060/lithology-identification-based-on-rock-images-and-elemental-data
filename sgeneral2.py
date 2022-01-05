import time
import torch
import numpy as np

def do_detects(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()

    img = torch.autograd.Variable(img)

    t1 = time.time()
    #print("img", img.size())

    output = model(img)
    #print(output[0].size())
    #print(output[1].size())

    t2 = time.time()

    #print('-----------------------------------')
    #print('           Preprocess : %f' % (t1 - t0))
    #print('      Model Inference : %f' % (t2 - t1))
    #print('-----------------------------------')

    return spost_processing( conf_thresh, nms_thresh, output)



def spost_processing( conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]

    # [batch, num, num_classes]
    confs = output[1][:,:,0:9]

    object = output[1][:,:,9]

    #print("confs nnd confs_w",object.size(), confs.size())

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()
        object = object.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]
        confs_ch = confs[i, argwhere]
        object_ch = object[i, argwhere]

        bboxes = []
        confs = []

        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]
            confs_chch = confs_ch[cls_argwhere]
            object_chch = object_ch[cls_argwhere]

            keep = nms_cpus(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]

                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]
                confs_chchch = confs_chch[keep]
                object_chchch = np.expand_dims(object_chch[keep], axis = -1)

                real_conf = confs_chchch/object_chchch
                #print("confs_chchch",confs_chchch.shape,object_chchch.shape)
                #print(confs_chchch, object_chchch)
                #print(real_conf)

                for k in range(ll_box_array.shape[0]):

                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                                   real_conf[k], object_chchch[k][0], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    #print('-----------------------------------')
    #print('       max and argmax : %f' % (t2 - t1))
    #print('                  nms : %f' % (t3 - t2))
    #print('Post processing total : %f' % (t3 - t1))
    #print('-----------------------------------')

    return bboxes_batch


def nms_cpus(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)
