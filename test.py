import os
import numpy as np
import torch
import torch.cuda
import torch.utils.data
import logging
import copy
import skimage.transform
import scipy.stats
from tqdm import tqdm

from utils.proposal import generate_anchors, proposal_layer
from utils.sphere import viewport_alignment
from flownet2.models import FlowNet2
from dataset.dataset_VQA_ODV import DS_VQA_ODV, VQA_ODV_Transform
from models import VP_net, VQ_net


def main(log_dir, batch_size, num_workers, flownet_ckpt, test_start_frame, test_interval):
    arguments = copy.deepcopy(locals())

    if not torch.cuda.is_available():
        raise RuntimeError('At least 1 GPU is needed by FlowNet2.')
    device_main = torch.device('cuda:0')

    # For viewport alignment on 8K frame, more than 6 GB GPU memory is needed,
    # and thus it needs a different GPU device or fallback to CPU
    if torch.cuda.device_count() > 1:
        device_alignment = torch.device('cuda:1')
    else:
        device_alignment = torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    logger = logging.getLogger("test")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logger.info("%s", repr(arguments))

    bandwidth = 128
    test_set = DS_VQA_ODV(root=os.path.join(log_dir, "VQA_ODV"), dataset_type='test', tr_te_file='tr_te_VQA_ODV.txt',
                          ds_list_file='VQA_ODV.txt', test_interval=test_interval, test_start_frame=test_start_frame,
                          transform=VQA_ODV_Transform(bandwidth=bandwidth, down_resolution=(1024, 2048), to_rgb=True))

    anchor_shape = (16, 16)
    anchors = torch.tensor(generate_anchors(np.array(anchor_shape)))

    # Gaussian center bias
    cb = np.load(os.path.join(log_dir, 'cb256.npy')).astype(np.float32)[np.newaxis, np.newaxis, ...]
    cb = torch.tensor(cb).to(device_main)
    # Mask for anchors
    anchor_mask = np.load(os.path.join(log_dir, 'anchor_mask.npy')).astype(np.int64)
    anchor_mask = torch.tensor(anchor_mask)

    vpnet = VP_net.Model()
    vpnet.to(device_main)
    vpnet.load_state_dict(torch.load(os.path.join(log_dir, 'vp_state.pkl')))
    logger.info("Successfully loaded VP-net pre-trained model.")

    vqnet = VQ_net.Model()
    vqnet.to(device_main)
    vqnet.load_state_dict(torch.load(os.path.join(log_dir, 'vq_state.pkl')))
    logger.info("Successfully loaded VQ-net pre-trained model.")

    class FlowNetParams:
        rgb_max = 255.0
        fp16 = False

    flownet = FlowNet2(args=FlowNetParams())
    flownet.to(device_main)

    if isinstance(flownet_ckpt, str):
        flownet_ckpt = torch.load(flownet_ckpt)
    flownet.load_state_dict(flownet_ckpt['state_dict'])
    logger.info("Successfully loaded FlowNet2 pre-trained model.")
    flownet.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=False, pin_memory=True, drop_last=False)

    pred = []
    targets = []

    vpnet.eval()
    vqnet.eval()

    for batch_idx, img_tuple in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            img_s2, img_original, img_down, img_gap_s2, gap_down, ref_original, target = img_tuple

            gap_down = gap_down.to(device_main)
            img_down = img_down.to(device_main)
            gap_down = gap_down.view((-1, *gap_down.shape[-3:]))
            img_down = img_down.view((-1, *img_down.shape[-3:]))

            # Optical flow
            flow = torch.stack((gap_down, img_down), dim=0).permute(1, 2, 0, 3, 4)
            flow = flownet(flow)
            flow = flow.cpu().numpy().transpose((2, 3, 1, 0))
            flow = skimage.transform.resize(flow, (bandwidth * 2, bandwidth * 2) + flow.shape[-2:], order=1,
                                            anti_aliasing=True, mode='reflect', preserve_range=True).astype(np.float32)
            flow_s2 = torch.tensor(flow.transpose((3, 2, 0, 1)))
            flow_s2 = flow_s2.to(device_main)

            # VP net
            img_s2 = img_s2.to(device_main)
            img_gap_s2 = img_gap_s2.to(device_main)
            img_s2 = img_s2.view((-1, *img_s2.shape[-3:]))
            img_gap_s2 = img_gap_s2.view((-1, *img_gap_s2.shape[-3:]))

            vp_hm_weight, vp_hm_offset, _ = vpnet(img_s2, flow_s2, cb)

            # Viewport softer NMS
            hm_after_nms, hm_weight = proposal_layer(vp_hm_weight, vp_hm_offset, 20, 7.5, anchors.to(vp_hm_offset),
                                                     mask=anchor_mask)

            # Viewport alignment
            hm_after_nms = hm_after_nms.to(device_alignment)

            img_original = img_original.to(device_alignment)
            img_original = img_original.view((-1, *img_original.shape[-3:]))
            img_viewport = viewport_alignment(img_original, hm_after_nms[:, 0], hm_after_nms[:, 1])
            del img_original
            img_viewport = img_viewport.to(device_main)

            ref_original = ref_original.to(device_alignment)
            ref_original = ref_original.view((-1, *ref_original.shape[-3:]))
            ref_viewport = viewport_alignment(ref_original, hm_after_nms[:, 0], hm_after_nms[:, 1])
            del ref_original
            ref_viewport = ref_viewport.to(device_main)

            # VQ net
            vq_score, _ = vqnet(img_viewport, ref_viewport - img_viewport)
            vq_score = vq_score.flatten()
            vq_score = (vq_score * hm_weight).sum(dim=0, keepdim=True)

            pred.append(float(vq_score))

            target = target.mean(dim=1).reshape((-1,))
            targets.append(target.numpy())

    pred = np.array(pred)
    targets = np.concatenate(targets, 0)
    video_cnt = len(test_set.cum_frame_num)
    pred = [pred[test_set.cum_frame_num_prev[i]:test_set.cum_frame_num[i]].mean() for i in range(video_cnt)]
    targets = [targets[test_set.cum_frame_num_prev[i]:test_set.cum_frame_num[i]].mean() for i in range(video_cnt)]
    np.savetxt(os.path.join(log_dir, 'test_pred_scores.txt'), np.array(pred))
    np.savetxt(os.path.join(log_dir, 'test_targets.txt'), np.array(targets))
    srocc, _ = scipy.stats.spearmanr(pred, targets)

    logger.info("SROCC:{:.4}".format(srocc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--flownet_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_start_frame", type=int, default=21)
    parser.add_argument("--test_interval", type=int, default=45)

    args = parser.parse_args()
    main(**args.__dict__)
