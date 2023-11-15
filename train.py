import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import matplotlib.pyplot as plt
from datasets.preprocess import write_cam, write_pfm, save_pfm
import logging
import cv2
from logger import setup_logger

cudnn.benchmark = True

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']="0" # GPU set


parser = argparse.ArgumentParser(description='A PyTorch Implementation of EG-MVSNet')
parser.add_argument('--mode', default='', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='', help='select model', choices=['mvsnet', 'rmvsnet', 'rednet', 'egnet'])
parser.add_argument('--dataset', default='common', help='select dataset')

parser.add_argument('--trainpath', default = '.../train', help='train datapath')
parser.add_argument('--testpath', default = '.../test', help='test datapath')

parser.add_argument('--loadckpt', default="", help='load a specific checkpoint')
# parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='.../EG-MVSNet-main/outputs/xxx', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--fext', type=str, default='.png', help='Type of images.')
# input parameters
parser.add_argument('--view_num', type=int, default=5, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--numdepth', type=int, default=94, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=384, help='Maximum image height')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=0.5, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=2, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True, help='Let image size to fit the network, including scaling and cropping')

# network architecture
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, "train", args.view_num, args, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, "test", args.view_num, args, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False)


# Use Gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model, optimizer
if args.model == 'mvsnet':
    from models.mvsnet import MVSNet, mvsnet_loss
    model = MVSNet(refine=False)
if args.model == 'rmvsnet':
    from models.rmvsnet import RMVSNet, mvsnet_loss
    model = RMVSNet()
if args.model == 'rednet':
    from models.rednet import REDNet, mvsnet_loss
    model = REDNet()
if args.model == 'egnet':
    from models.egnet import EGNet, mvsnet_loss
    model = EGNet()


if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)


model.cuda()
model_loss = mvsnet_loss
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, weight_decay=args.wd)
torch.autograd.set_detect_anomaly(True)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    logger_wirter = logging.getLogger("EG-MVSNet.train")
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
                logger_wirter.info(
                    'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}, train_result = {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time, scalar_outputs))
            del scalar_outputs, image_outputs

        # testing
        meters = dict()
        meters['abs_depth_error'] = []
        meters['thres1m_error'] = []
        meters['thres0_6m_error'] = []
        meters['d1'] = []
        meters['d2'] = []
        meters['d3'] = []
        meters['rms'] = []
        meters['log_rms'] = []
        meters['abs_rel'] = []
        meters['sq_rel'] = []
        meters['silog'] = []
        meters['log10'] = []
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs, saved_outputs = test_sample(sample, meters, detailed_summary=do_summary)
            #if do_summary:
            save_scalars(logger, 'test', scalar_outputs, global_step)
            save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            logger_wirter.info('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}, {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time, scalar_outputs))
            del scalar_outputs, image_outputs, saved_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        abs_depth_error = avg_test_scalars.mean()["abs_depth_acc"]


        logger_wirter.info(
            ' Mean: abs depth error: {:.4f}, thres1m_error: {:.4f}, thres0_6m_error:{:.4f}, d1: {:.4f}, d2: {:.4f}, d3: {:.4f}, rms: {:.4f}, log_rms: {:.4f}, abs_rel: {:.4f}, sq_rel: {:.4f}, silog: {:.4f}, log10: {:.4f}'.format(
                sum(meters['abs_depth_error']) / len(meters['abs_depth_error']),
                sum(meters['thres1m_error']) / len(meters['thres1m_error']),
                sum(meters['thres0_6m_error']) / len(meters['thres0_6m_error']),
                sum(meters['d1']) / len(meters['d1']),
                sum(meters['d2']) / len(meters['d2']),
                sum(meters['d3']) / len(meters['d3']),
                sum(meters['rms']) / len(meters['rms']),
                sum(meters['log_rms']) / len(meters['log_rms']),
                sum(meters['abs_rel']) / len(meters['abs_rel']),
                sum(meters['sq_rel']) / len(meters['sq_rel']),
                sum(meters['silog']) / len(meters['silog']),
                sum(meters['log10']) / len(meters['log10'])))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}_{:.4f}.ckpt".format(args.logdir, epoch_idx, abs_depth_error))

        # gc.collect()


def test():
    logger_wirter = logging.getLogger("EG-MVSNet.test")
    # create output folder
    output_folder = os.path.join(args.logdir, 'depths_mvsnet')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    list_index = []
    f = open(output_folder + '/index.txt', "w")

    meters = dict()
    meters['abs_depth_error'] = []
    meters['thres1m_error'] = []
    meters['thres0_6m_error'] = []
    meters['d1'] = []
    meters['d2'] = []
    meters['d3'] = []
    meters['rms'] = []
    meters['log_rms'] = []
    meters['abs_rel'] = []
    meters['sq_rel'] = []
    meters['silog'] = []
    meters['log10'] = []

    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs, saved_outputs = test_sample(sample, meters, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        print("Iter {}/{}, time = {:3f}, test results = {}".format(batch_idx, len(TestImgLoader), time.time() - start_time, scalar_outputs))

        # save results
        depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))
        ref_image = np.squeeze(tensor2numpy(saved_outputs["outimage"]))
        ref_cam = np.squeeze(tensor2numpy((saved_outputs["outcam"])))
        out_location = np.squeeze(saved_outputs["outlocation"])
        edge_map = np.float32(np.squeeze(tensor2numpy(image_outputs["edge_map"])))

        ##  aerial dataset
        vid = saved_outputs["out_view"][0]
        name = saved_outputs["out_name"][0]

        # paths
        output_folder2 = output_folder + ('/%s/' % vid)
        if not os.path.exists(output_folder2):
            os.mkdir(output_folder2)

        if not os.path.exists(output_folder2+'/color/'):
            os.mkdir(output_folder2+'/color/')

        init_depth_map_path = output_folder2 + ('%s_init.pfm' % name)
        prob_map_path = output_folder2 + ('%s_prob.pfm' % name)
        out_ref_image_path = output_folder2 + ('%s.jpg' % name)
        out_ref_cam_path = output_folder2 + ('%s.txt' % name)
        edge_map_path = output_folder2 + ('%s.png' % name)

        if vid not in list_index:
            # if (list_index.index(out_index)==-1):
            out_location[3] = str(args.max_w)
            out_location[4] = str(args.max_h)
            list_index.append(vid)
            for word in out_location:
                f.write(str(word) + ' ')
            f.write('\n')

        # save output
        save_pfm(init_depth_map_path, depth_est)
        save_pfm(prob_map_path, prob)
        plt.imsave(out_ref_image_path, ref_image, format='jpg')
        write_cam(out_ref_cam_path, ref_cam, out_location)
        # edge map save
        edge_map = np.expand_dims(edge_map, axis=0)
        refined_edge = np.moveaxis(edge_map, 0, 2)
        refined_edge2 = np.array(refined_edge)
        refined_edge2 = cv2.normalize(refined_edge, refined_edge2, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(edge_map_path, refined_edge2)

        size1 = len(depth_est)
        size2 = len(depth_est[1])
        e = np.ones((size1, size2), dtype=np.float32)
        out_init_depth_image = e * 36000 - depth_est
        plt.imsave(output_folder2 + ('/color/%s_init.png' % name), out_init_depth_image, format='png')
        plt.imsave(output_folder2 + ('/color/%s_prob.png' % name), prob, format='png')

        del scalar_outputs, image_outputs, saved_outputs

    logger_wirter.info("final, time = {:3f}, test results = {}".format(time.time() - start_time, avg_test_scalars.mean()))
    logger_wirter.info(
        ' Mean: abs depth error: {:.4f}, thres1m_error: {:.4f}, thres0_6m_error:{:.4f}, d1: {:.4f}, d2: {:.4f}, d3: {:.4f}, rms: {:.4f}, log_rms: {:.4f}, abs_rel: {:.4f}, sq_rel: {:.4f}, silog: {:.4f}, log10: {:.4f}'.format(
            sum(meters['abs_depth_error']) / len(meters['abs_depth_error']),
            sum(meters['thres1m_error']) / len(meters['thres1m_error']),
            sum(meters['thres0_6m_error']) / len(meters['thres0_6m_error']),
            sum(meters['d1']) / len(meters['d1']),
            sum(meters['d2']) / len(meters['d2']),
            sum(meters['d3']) / len(meters['d3']),
            sum(meters['rms']) / len(meters['rms']),
            sum(meters['log_rms']) / len(meters['log_rms']),
            sum(meters['abs_rel']) / len(meters['abs_rel']),
            sum(meters['sq_rel']) / len(meters['sq_rel']),
            sum(meters['silog']) / len(meters['silog']),
            sum(meters['log10']) / len(meters['log10'])))

def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    coarse_est = outputs['coarse_depth']
    edge_est = outputs['edge_map']
    depth_est = outputs["depth"]

    # upsample depth
    depth_est = torch.unsqueeze(depth_est, dim=1)  
    batch, img_height, img_width = depth_gt.shape[0], depth_gt.shape[1], depth_gt.shape[2]
    depth_est = F.interpolate(depth_est, [img_height, img_width], mode='bilinear')
    depth_est = torch.squeeze(depth_est, dim=1)

    coarse_est = torch.unsqueeze(coarse_est, dim=1)
    coarse_est = F.interpolate(coarse_est, [img_height, img_width], mode='bilinear')
    coarse_est = torch.squeeze(coarse_est, dim=1)

    edge_est = torch.unsqueeze(edge_est, dim=1)
    edge_est = F.interpolate(edge_est, [img_height, img_width], mode='bilinear')
    edge_est = torch.squeeze(edge_est, dim=1)

    loss = model_loss(depth_est, coarse_est, edge_est, depth_gt, mask)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)  # <10
        scalar_outputs["thres1m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)
        scalar_outputs["thres0.6m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 0.6)  # 0.6
        # scalar_outputs["thres3interval_error"] = Inter_metrics(depth_est, depth_gt, depth_interval, mask > 0.5, 3)
        pred = depth_est.cpu().detach().numpy()
        gt = depth_gt.cpu().detach().numpy()
        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()
        # d1 = (thresh < 0.5).mean()
        # d2 = (thresh < 0.5 ** 2).mean()
        # d3 = (thresh < 0.5 ** 3).mean()

        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)

        d1 = torch.tensor(d1)
        d2 = torch.tensor(d2)
        d3 = torch.tensor(d3)
        rms = torch.tensor(rms)
        log_rms = torch.tensor(log_rms)
        abs_rel = torch.tensor(abs_rel)
        sq_rel = torch.tensor(sq_rel)
        silog = torch.tensor(silog)
        log10 = torch.tensor(log10)
        scalar_outputs["d1"] = d1
        scalar_outputs["d2"] = d2
        scalar_outputs["d3"] = d3
        scalar_outputs["rms"] = rms
        scalar_outputs["log_rms"] = log_rms
        scalar_outputs["abs_rel"] = abs_rel
        scalar_outputs["sq_rel"] = sq_rel
        scalar_outputs["silog"] = silog
        scalar_outputs["log10"] = log10

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs



@make_nograd_func
def test_sample(sample, meters, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    coarse_est = outputs['coarse_depth']
    edge_est = outputs['edge_map']
    depth_est = outputs["depth"]
    photometric_confidence = outputs["photometric_confidence"]

    # upsample depth
    depth_est = torch.unsqueeze(depth_est, dim=1)  # 在第2个维度上扩展
    batch, img_height, img_width = depth_gt.shape[0], depth_gt.shape[1], depth_gt.shape[2]
    depth_est = F.interpolate(depth_est, [img_height, img_width], mode='bilinear')
    depth_est = torch.squeeze(depth_est, dim=1)

    coarse_est = torch.unsqueeze(coarse_est, dim=1)
    coarse_est = F.interpolate(coarse_est, [img_height, img_width], mode='bilinear')
    coarse_est = torch.squeeze(coarse_est, dim=1)

    edge_est = torch.unsqueeze(edge_est, dim=1)
    edge_est = F.interpolate(edge_est, [img_height, img_width], mode='bilinear')
    edge_est = torch.squeeze(edge_est, dim=1)

    loss = model_loss(depth_est, coarse_est, edge_est, depth_gt, mask)

    # edge_est = torch.unsqueeze(edge_est, dim=1)
    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est,
                     "photometric_confidence": photometric_confidence,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"],
                     "edge_map": edge_est}
    saved_outputs = {"outimage": sample["outimage"],
                     "outcam": sample["outcam"],
                     "outlocation": sample["outlocation"],
                     "out_view": sample["out_view"],
                     "out_name": sample["out_name"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_acc"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)  # <10
    scalar_outputs["1m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)   # 0.1m = 1interval
    scalar_outputs["0.6m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 0.6)   #0.6
    # scalar_outputs["3interval_acc"] = Inter_metrics(depth_est, depth_gt, depth_interval, mask > 0.5, 3)
    pred = depth_est.cpu().detach().numpy()
    gt = depth_gt.cpu().detach().numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    # d1 = (thresh < 0.5).mean()
    # d2 = (thresh < 0.5 ** 2).mean()
    # d3 = (thresh < 0.5 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    if torch.isnan(scalar_outputs["abs_depth_acc"]).all() == False:
        meters['abs_depth_error'].append(scalar_outputs["abs_depth_acc"])
        meters['thres1m_error'].append(scalar_outputs["1m_acc"])
        meters['thres0_6m_error'].append(scalar_outputs["0.6m_acc"])
        meters['d1'].append(d1)
        meters['d2'].append(d2)
        meters['d3'].append(d3)
        meters['rms'].append(rms)
        meters['log_rms'].append(log_rms)
        meters['abs_rel'].append(abs_rel)
        meters['sq_rel'].append(sq_rel)
        meters['silog'].append(silog)
        meters['log10'].append(log10)

        d1 = torch.tensor(d1)
        d2 = torch.tensor(d2)
        d3 = torch.tensor(d3)
        rms = torch.tensor(rms)
        log_rms = torch.tensor(log_rms)
        abs_rel = torch.tensor(abs_rel)
        sq_rel = torch.tensor(sq_rel)
        silog = torch.tensor(silog)
        log10 = torch.tensor(log10)
        scalar_outputs["d1"] = d1
        scalar_outputs["d2"] = d2
        scalar_outputs["d3"] = d3
        scalar_outputs["rms"] = rms
        scalar_outputs["log_rms"] = log_rms
        scalar_outputs["abs_rel"] = abs_rel
        scalar_outputs["sq_rel"] = sq_rel
        scalar_outputs["silog"] = silog
        scalar_outputs["log10"] = log10

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, saved_outputs



def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        # 创建日志，并将相关配置和GPU信息写入
        logger1 = setup_logger("EG-MVSNet", args.logdir, prefix="train")
        logger1.info(args)
        train()
    elif args.mode == "test":
        logger1 = setup_logger("EG-MVSNet", args.logdir, prefix="test")
        logger1.info(args)
        test()
    elif args.mode == "profile":
        profile()
