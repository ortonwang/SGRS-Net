import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import warnings

warnings.filterwarnings('ignore')
import argparse
import logging
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.mixmatch_util import mix_match_just_k1

from utils.util2 import  WeightEMA

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas_CT', help='Pancreas_CT,LA,BraTS2019')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='debug', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')  # 2
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')  # 4
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=6, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')
parser.add_argument('--dice_w', type=float, default=20., help='num of embeddings per class in memory bank')
parser.add_argument('--ce_w', type=float, default=0.2, help='num of embeddings per class in memory bank')
parser.add_argument('--tau', type=float, default=0.410, help='num of embeddings per class in memory bank')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 2
if args.dataset_name == "LA":
    # patch_size = (32, 32, 32)     #debug
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
    args.max_iteration = 15000
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    # patch_size = (64, 64, 64)     #debug
    args.root_path = args.root_path + 'data/Pancreas'
    args.max_samples = 62
    args.max_iteration = 15000
elif args.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    # patch_size = (48, 48, 48) #debug
    args.root_path = args.root_path + 'data/BraTS2019'
    args.max_samples = 250
    args.max_iteration = 60000
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
from utils.util2 import FullAugmentor
augmentor = FullAugmentor()

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model(ema=False)
    ema_model = create_model(ema=True)

    if args.dataset_name == "LA":
        db_train = LAHeart_no_read(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),with_idx=True)
    elif args.dataset_name == "Pancreas_CT":   # Pancreas_no_read # Pancreas
        db_train = Pancreas_no_read(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]),with_idx=True)
    elif args.dataset_name == "BraTS2019":
        db_train = BraTS2019_no_read(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                 RandomCrop(patch_size),
                                 ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ema_optimizer = WeightEMA(model, ema_model, alpha=0.99)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    # dice_loss = losses.Binary_dice_loss
    bce_loss = nn.BCELoss()
    iter_num = 0
    best_dice = 0
    ce_loss = CrossEntropyLoss()
    ce_loss2 = CrossEntropyLoss(reduction='none')
    ce_loss_soft = CrossEntropyLoss(label_smoothing=0.1)
    ce_loss_soft2 = CrossEntropyLoss(label_smoothing=args.ce_w,reduction='none')
    dice_loss = losses.DiceLoss(num_classes)
    # dice_loss_soft = losses.NoiseRobustDiceLoss()
    dice_loss_soft0 = losses.SoftDiceLoss(smooth=1.)
    dice_loss_soft1 = losses.SoftDiceLoss(smooth=args.dice_w)
    kl_distance = nn.KLDivLoss(reduction='none')
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    augu = torch.tensor(1).cuda()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image = volume_batch[:args.labeled_bs]#.cpu().numpy()
            l_label = label_batch[:args.labeled_bs]#.cpu().numpy()

            ul_image = volume_batch[args.labeled_bs:]
            X = list(zip(l_image, l_label))
            X_prime,U_prime, pseudo_label,X_cap = mix_match_just_k1(X,ul_image, eval_net=ema_model, K=1, T=0.5, alpha=0.75,
                                                          mixup_mode='_x', aug_factor=augu)
            model.train()

            U_data_m = torch.cat([torch.unsqueeze(U_prime[0][0], 0), torch.unsqueeze(U_prime[1][0], 0)],0)
            X_data = l_image.cuda()
            X_label = l_label.cuda().float()
            U_data = ul_image.cuda()
            U_data_m = U_data_m.cuda()
            U_data_pseudo = pseudo_label.cuda().float()

            U_data_pseudo2 = torch.cat((U_data_pseudo,U_data_pseudo), 0)
            X = torch.cat((X_data,U_data, U_data_m), 0)
            output_all= model(X)
            output_all_softmax = torch.softmax(output_all, dim=1)

            loss_seg_ce_lab, loss_seg_ce_unlab = 0, 0
            loss_seg_dice_lab, loss_seg_dice_unlab = 0, 0

            loss_seg_ce_lab += ce_loss(output_all[:args.labeled_bs], X_label.long())
            loss_seg_dice_lab += dice_loss(output_all_softmax[:args.labeled_bs], X_label.long().unsqueeze(1))
            supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)


            out_soft_u = output_all_softmax[args.labeled_bs:2*args.labeled_bs]
            out_soft_u_mix = output_all_softmax[2 * args.labeled_bs:]

            u_predict = torch.max(out_soft_u, 1, )[1]
            u_mix_predict = torch.max(out_soft_u_mix, 1, )[1]
            pp_outsoft_u = out_soft_u[:,0,:,:,:]
            pp_outsoft_mix = out_soft_u[:, 0, :, :, :]
            diff_pre = ((u_predict == 1) & (u_mix_predict == 0)).to(torch.int32) + ((u_predict == 0) & (u_mix_predict == 1)).to(torch.int32)
            consis_pre = ((u_predict == 0) & (u_mix_predict == 0)).to(torch.int32) + ((u_predict == 1) & (u_mix_predict == 1)).to(torch.int32)
            ignore_pre_u = (pp_outsoft_u > args.tau) & ( pp_outsoft_u < (1- args.tau )).to(torch.int32)
            ignore_pre_m = (pp_outsoft_mix > args.tau) & (pp_outsoft_mix < (1 - args.tau)).to(torch.int32)
            ignore_pre = ignore_pre_u + ignore_pre_m  # combine the double  \cup process
            ignore_pre = torch.clip(ignore_pre,0,1)

            consis_ignore = ignore_pre * consis_pre
            dif_ignore = ignore_pre * diff_pre
            diff_pre1 = (diff_pre- dif_ignore ).unsqueeze(1)#.repeat(1, 2, 1, 1, 1)
            consis_pre1 = (consis_pre - consis_ignore).unsqueeze(1)#.repeat(1, 2, 1, 1, 1)
            # ce part
            output_all_u = output_all[args.labeled_bs:args.labeled_bs*2]
            output_all_u_mix = output_all[args.labeled_bs*2:]

            loss_seg_ce_unlab_u_consis = torch.mean(ce_loss2(output_all_u, U_data_pseudo.long())*consis_pre +
                                                    ce_loss_soft2(output_all_u, U_data_pseudo.long()) * diff_pre1)
            loss_seg_ce_unlab_u_mix_consis = torch.mean(ce_loss2(output_all_u_mix, U_data_pseudo.long())*consis_pre+
                                                        ce_loss_soft2(output_all_u_mix, U_data_pseudo.long()) * diff_pre1)
            loss_seg_ce_unlab += (loss_seg_ce_unlab_u_consis+loss_seg_ce_unlab_u_mix_consis)/2


            # loss_seg_dice_unlab += dice_loss(output_all[args.labeled_bs:], U_data_pseudo.long().unsqueeze(1))
            loss_seg_dice_unlab +=( dice_loss_soft0(out_soft_u,U_data_pseudo.long().unsqueeze(1),loss_mask=consis_pre1) +
                                    dice_loss_soft0(out_soft_u_mix,U_data_pseudo.long().unsqueeze(1),loss_mask=consis_pre1)+
                                    dice_loss_soft1(out_soft_u,U_data_pseudo.long().unsqueeze(1),loss_mask=diff_pre1)+
                                    dice_loss_soft1(out_soft_u_mix,U_data_pseudo.long().unsqueeze(1),loss_mask=diff_pre1) ) /4


            pseudo_loss = 0.5 * (loss_seg_dice_unlab + loss_seg_ce_unlab)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + pseudo_loss*consistency_weight
            # loss_consist = 0
            iter_num = iter_num + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_consist = 0

            ema_optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)
            consistency_loss1 = 0
            # if iter_num % 100 == 0:
            #     logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (
            #         iter_num, loss, supervised_loss, loss_consist))

            if iter_num >= 1000 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case_base_out(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case_base_out(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=32, stride_z=32, dataset_name='Pancreas_CT')
                elif args.dataset_name == "BraTS2019":
                    dice_sample = test_patch.var_all_case_base_out(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='BraTS2019')
                else:dice_sample=0

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                print('best_dice',best_dice)
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
