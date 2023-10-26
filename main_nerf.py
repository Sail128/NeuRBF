# Modified from https://github.com/apchenstu/TensoRF/blob/main/train.py

import os
from tqdm.auto import tqdm
from thirdparty.tensorf.opt import config_parser

import json, random
from thirdparty.tensorf.renderer import *
from thirdparty.tensorf.utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from thirdparty.tensorf.dataLoader import dataset_dict
import sys
import pprint
import numpy as np
import torch

from thirdparty.tensorf.models.tensorBase import count_params
import thirdparty.tensorf.models.tensorBase
import thirdparty.tensorf.models.tensoRF
import thirdparty.nrff.nrff
import nerf_tensorf.network
import util_misc

import time
import pprint


gpu = util_misc.select_devices('1#', force_reselect=True, excludeID=[])[0]
if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu}")
else:
    print(f"Did not set GPU.")

temp = torch.ones([100, 100], device=0)
del temp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def load_model_from_ckpt(ckpt_path, model_name):
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    if 'args' in kwargs and 'rbf_config' in kwargs['args']:
        kwargs['args']['rbf_config']['init_rbf'] = False
    tensorf = eval(model_name)(**kwargs)
    tensorf.load(ckpt)
    tensorf.to(device)
    return tensorf


@torch.no_grad()
def export_mesh(args):
    tensorf = load_model_from_ckpt(args.ckpt, args.model_name)
    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    tensorf = load_model_from_ckpt(args.ckpt, args.model_name)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_train, _, _, _ = evaluation(
            train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {logfolder} train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


def reconstruction(args, init_data=None):
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    # init log file
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    reso_real_cur = N_to_reso(args.N_voxel_real_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    t = time.time()
    if args.ckpt is not None:
        tensorf = load_model_from_ckpt(args.ckpt, args.model_name)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, 
                    alphaMask_thres=args.alpha_mask_thre, rayMarch_weight_thres=args.rm_weight_mask_thre,
                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, 
                    args=args, gridSize_real=reso_real_cur, init_data=init_data)
    tensorf.to(device)
    torch.cuda.synchronize()
    t_init = time.time() - t

    n_params = count_params(tensorf)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
    N_voxel_real_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_real_init), np.log(args.N_voxel_real_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    batch_size = args.batch_size_init

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    t_train = 0
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        t = time.time()
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            rgb_map, alphas_map, depth_map, weights, others = renderer(rays_train, tensorf, chunk=batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

            total_loss = torch.mean((rgb_map - rgb_train) ** 2)
            loss = total_loss.detach().item()

            # loss
            if others['normals'] is not None and args.Ro_weight > 0:
                Ro = torch.sum(others['normals'] * others['valid_viewdirs'], dim=-1)
                Ro = F.relu(Ro).pow(2) * others['valid_weights']
                Ro  = Ro.mean()
                loss_Ro = args.Ro_weight * Ro
                total_loss += loss_Ro
                summary_writer.add_scalar('train/reg_Ro', loss_Ro.detach().item(), global_step=iteration)

            if Ortho_reg_weight > 0:
                loss_reg = tensorf.vector_comp_diffs()
                total_loss += Ortho_reg_weight*loss_reg
                summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
            if L1_reg_weight > 0:
                loss_reg_L1 = tensorf.density_L1()
                total_loss += L1_reg_weight*loss_reg_L1
                summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

            if TV_weight_density>0:
                TV_weight_density *= lr_factor
                loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
            if TV_weight_app>0:
                TV_weight_app *= lr_factor
                loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        if args.fp16:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t_train += time.time() - t

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)
        summary_writer.add_scalar('train/stepSize', tensorf.stepSize, global_step=iteration)
        summary_writer.add_scalar('train/nSamples', tensorf.nSamples, global_step=iteration)

        # Update lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test, _, _, _ = evaluation(
                test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray,
                compute_extra_metrics=False, save_img=args.save_img, save_video=args.save_video)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)


        t = time.time()
        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            
            do_shrink = True
            if args.shrink_0:
                do_shrink = iteration == update_AlphaMask_list[0]
            if do_shrink:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                batch_size = args.batch_size
                trainingSampler = SimpleSampler(allrgbs.shape[0], batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            n_voxels_real = N_voxel_real_list.pop(0)
            reso_real_cur = N_to_reso(n_voxels_real, tensorf.aabb)
            if args.scale_reso:
                reso_real_cur = scale_reso(reso_real_cur, n_voxels_real)
            tensorf.upsample_volume_grid(reso_cur, reso_real_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

            if args.no_upsample:
                if iteration == upsamp_list[0]:
                    batch_size = args.batch_size
                    trainingSampler = SimpleSampler(allrgbs.shape[0], batch_size)

        torch.cuda.synchronize()
        t_train += time.time() - t

        init_data = tensorf.step_after_iter(iteration + 1)
        if init_data is not None:
            return init_data

        n_params = count_params(tensorf, False)
        for k, v in n_params.items():
            summary_writer.add_scalar(f'n_params/{k}', v, global_step=iteration)
            
        summary_writer.add_scalar(f'time/train', t_train, global_step=iteration)

    summary_writer.add_scalar(f'time/init', t_init, global_step=iteration)
    summary_writer.add_scalar(f'time/total', t_init + t_train, global_step=iteration)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_train, _, _, _ = evaluation(
            train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
            save_img=args.save_img, save_video=args.save_video)
        print(f'======> train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test, ssims_test, l_vgg_test, l_alex_test = evaluation(
            test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
            save_img=args.save_img, save_video=args.save_video)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        summary_writer.add_scalar('test/ssim_all', np.mean(ssims_test), global_step=iteration)
        summary_writer.add_scalar('test/lpips_vgg_all', np.mean(l_vgg_test), global_step=iteration)
        summary_writer.add_scalar('test/lpips_alex_all', np.mean(l_alex_test), global_step=iteration)
        print(f'======> test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        
    n_params = count_params(tensorf, False)
    print({args.expname})

    if args.save_ckpt:
        tensorf.save(f'{logfolder}/ckpt.th')

    stats_t = {'t_init': t_init, 't_train': t_train, 't_total': t_init + t_train}
    stats_misc = {'n_iter': args.n_iters, 'batch_size': args.batch_size, 'n_param': n_params}
    print(stats_t)
    print(stats_misc)
    if args.render_test:
        stats_metric = {'psnr': np.mean(PSNRs_test), 'ssim': np.mean(ssims_test), 'lpips_vgg': np.mean(l_vgg_test), 'lpips_alex': np.mean(l_alex_test)}
        print(stats_metric)
    
    save_fn = os.path.join(logfolder, 'stats')
    with open(f'{save_fn}.txt', 'w') as f:
        pprint.pprint(stats_t, f, sort_dicts=False)
        pprint.pprint(stats_misc, f, sort_dicts=False)
        if args.render_test:
            pprint.pprint(stats_metric, f, sort_dicts=False)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    pprint.pprint(args)
    print(args.expname)

    if args.export_mesh:
        export_mesh(args)
    elif args.render_only:
        if args.render_train or args.render_test or args.render_path:
            render_test(args)
    else:
        init_data = None
        if args.config_init is not None:
            print('Get init data...')
            args = config_parser(mode='config_init')
            pprint.pprint(args)
            print(args.expname)
            if os.path.exists(args.rbf_config.init_data_fp):
                init_data = torch.load(args.rbf_config.init_data_fp)
            else:
                init_data = reconstruction(args)
                torch.cuda.empty_cache()

        print('Training...')
        args = config_parser()
        pprint.pprint(args)
        print(args.expname)
        reconstruction(args, init_data)
