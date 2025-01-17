import torch.optim as optim
from ltr.dataset import FE108
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import mmht_fusion
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from admin.multigpu import MultiGPU



def run(settings):
    settings.description = 'Default train settings for MMHT.'
    settings.batch_size = 2
    settings.num_workers = 1
    settings.multi_gpu = False
    settings.print_interval = 500
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18  #18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.print_stats = ['Loss/total', 'Loss/bb_ce', 'ClfTrain/clf_ce']

    # Train datasets
    fe108_train = FE108(settings.env.fe108_dir, split='train')
    fe108_val   = FE108(settings.env.fe108_dir, split='val')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    transform_val   = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz, 'normalize': True}

    data_processing_train = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)
    data_processing_val   = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_val,
                                                        joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.DiMPSampler([fe108_train], [1],
                                        samples_per_epoch=40000, max_gap=50, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=1)
    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler([fe108_val], [1],
                                      samples_per_epoch=5000, max_gap=50, num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers, shuffle=False, drop_last=True, epoch_interval=1, stack_dim=1)

    # Create network and actor
    net = mmht_fusion.klcedimpnetFussion(filter_size=settings.target_filter_sz, backbone_pretrained=[True, False], optim_iter=5,
                            clf_feat_norm=True, final_conv=True, optim_init_step=1.0, optim_init_reg=0.05, optim_min_reg=0.05,
                            gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True, init_initializer='zero')

    objective = {'bb_ce': klreg_losses.KLRegression(), 'clf_ce': klreg_losses.KLRegressionGrid()}
    loss_weight = {'bb_ce': 0.25, 'clf_ce': 0.25, 'clf_ce_init': 0.25, 'clf_ce_iter': 1.0}
    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.classifier.parameters(),   'lr': 1e-3},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.fusion_block.parameters(), 'lr': 1e-3},
                            {'params': actor.net.feature_extractor_ann.parameters(), 'lr': 1e-4},
                            {'params': actor.net.feature_extractor_snn.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    trainer.train(50, load_latest=False, fail_safe=True)
