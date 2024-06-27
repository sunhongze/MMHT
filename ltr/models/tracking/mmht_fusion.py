import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
import ltr.models.fusion as fusion
from ltr.admin.model_constructor import model_constructor

class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, fusion_block, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()

        self.feature_extractor_ann = feature_extractor[0]
        self.feature_extractor_snn = feature_extractor[1]
        self.fusion_block = fusion_block
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_event_stack, test_event_stack, train_bb,  test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        # Extract ann backbone features
        train_feat_ann = self.extract_backbone_features(im=train_imgs.reshape(-1, *train_imgs.shape[-3:]), modal='vis')
        test_feat_ann = self.extract_backbone_features(im=test_imgs.reshape(-1, *test_imgs.shape[-3:]), modal='vis')
        # Extract snn backbone features
        train_feat_snn = self.extract_backbone_features(im=torch.stack([train_event_stack[0].reshape(-1, *train_event_stack[0].shape[-3:]),
                                                                    train_event_stack[1].reshape(-1, *train_event_stack[1].shape[-3:]),
                                                                    train_event_stack[2].reshape(-1, *train_event_stack[2].shape[-3:])], axis=1), modal='dvs')
        test_feat_snn  = self.extract_backbone_features(im=torch.stack([test_event_stack[0].reshape(-1, *test_event_stack[0].shape[-3:]),
                                                                    test_event_stack[1].reshape(-1, *test_event_stack[1].shape[-3:]),
                                                                    test_event_stack[2].reshape(-1, *test_event_stack[2].shape[-3:])], axis=1), modal='dvs')

        #############################################sunfusion12##############################################
        train_fea_l, train_fea_h = self.fusion_block(train_feat_ann['layer2'], train_feat_snn['layer2'], train_feat_ann['layer3'], train_feat_snn['layer3'])
        test_fea_l,  test_fea_h  = self.fusion_block(test_feat_ann['layer2'],  test_feat_snn['layer2'],  test_feat_ann['layer3'],  test_feat_snn['layer3'])

        # Run classifier module
        target_scores = self.classifier(train_fea_h, test_fea_h, train_bb, *args, **kwargs)
        # Run the IoUNet module
        iou_pred = self.bb_regressor([train_fea_l, train_fea_h], [test_fea_l,  test_fea_h], train_bb, test_proposals)

        # Run classifier module
        # target_scores = self.classifier(train_feat[-1], test_feat[-1], train_bb, *args, **kwargs)
        # Run the IoUNet module
        # iou_pred = self.bb_regressor(train_feat, test_feat, train_bb, test_proposals)


        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        # feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        # if len(self.classification_layer) == 1:
        #     return feat[self.classification_layer[0]]
        return backbone_feat[-1]

    def get_backbone_bbreg_feat(self, backbone_feat):
        return backbone_feat

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None, modal='vis'):
        if layers is None:
            layers = self.output_layers
        if modal == 'vis':
            return self.feature_extractor_ann(im)
        elif modal == 'dvs':
            return self.feature_extractor_snn(im)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})

@model_constructor
def klcedimpnetFussion(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=[True, False], clf_feat_blocks=1,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=256, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, init_pool_square=False,
                  frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    if type(backbone_pretrained) is bool:
        backbone_ann = backbones.resnet18(output_layers=['layer2', 'layer3'], pretrained=backbone_pretrained)
        backbone_snn = backbones.alexsnn(output_layers=['layer2', 'layer3'], pretrained=backbone_pretrained)
    elif type(backbone_pretrained) is list:
        backbone_ann = backbones.resnet18(output_layers=['layer2', 'layer3'], pretrained=backbone_pretrained[0])
        backbone_snn = backbones.alexsnn(output_layers=['layer2', 'layer3'], pretrained=backbone_pretrained[1])




    # fusion block
    fusion_block = fusion.FiT(num_patches=81, patch_dim=256, head=2, num_fusion_layer=2)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(feature_dim=out_feature_dim, num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale, out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer,
                                                          pool_square=init_pool_square)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(128,256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=[backbone_ann, backbone_snn], fusion_block=fusion_block, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


