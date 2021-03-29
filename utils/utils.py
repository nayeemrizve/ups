import math
import os
import random
import shutil
import numpy as np
import torch
import sys
sys.path.append('../..')
from torch.optim.lr_scheduler import LambdaLR

def save_checkpoint(state, is_best, checkpoint, itr):
    filename=f'checkpoint_{itr}.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best_{itr}.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=args.dropout,
                                        num_classes=args.num_classes)

    elif args.arch == 'cnn13':
        import models.cnn13 as models
        model = models.cnn13(num_classes=args.num_classes, dropout=args.dropout)
    
    elif args.arch == 'shakeshake':
        import models.shakeshake as models
        model = models.shakeshake_resnet(num_classes=args.num_classes, dropout=args.dropout)
    
    return model