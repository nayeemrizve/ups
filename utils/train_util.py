import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .misc import AverageMeter, accuracy


def train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, nl_loader)
    model.train()
    for batch_idx, (data_x, data_nl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs_x, targets_x, _, nl_mask_x = data_x
        inputs_nl, targets_nl, _, nl_mask_nl = data_nl

        inputs = torch.cat((inputs_x, inputs_nl)).to(args.device)
        targets = torch.cat((targets_x, targets_nl)).to(args.device)
        nl_mask = torch.cat((nl_mask_x, nl_mask_nl)).to(args.device)

        #network outputs
        logits = model(inputs)

        positive_idx = nl_mask.sum(dim=1) == args.num_classes #the mask for negative learning is all ones
        nl_idx = (nl_mask.sum(dim=1) != args.num_classes) * (nl_mask.sum(dim=1) > 0)
        loss_ce = 0
        loss_nl = 0

        #positive learning
        if sum(positive_idx*1) > 0:
            loss_ce += F.cross_entropy(logits[positive_idx], targets[positive_idx], reduction='mean')

        #negative learning
        if sum(nl_idx*1) > 0:
            nl_logits = logits[nl_idx]
            pred_nl = F.softmax(nl_logits, dim=1)
            pred_nl = 1 - pred_nl
            pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
            nl_mask = nl_mask[nl_idx]
            y_nl = torch.ones((nl_logits.shape)).to(device=args.device, dtype=logits.dtype)
            loss_nl += torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))

        loss = loss_ce + loss_nl
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.iterations,
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg


def train_initial(args, train_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    model.train()
    for batch_idx, (inputs, targets, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.iterations,
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()

    return losses.avg