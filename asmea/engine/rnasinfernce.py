import logging
import time

import torch.nn as nn
import torch
import numpy as np

class data_prefetcher():
    def __init__(self, loader, mean=None, std=None, is_cutout=False, cutout_length=16):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if mean is None:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([m * 255 for m in mean]).cuda().view(1, 3, 1, 1)
        if std is None:
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        else:
            self.std = torch.tensor([s * 255 for s in std]).cuda().view(1, 3, 1, 1)
        self.is_cutout = is_cutout
        self.cutout_length = cutout_length
        self.preload()

    def normalize(self, data):
        data = data.float()
        data = data.sub_(self.mean).div_(self.std)
        return data

    def cutout(self, data):
        batch_size, h, w = data.shape[0], data.shape[2], data.shape[3]
        mask = torch.ones(batch_size, h, w).cuda()
        y = torch.randint(low=0, high=h, size=(batch_size,))
        x = torch.randint(low=0, high=w, size=(batch_size,))

        y1 = torch.clamp(y - self.cutout_length // 2, 0, h)
        y2 = torch.clamp(y + self.cutout_length // 2, 0, h)
        x1 = torch.clamp(x - self.cutout_length // 2, 0, w)
        x2 = torch.clamp(x + self.cutout_length // 2, 0, w)
        for i in range(batch_size):
            mask[i][y1[i]: y2[i], x1[i]: x2[i]] = 0.
        mask = mask.expand_as(data.transpose(0, 1)).transpose(0, 1)
        data *= mask
        return data

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.normalize(self.next_input)
            if self.is_cutout:
                self.next_input = self.cutout(self.next_input)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
class Trainer(object):
    def __init__(self, train_data, val_data, criterion=None, config=None, report_freq=None):
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.config = config
        self.report_freq = report_freq

    def train(self, model, optimizer, epoch):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        model.train()
        start = time.time()

        prefetcher = data_prefetcher(self.train_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            data_t = time.time() - start
            n, h, w = input.size(0), input.size(2), input.size(3)
            if step == 0:
                logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()
            logits, logits_aux = model(input)
            if self.config.optim.label_smooth:
                loss = self.criterion(logits, target)
                if self.config.optim.auxiliary:
                    loss_aux = self.criterion(logits_aux, target)
                    loss += self.config.optim.auxiliary_weight * loss_aux
            else:
                loss = self.criterion(logits, target)
                if self.config.optim.auxiliary:
                    loss_aux = self.criterion(logits_aux, target)
                    loss += self.config.optim.auxiliary_weight * loss_aux

            loss.backward()
            if self.config.optim.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
            optimizer.step()

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))

            batch_t = time.time() - start
            start = time.time()

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)
            # if step > 5:
            #    break
            if step != 0 and step % self.report_freq == 0:
                logging.info(
                    'Train epoch %03d step %03d | loss %.4f  top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f',
                    epoch, step, objs.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            input, target = prefetcher.next()
            step += 1
        logging.info('EPOCH%d Train_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', epoch, top1.avg, top5.avg,
                     batch_time.avg, data_time.avg)

        return top1.avg, top5.avg, objs.avg, batch_time.avg, data_time.avg

    def infer(self, model, epoch=0):
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        model.eval()

        start = time.time()
        prefetcher = data_prefetcher(self.val_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)

            logits, logits_aux = model(input)
            print(logits)
            print(target)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))

            batch_t = time.time() - start
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step % self.report_freq == 0:
                print('Val epoch %03d step %03d | top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f',
                      epoch, step, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            start = time.time()
            input, target = prefetcher.next()

        logging.info('EPOCH%d Valid_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', epoch, top1.avg, top5.avg,
                     batch_time.avg, data_time.avg)
        return top1.avg, top5.avg, batch_time.avg, data_time.avg
