import os
import logging

import torch
import numpy as np
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, CNAME_TEMPLATES, A_CNAME_TEMPLATES, PHOTO_TEMPLATES
from .precision import get_autocast
from .metrics import get_imb_metrics, get_nc_metrics, get_corrs


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    all_features, all_targets = [], []
    confusion_matrix = torch.zeros(classifier.shape[-1], classifier.shape[-1], device=args.device)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'].float() if isinstance(output, dict) else output[0]
                image_features_ori = output['image_features_ori'].float() if isinstance(output, dict) else output[-3]
                logits = 100. * image_features @ classifier

            if args.nc_metrics:
                all_features.append(image_features_ori)
                all_targets.append(target)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            
            # update confusion matrix
            _, predicted = torch.max(logits, 1)
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    top1 = (top1 / n)
    top5 = (top5 / n)

    val_metrics = {'top1': top1, 'top5': top5}
    if classifier.shape[-1] == 100:
        idxs = np.loadtxt(args.imagenet100_index_file, dtype=int).tolist()
    else:
        idxs = None
    save_metrics = {}
    if args.imb_metrics:
        per_class_accs, per_class_pred_cnt = get_imb_metrics(confusion_matrix)
        save_metrics.update({'confusion': confusion_matrix.cpu(), 'per_class_accs': per_class_accs.cpu(), 'per_class_pred_cnt': per_class_pred_cnt.cpu()})
    if args.nc_metrics:
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest, cos_W_nearest = get_nc_metrics(model, all_features, all_targets)
        save_metrics.update({'Sw_invSb': Sw_invSb.cpu(), 'cos_M_all': cos_M_all.cpu(), 'cos_W_all': cos_W_all.cpu(), 'cos_M_nearest': cos_M_nearest.cpu(), 'cos_W_nearest': cos_W_nearest.cpu()})
        del all_features, all_targets
    
    if args.imb_metrics or args.nc_metrics:
        metrics_path = os.path.join(args.log_base_path, 'metrics')
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
        
        try:
            freqs = np.array([int(line.strip().split('\t')[1]) for line in open(args.frequency_file, 'r')])
            if idxs is not None:
                freqs = freqs[idxs]
            corr_statistics = get_corrs(save_metrics, freqs)
            val_metrics.update(corr_statistics)
        except:
            print("Frequency file not found, skipping correlation statistics.")
    
    return val_metrics, save_metrics

def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-100' not in data and 'cub' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    if args.template_type == 'cname':
        templates = CNAME_TEMPLATES
    elif args.template_type == 'a+cname':
        templates = A_CNAME_TEMPLATES
    elif args.template_type == 'photo':
        templates = PHOTO_TEMPLATES
    elif args.template_type == 'simple':
        templates = SIMPLE_IMAGENET_TEMPLATES
    else:
        templates = OPENAI_IMAGENET_TEMPLATES
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        val_metrics, save_metrics = run(model, classifier, data['imagenet-val'].dataloader, args)
        results.update({'imagenet-zeroshot-val-' + k: v for k, v in val_metrics.items()})
        if args.imb_metrics and args.nc_metrics:
            metrics_path = os.path.join(args.log_base_path, 'metrics')
            if 'train' in data:
                torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet_zeroshot_val_ep{args.current_epoch}.pt"))
            else:
                if args.template_type != "openai":
                    metrics_path += '_{}'.format(args.template_type)
                    os.makedirs(metrics_path, exist_ok=True)
            torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet_zeroshot_val_latest.pt"))
    if 'imagenet-v2' in data:
        val_metrics, save_metrics = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results.update({'imagenet-zeroshot-v2-' + k: v for k, v in val_metrics.items()})
        if args.imb_metrics and args.nc_metrics:
            metrics_path = os.path.join(args.log_base_path, 'metrics')
            if 'train' in data:
                torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet_zeroshot_v2_ep{args.current_epoch}.pt"))
            else:
                if args.template_type != "openai":
                    metrics_path += '_{}'.format(args.template_type)
                    os.makedirs(metrics_path, exist_ok=True)
            torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet_zeroshot_v2_latest.pt"))
    if 'imagenet-100' in data:
        idxs = np.loadtxt(args.imagenet100_index_file, dtype=int).tolist()
        val_metrics, save_metrics = run(model, classifier[:, idxs], data['imagenet-100'].dataloader, args)
        results.update({'imagenet100-zeroshot-val-' + k: v for k, v in val_metrics.items()})
        if args.imb_metrics and args.nc_metrics:
            metrics_path = os.path.join(args.log_base_path, 'metrics')
            if 'train' in data:
                torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet100_zeroshot_val_ep{args.current_epoch}.pt"))
            else:
                if args.template_type != "openai":
                    metrics_path += '_{}'.format(args.template_type)
                    os.makedirs(metrics_path, exist_ok=True)
            torch.save(save_metrics, os.path.join(metrics_path, f"metrics_imagenet100_zeroshot_val_latest.pt"))
    logging.info('Finished zero-shot evaluation.')
    return results
