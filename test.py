import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Overwriting nextvit_small in registry")
warnings.filterwarnings("ignore", message="Overwriting nextvit_base in registry")
warnings.filterwarnings("ignore", message="Overwriting nextvit_large in registry")
warnings.filterwarnings("ignore")

import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from KUPCP_dataset import CompositionDataset, composition_cls
from Cropping_dataset import FCDBDataset, FLMSDataset
from GAICD_dataset import GAICDataset
from TAD66K_dataset import ThemeDataset, theme_cls
from config_cropping import cfg
from CACNet import CACNet


device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def read_image_unicode(image_file):
    image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_file}')
    return image


def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    """
    :param gt_crop: Tensor, shape [N, 4], [x1, y1, x2, y2]
    :param pre_crop: Tensor, shape [M, 4], [x1, y1, x2, y2]
    :return: best_iou, best_disp
    """
    gt_crop = gt_crop[gt_crop[:, 0] >= 0]
    if gt_crop.numel() == 0:
        return 0.0, 1.0

    gt_crop = gt_crop.to(pre_crop.device)
    zero_t = torch.zeros(gt_crop.shape[0], device=gt_crop.device)

    over_x1 = torch.maximum(gt_crop[:, 0], pre_crop[:, 0])
    over_y1 = torch.maximum(gt_crop[:, 1], pre_crop[:, 1])
    over_x2 = torch.minimum(gt_crop[:, 2], pre_crop[:, 2])
    over_y2 = torch.minimum(gt_crop[:, 3], pre_crop[:, 3])
    over_w = torch.maximum(zero_t, over_x2 - over_x1)
    over_h = torch.maximum(zero_t, over_y2 - over_y1)
    inter = over_w * over_h

    area1 = (gt_crop[:, 2] - gt_crop[:, 0]) * (gt_crop[:, 3] - gt_crop[:, 1])
    area2 = (pre_crop[:, 2] - pre_crop[:, 0]) * (pre_crop[:, 3] - pre_crop[:, 1])
    union = area1 + area2 - inter + 1e-12
    iou = inter / union

    disp = (
        (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w +
        (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    )

    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()


def build_crop_test_dataset(dataset_name):
    if dataset_name == 'FCDB':
        return FCDBDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
    if dataset_name == 'FLMS':
        return FLMSDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
    if dataset_name == 'GAICD':
        return GAICDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
    raise ValueError(f'Undefined test set: {dataset_name}')


def evaluate_on_FCDB_and_FLMS(model, dataset, save_results=False):
    model.eval()
    model_device = next(model.parameters()).device

    accum_disp = 0.0
    accum_iou = 0.0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0

    if save_results:
        save_file = os.path.join(results_dir, dataset + '.json')
        crop_dir = os.path.join(results_dir, dataset)
        os.makedirs(crop_dir, exist_ok=True)
        test_results = {}

    print('=' * 5, f'Evaluating on {dataset}', '=' * 5)

    with torch.no_grad():
        test_dataset = build_crop_test_dataset(dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False
        )

        for _, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(model_device)
            gt_crop = batch_data[1]
            width = batch_data[2].item()
            height = batch_data[3].item()
            image_file = batch_data[4][0]
            image_name = os.path.basename(image_file)

            _, _, crop = model(im, operation='cropping')

            crop[:, 0::2] = crop[:, 0::2] / im.shape[-1] * width
            crop[:, 1::2] = crop[:, 1::2] / im.shape[-2] * height
            pred_crop = crop.detach().cpu()
            gt_crop = gt_crop.reshape(-1, 4).float()

            pred_crop[:, 0::2] = torch.clip(pred_crop[:, 0::2], min=0, max=width)
            pred_crop[:, 1::2] = torch.clip(pred_crop[:, 1::2], min=0, max=height)

            iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
            if iou >= alpha:
                alpha_cnt += 1
            accum_iou += iou
            accum_disp += disp
            cnt += 1

            if save_results:
                best_crop = pred_crop[0].numpy().tolist()
                best_crop = [int(x) for x in best_crop]
                test_results[image_name] = best_crop

                try:
                    source_img = read_image_unicode(image_file)
                    cropped_img = source_img[best_crop[1]:best_crop[3], best_crop[0]:best_crop[2]]
                    cv2.imwrite(os.path.join(crop_dir, image_name), cropped_img)
                except Exception as exc:
                    print(f'failed to save crop for {image_file}: {exc}')

    if cnt == 0:
        raise RuntimeError(f'No samples were evaluated for dataset {dataset}.')

    if save_results:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

    avg_iou = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


def visualize_com_prediction(image_path, logits, kcm, category, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    _, predicted = torch.max(logits.data, 1)
    label = composition_cls[predicted[0].item()]
    gt_label = [composition_cls[c] for c in category[0].numpy().tolist()]

    im = read_image_unicode(image_path[0])
    height, width, _ = im.shape
    dst = im.copy()
    dst = cv2.putText(dst, f'gt:{gt_label}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    dst = cv2.putText(dst, f'predict:{label}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

    kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
    norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
    norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
    heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
    heat_im = cv2.resize(heat_im, (width, height))
    fuse_im = cv2.addWeighted(im, 0.2, heat_im, 0.8, 0)
    fuse_im = np.concatenate([dst, fuse_im], axis=1)
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path[0])), fuse_im)


def prepare_eval_labels(labels):
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    return labels.long()


def evaluate_composition_classification(model):
    model.eval()
    model_device = next(model.parameters()).device
    print('=' * 5, 'Evaluating on Composition Classification Dataset', '=' * 5)

    total = 0
    correct = 0
    cls_cnt = [0 for _ in range(len(composition_cls))]
    cls_correct = [0 for _ in range(len(composition_cls))]

    with torch.no_grad():
        test_dataset = CompositionDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False
        )

        for _, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(model_device)
            labels = prepare_eval_labels(batch_data[1]).cpu()

            logits, _ = model(im, operation='composition')
            logits = logits.cpu()
            _, predicted = torch.max(logits.data, 1)

            total += labels.shape[0]
            pr = predicted[0].item()
            gt = labels.numpy().tolist()

            if pr in gt:
                correct += 1
                cls_cnt[pr] += 1
                cls_correct[pr] += 1
            else:
                cls_cnt[gt[0]] += 1

    acc = safe_ratio(correct, total)
    print('Test on {} images, {} Correct, Acc {:.2%}'.format(total, correct, acc))
    for i in range(len(cls_cnt)):
        cls_acc = safe_ratio(cls_correct[i], cls_cnt[i])
        print('{}: total {} images, {} correct, Acc {:.2%}'.format(
            composition_cls[i], cls_cnt[i], cls_correct[i], cls_acc
        ))
    return acc


def evaluate_theme_classification(model):
    model.eval()
    model_device = next(model.parameters()).device
    print('=' * 5, 'Evaluating on Theme Classification Dataset', '=' * 5)

    total = 0
    correct = 0
    theme_cnt = [0 for _ in range(len(theme_cls))]
    theme_correct = [0 for _ in range(len(theme_cls))]

    with torch.no_grad():
        test_theme_dataset = ThemeDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
        test_theme_loader = torch.utils.data.DataLoader(
            test_theme_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False
        )

        for _, batch_data in enumerate(tqdm(test_theme_loader)):
            im = batch_data[0].to(model_device)
            labels = prepare_eval_labels(batch_data[1]).cpu()

            logits, _ = model(im, operation='theme')
            logits = logits.cpu()
            _, predicted = torch.max(logits.data, 1)

            total += labels.shape[0]
            pr = predicted[0].item()
            gt = labels.numpy().tolist()

            if pr in gt:
                correct += 1
                theme_cnt[pr] += 1
                theme_correct[pr] += 1
            else:
                theme_cnt[gt[0]] += 1

    acc = safe_ratio(correct, total)
    print('Test on {} images, {} Correct, Acc {:.2%}'.format(total, correct, acc))
    for i in range(len(theme_cnt)):
        theme_acc = safe_ratio(theme_correct[i], theme_cnt[i])
        print('{}: total {} images, {} correct, Acc {:.2%}'.format(
            theme_cls[i], theme_cnt[i], theme_correct[i], theme_acc
        ))
    return acc


if __name__ == '__main__':
    print('test script started')

    weight_file = './experiments/cropping_0.6croploss_0.2compositionloss_0.2themeloss_0.2contrastive/checkpoints/best-FCDB_disp.pth'

    model = CACNet(loadweights=False)
    loaded_state = torch.load(weight_file, map_location=device)
    model.load_state_dict(loaded_state)
    model = model.to(device).eval()

    evaluate_on_FCDB_and_FLMS(model, dataset='FCDB', save_results=True)
    evaluate_on_FCDB_and_FLMS(model, dataset='FLMS', save_results=True)

    # 如需验证辅助任务，可取消下面两行注释
    # evaluate_composition_classification(model)
    # evaluate_theme_classification(model)