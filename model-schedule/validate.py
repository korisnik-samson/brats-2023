import os
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.inferers import sliding_window_inference

from ..utils.model_utils import make_dataloader, compute_loss
from ..utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping, probs_to_preds, one_hot_channels_to_three_labels
from ..processing.plot import plot_slices

def validate(data_dir, ckpt_path, eval_regions='overlapping', out_dir=None, make_plots=False, batch_size=1, device=None):
    """Routine to validate a trained model on validation data. Optionally plots predictions against ground truth segmentations.
    Args:
        data_dir: Directory of validation data.
        ckpt_path: Path of trained model.
        eval_regions: Whether to evaluate on 'disjoint' or 'overlapping' regions. Defaults to 'overlapping'.
        out_dir: Directory in which to save plots. Defaults to None.
        make_plots: Whether to produce plots of predictions and ground truth segmentations. Defaults to False.
        batch_size: Batch size of dataloader. Defaults to 1.
        device: torch.device to use. If None, auto-detects.
    """
    # Device management
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up directories.
    if out_dir is None:
        out_dir = os.getcwd()

    if make_plots:
        plots_dir = os.path.join(out_dir, 'plots')

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            os.system(f'chmod a+rwx {plots_dir}')

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = checkpoint['model'].to(device)
    loss_functions = [lf.to(device) for lf in checkpoint['loss_functions']]
    loss_weights = checkpoint['loss_weights']
    training_regions = checkpoint['training_regions']

    epoch = checkpoint['epoch']
    model_sd = checkpoint['model_sd']

    model.load_state_dict(model_sd)

    print('Model loaded.')

    print("-" * 50)
    print("TRAINING SUMMARY")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}")
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")

    print("-" * 50)
    print("VALIDATION SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    print(f"Trained model checkpoint path: {ckpt_path}")
    print(f"Evaluation regions: {eval_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Make plots: {make_plots}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)

    # For validation, we use do_crop=False to get the full volume for sliding window
    val_loader = make_dataloader(data_dir, shuffle=False, mode='val', batch_size=batch_size, do_crop=False)

    val_loss_vals = []

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    roi_size = (192, 192, 128)
    sw_batch_size = 4

    print('Validation starts.')
    with torch.no_grad():
        for subject_names, imgs, seg in val_loader:

            model.eval()

            # Move data to device.
            imgs = [img.to(device) for img in imgs] # img is B1HWD
            seg = seg.to(device)

            # Split segmentation into 3 channels.
            seg = seg_to_one_hot_channels(seg) # seg is B3HWD

            if training_regions == 'overlapping':
                seg_train = disjoint_to_overlapping(seg)
            elif training_regions == 'disjoint':
                seg_train = seg

            x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
            
            # Use sliding window inference for validation
            output = sliding_window_inference(
                inputs=x_in, 
                roi_size=roi_size, 
                sw_batch_size=sw_batch_size, 
                predictor=model,
                overlap=0.5
            )
            output = output.float()

            # Compute weighted loss
            val_loss = compute_loss(output, seg_train, loss_functions, loss_weights, device)
            val_loss_vals.append(val_loss.detach().cpu())

            # Convert logits to probabilities for prediction and plotting
            probs = torch.sigmoid(output)
            preds = probs_to_preds(probs, training_regions)
            eval_region_names = []

            if eval_regions == 'overlapping':
                eval_region_names = ['WT', 'TC', 'ET']
                seg_eval = disjoint_to_overlapping(seg)
                preds_eval = disjoint_to_overlapping(preds)

            elif eval_regions == 'disjoint':
                eval_region_names = ['NCR', 'ED', 'ET']
                seg_eval = seg
                preds_eval = preds

            # Compute metrics
            dice_metric(y_pred = preds_eval, y=seg_eval)
            hd_metric(y_pred = preds_eval, y=seg_eval)

            if make_plots:
                for i, subject_name in enumerate(subject_names):
                    batch_imgs = [img[i, 0].cpu().detach() for img in imgs]
                    seg3 = one_hot_channels_to_three_labels(seg[i].cpu().detach())
                    pred3 = one_hot_channels_to_three_labels(preds[i].cpu().detach())

                    fig = plot_slices(batch_imgs, seg3, pred3)
                    fig.savefig(os.path.join(plots_dir, subject_name))

    # Compute and report validation loss.
    average_val_loss = np.mean(val_loss_vals)
    print(f'Validation completed. Average validation loss = {average_val_loss}')

    # Aggregate and report the metrics.
    dice_metric_batch = dice_metric.aggregate()
    for i, eval_region in enumerate(eval_region_names):
        print(f'Dice Score {eval_region} = {dice_metric_batch[i].item()}')

    hd_metric_batch = hd_metric.aggregate()

    for i, eval_region in enumerate(eval_region_names):
        print(f'HD95 {eval_region} = {hd_metric_batch[i].item()}')

if __name__ == '__main__':

    data_dir = '../data'
    ckpt_path = '../data/checkpoints'
    out_dir = '../data/output'

    validate(data_dir, ckpt_path, out_dir=out_dir)