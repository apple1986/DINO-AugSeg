
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95, dc   # pip install medpy
import os
from PIL import Image
from scipy.ndimage import zoom
import h5py

def evaluate_case(pred, target, num_classes=4, ignore_background=True):
    dice_scores, hd95_scores = [], []
    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    for c in class_range:
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_scores.append(1.0)
            hd95_scores.append(0.0)
            continue
        if target_c.sum() == 0 or pred_c.sum() == 0:
            dice_scores.append(0.0)
            hd95_scores.append(100.0)
            continue

        dice_scores.append(dc(pred_c, target_c))
        hd95_scores.append(hd95(pred_c, target_c))

    return dice_scores, hd95_scores

def evaluate_slice_dice(pred, target, num_classes=2, ignore_background=True):
    """
    Compute Dice score for one 2D slice.
    If both pred and target are empty (all zero), Dice = 0.

    Returns:
        dice_scores: list of Dice scores per class
    """
    dice_scores = []
    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    for c in class_range:
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        # ✅ If both are empty → Dice = 0
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_scores.append(0.0)
            continue

        # ✅ If only one is empty → Dice = 0
        if target_c.sum() == 0 or pred_c.sum() == 0:
            dice_scores.append(0.0)
            continue

        # ✅ Normal Dice computation
        dice_scores.append(dc(pred_c, target_c))

    return dice_scores


def evaluate_case_dice(pred, target, num_classes=4, ignore_background=True):
    dice_scores  = []
    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    for c in class_range:
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_scores.append(1.0)
            continue
        if target_c.sum() == 0 or pred_c.sum() == 0:
            dice_scores.append(0.0)
            continue

        dice_scores.append(dc(pred_c, target_c))

    return dice_scores

def test_model(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=4, device="cuda", split="test"):
    model = model.to(device)
    model.eval()

    results = []
    # ---- output directory (same folder as save_file) ----
    save_file = os.path.abspath(save_file)
    out_root = os.path.dirname(save_file)
    os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(root, "dataset/ACDC/"+split+".list"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/ACDC/"+split+"_data")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case + "_mr.nii.gz")
            mask_path = os.path.join(data_dir, case + "_gt.nii.gz")

            img_vol = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()
            orig_H, orig_W, num_slices = img_vol.shape

            case_dice, case_hd95 = [], []
            pred_vol = np.zeros_like(mask_vol)
            for s in range(num_slices):
                img = img_vol[:, :, s]                
                # --- preprocess with same training transform ---
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb  = np.stack([img_norm]*3, axis=-1)
                img_pil  = Image.fromarray((img_rgb*255).astype(np.uint8))
                img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

                # --- forward ---
                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                # --- resize prediction back to original size ---
                pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
                pred_resized = np.array(pred_resized)
                pred_vol[:, :, s] = pred_resized

            # --- metrics for the whole volume (ignore background) ---
            case_dice, case_hd95 = evaluate_case(pred_vol, mask_vol, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice RV/MYO/LV: {case_dice} | HD95: {case_hd95}")


            # =====================================================
            # Save volumes (same folder as save_file / case_name)
            # =====================================================
            case_out_dir = os.path.join(out_root, case)
            os.makedirs(case_out_dir, exist_ok=True)

            affine = np.eye(4)

            nib.save(
                nib.Nifti1Image(img_vol.astype(np.float32), affine),
                os.path.join(case_out_dir, f"{case}_img.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(mask_vol.astype(np.uint8), affine),
                os.path.join(case_out_dir, f"{case}_gt.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(pred_vol.astype(np.uint8), affine),
                os.path.join(case_out_dir, f"{case}_pred.nii.gz")
            )



    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice RV/MYO/LV: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice RV/MYO/LV: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 RV/MYO/LV:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")


    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results
#######################################
def test_model_acdc_save_fig(
    model,
    root,
    img_transform,
    save_path="acdc_vis",
    model_name="DINO-AugSeg",
    num_classes=4,   # background + 3 structures
    device="cuda"
):
    import os, numpy as np, torch, nibabel as nib
    from PIL import Image

    model = model.to(device)
    model.eval()
    results = []

    # ✅ NEW LOADING STYLE
    with open(os.path.join(root, f"dataset/ACDC/test.list"), 'r') as f:
        test_cases = [line.strip() for line in f.readlines()]

    data_dir = os.path.join(root, f"dataset/ACDC/test_data")
    save_dir = os.path.join(root, "dino_seg/seg_result", save_path)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ shared best-slice record (from cross_guide_wt_unet)
    slice_record_file = os.path.join(save_dir, "best_slices.txt")

    def load_slice_record():
        d = {}
        if os.path.exists(slice_record_file):
            with open(slice_record_file, "r") as f:
                for line in f:
                    case, idx = line.strip().split(",")
                    d[case] = int(idx)
        return d

    def save_slice_record(d):
        with open(slice_record_file, "w") as f:
            for k, v in d.items():
                f.write(f"{k},{v}\n")

    slice_dict = load_slice_record()

    # ✅ ACDC class colors: LV / MYO / RV
    class_colors = {
        1: np.array([255, 0, 0], dtype=np.uint8),    # LV - Red
        2: np.array([0, 255, 0], dtype=np.uint8),    # MYO - Green
        3: np.array([0, 0, 255], dtype=np.uint8),    # RV - Blue
    }

    alpha = 0.35

    def overlay_multiclass(base_img, label_map):
        out = base_img.astype(np.float32)
        for c, color in class_colors.items():
            mask_c = (label_map == c)
            if mask_c.sum() == 0:
                continue
            mask_3 = np.stack([mask_c] * 3, axis=-1)
            out = (
                out * (1 - mask_3)
                + ((1 - alpha) * out + alpha * color) * mask_3
            )
        return np.clip(out, 0, 255).astype(np.uint8)

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case + "_mr.nii.gz")
            mask_path = os.path.join(data_dir, case + "_gt.nii.gz")

            img_vol = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()

            orig_H, orig_W, num_slices = img_vol.shape
            pred_vol = np.zeros_like(mask_vol, dtype=np.uint8)

            print(f"\n✅ Processing ACDC case: {case}")

            # ============================================================
            # ✅ ONLY cross_guide_wt_unet SELECTS BEST SLICE
            # ============================================================
            if model_name == "cross_guide_wt_unet":
                best_dice = -1
                best_slice_idx = -1
                best_img_rgb = None
                best_mask = None
                best_pred = None

                for s in range(num_slices):
                    img = img_vol[:, :, s]
                    mask = mask_vol[:, :, s]

                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img_rgb = np.stack([img_norm] * 3, axis=-1)

                    img_tensor = img_transform(
                        Image.fromarray((img_rgb * 255).astype(np.uint8))
                    ).unsqueeze(0).to(device)

                    pred = model(img_tensor)
                    pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                    pred_resized = np.array(
                        Image.fromarray(pred.astype(np.uint8)).resize(
                            (orig_W, orig_H), resample=Image.NEAREST
                        )
                    )

                    pred_vol[:, :, s] = pred_resized

                    slice_dice_list = evaluate_slice_dice(
                        pred_resized, mask,
                        num_classes=num_classes,
                        ignore_background=True
                    )
                    slice_dice = float(np.mean(slice_dice_list))

                    if slice_dice > best_dice:
                        best_dice = slice_dice
                        best_slice_idx = s
                        best_img_rgb = (img_rgb * 255).astype(np.uint8)
                        best_mask = mask.copy()
                        best_pred = pred_resized.copy()

                slice_dict[case] = best_slice_idx
                save_slice_record(slice_dict)

                print(f"✅ Best slice saved: {best_slice_idx} | Dice={best_dice:.4f}")

            # ============================================================
            # ✅ OTHER MODELS LOAD SLICE FROM cross_guide_wt_unet
            # ============================================================
            else:
                best_slice_idx = slice_dict.get(case, None)
                if best_slice_idx is None:
                    print(f"⚠️ No saved best slice for {case}, skipping overlay.")

            # ============================================================
            # ✅ FULL VOLUME PREDICTION (ALL MODELS)
            # ============================================================
            for s in range(num_slices):
                img = img_vol[:, :, s]
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb = np.stack([img_norm] * 3, axis=-1)

                img_tensor = img_transform(
                    Image.fromarray((img_rgb * 255).astype(np.uint8))
                ).unsqueeze(0).to(device)

                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                pred_resized = np.array(
                    Image.fromarray(pred.astype(np.uint8)).resize(
                        (orig_W, orig_H), resample=Image.NEAREST
                    )
                )

                pred_vol[:, :, s] = pred_resized

            # ============================================================
            # ✅ SAVE OVERLAY IMAGES (3-CLASS)
            # ============================================================
            if best_slice_idx is not None and best_slice_idx >= 0:
                s = best_slice_idx

                img = img_vol[:, :, s]
                mask = mask_vol[:, :, s]
                pred_slice = pred_vol[:, :, s]

                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                best_img_rgb = (np.stack([img_norm] * 3, axis=-1) * 255).astype(np.uint8)

                pred_overlay = overlay_multiclass(best_img_rgb, pred_slice)

                saved_pd_name = os.path.join(
                    save_dir, f"{model_name}_{case}_slice{best_slice_idx}_pd.jpg"
                )
                Image.fromarray(pred_overlay).save(saved_pd_name)

                # ✅ ONLY cross_guide_wt_unet SAVES GT
                if model_name == "cross_guide_wt_unet":
                    gt_overlay = overlay_multiclass(best_img_rgb, mask)

                    saved_gt_name = os.path.join(
                        save_dir, f"{case}_slice{best_slice_idx}_gt.jpg"
                    )
                    Image.fromarray(gt_overlay).save(saved_gt_name)

            # ============================================================
            # ✅ VOLUME METRICS
            # ============================================================
            case_dice, case_hd95 = evaluate_case(
                pred_vol, mask_vol,
                num_classes=num_classes,
                ignore_background=True
            )

            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice: {case_dice} | HD95: {case_hd95}")

    return results


###############################################################################
def eval_model_all(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=4, device="cuda", split="test", case_num=None):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/ACDC/"+split+".list"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]
    if split == "train":
        test_cases = test_cases[:case_num]  # only eval on 30 training cases for quick demo

    data_dir = os.path.join(root, "dataset/ACDC/"+split+"_data")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case + "_mr.nii.gz")
            mask_path = os.path.join(data_dir, case + "_gt.nii.gz")

            img_vol = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()
            orig_H, orig_W, num_slices = img_vol.shape

            case_dice, case_hd95 = [], []
            pred_vol = np.zeros_like(mask_vol)
            for s in range(num_slices):
                img = img_vol[:, :, s]                
                # --- preprocess with same training transform ---
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb  = np.stack([img_norm]*3, axis=-1)
                img_pil  = Image.fromarray((img_rgb*255).astype(np.uint8))
                img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

                # --- forward ---
                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                # --- resize prediction back to original size ---
                pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
                pred_resized = np.array(pred_resized)
                pred_vol[:, :, s] = pred_resized

            # --- metrics for the whole volume (ignore background) ---
            # pred_vol shape: [H,W,num_slices]
            # # Compute zoom factors
            # zoom_factors = (224 / orig_H,
            #                 224 / orig_W,
            #                 1)
            # pred_vol = zoom(pred_vol, zoom_factors, order=0)
            # mask_vol = zoom(mask_vol, zoom_factors, order=0)
            case_dice, case_hd95 = evaluate_case(pred_vol, mask_vol, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice RV/MYO/LV: {case_dice} | HD95: {case_hd95}")

    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice RV/MYO/LV: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice RV/MYO/LV: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 RV/MYO/LV:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results

##############################################################################

def test_model_la2018(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/LA2018/test.list"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/LA2018")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case, "mri_norm2.h5")

            """Load a volume from h5 files."""
            vol_obj = h5py.File(img_path, 'r')
            img_vol = vol_obj['image'][:] # HWD
            mask_vol = vol_obj['label'][:] # HWD
            orig_H, orig_W, num_slices = img_vol.shape

            case_dice, case_hd95 = [], []
            pred_vol = np.zeros_like(mask_vol)
            for s in range(num_slices):
                img = img_vol[:, :, s]                
                # --- preprocess with same training transform ---
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb  = np.stack([img_norm]*3, axis=-1)
                img_pil  = Image.fromarray((img_rgb*255).astype(np.uint8))
                img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

                # --- forward ---
                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                # --- resize prediction back to original size ---
                pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
                pred_resized = np.array(pred_resized)
                pred_vol[:, :, s] = pred_resized

            case_dice, case_hd95 = evaluate_case(pred_vol, mask_vol, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice LV: {case_dice} | HD95: {case_hd95}")

    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice LV: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice LV: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 LV:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        # for d in mean_dice:
        #     f.write(f"{d*100:.2f}\t")
        # for h in mean_hd95:
        #     f.write(f"{h:.2f}\t")
        f.write("\n")

    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results


def test_model_la2018_save_fig(
    model,
    root,
    img_transform,
    save_path="la2018_vis",
    model_name="DINO-AugSeg",
    num_classes=2,
    device="cuda"
):
    import os, h5py, numpy as np, torch
    from PIL import Image

    model = model.to(device)
    model.eval()
    results = []

    with open(os.path.join(root, "dataset/LA2018/test.list"), 'r') as f:
        test_cases = [line.strip() for line in f.readlines()]

    data_dir = os.path.join(root, "dataset/LA2018")
    save_dir = os.path.join(root, "dino_seg/seg_result", save_path)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ shared best-slice record
    slice_record_file = os.path.join(save_dir, "best_slices.txt")

    def load_slice_record():
        d = {}
        if os.path.exists(slice_record_file):
            with open(slice_record_file, "r") as f:
                for line in f:
                    case, idx = line.strip().split(",")
                    d[case] = int(idx)
        return d

    def save_slice_record(d):
        with open(slice_record_file, "w") as f:
            for k, v in d.items():
                f.write(f"{k},{v}\n")

    slice_dict = load_slice_record()

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case, "mri_norm2.h5")
            vol_obj = h5py.File(img_path, 'r')

            img_vol = vol_obj['image'][:]    # H W D
            mask_vol = vol_obj['label'][:]  # H W D
            orig_H, orig_W, num_slices = img_vol.shape
            pred_vol = np.zeros_like(mask_vol)

            print(f"\n✅ Processing case: {case}")

            # ============================================================
            # ✅ ONLY cross_guide_wt_unet SELECTS BEST SLICE
            # ============================================================
            if model_name == "cross_guide_wt_unet":
                best_dice = -1
                best_slice_idx = -1
                best_pred = None
                best_img_rgb = None
                best_mask_np = None

                for s in range(num_slices):
                    img = img_vol[:, :, s]
                    mask = mask_vol[:, :, s]

                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img_rgb = np.stack([img_norm] * 3, axis=-1)
                    img_tensor = img_transform(
                        Image.fromarray((img_rgb * 255).astype(np.uint8))
                    ).unsqueeze(0).to(device)

                    pred = model(img_tensor)
                    pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                    pred_resized = np.array(
                        Image.fromarray(pred.astype(np.uint8)).resize(
                            (orig_W, orig_H), resample=Image.NEAREST
                        )
                    )

                    pred_vol[:, :, s] = pred_resized

                    slice_dice_list = evaluate_slice_dice(
                        pred_resized, mask,
                        num_classes=num_classes,
                        ignore_background=True
                    )

                    slice_dice = float(np.mean(slice_dice_list))

                    if slice_dice > best_dice:
                        best_dice = slice_dice
                        best_slice_idx = s
                        best_pred = pred_resized.copy()
                        best_img_rgb = (img_rgb * 255).astype(np.uint8)
                        best_mask_np = (mask > 0).astype(np.uint8)

                slice_dict[case] = best_slice_idx
                save_slice_record(slice_dict)

                print(f"✅ Best slice saved: {best_slice_idx} | Dice={best_dice:.4f}")

            # ============================================================
            # ✅ OTHER MODELS: LOAD SLICE FROM cross_guide_wt_unet
            # ============================================================
            else:
                if case not in slice_dict:
                    print(f"⚠️ No saved best slice for {case}, skipping overlay.")
                    best_slice_idx = None
                else:
                    best_slice_idx = slice_dict[case]

            # ============================================================
            # ✅ FULL VOLUME PREDICTION (ALL MODELS)
            # ============================================================
            for s in range(num_slices):
                img = img_vol[:, :, s]
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb = np.stack([img_norm] * 3, axis=-1)

                img_tensor = img_transform(
                    Image.fromarray((img_rgb * 255).astype(np.uint8))
                ).unsqueeze(0).to(device)

                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                pred_resized = np.array(
                    Image.fromarray(pred.astype(np.uint8)).resize(
                        (orig_W, orig_H), resample=Image.NEAREST
                    )
                )

                pred_vol[:, :, s] = pred_resized

            # ============================================================
            # ✅ SAVE OVERLAYS
            # ============================================================
            if best_slice_idx is not None and best_slice_idx >= 0:
                s = best_slice_idx

                img = img_vol[:, :, s]
                mask = mask_vol[:, :, s]
                pred_slice = pred_vol[:, :, s]

                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                best_img_rgb = (np.stack([img_norm]*3, axis=-1) * 255).astype(np.uint8)

                saved_pd_name = os.path.join(
                    save_dir, f"{model_name}_{case}_slice{best_slice_idx}_pd.jpg"
                )

                # ✅ SAFE OVERLAY (NO CORRUPTION)
                pred_color = np.array([255, 0, 0], dtype=np.uint8)
                alpha = 0.35
                pred_mask = (pred_slice > 0)
                pred_mask_3 = np.stack([pred_mask]*3, axis=-1)

                pred_overlay = best_img_rgb.astype(np.float32)
                pred_overlay = (
                    pred_overlay * (1 - pred_mask_3)
                    + ((1 - alpha) * pred_overlay + alpha * pred_color) * pred_mask_3
                )

                pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)
                Image.fromarray(pred_overlay).save(saved_pd_name)

                # ✅ ONLY cross_guide_wt_unet SAVES GT
                if model_name == "cross_guide_wt_unet":
                    gt_color = np.array([0, 255, 0], dtype=np.uint8)
                    gt_mask = (mask > 0)
                    gt_mask_3 = np.stack([gt_mask]*3, axis=-1)

                    gt_overlay = best_img_rgb.astype(np.float32)
                    gt_overlay = (
                        gt_overlay * (1 - gt_mask_3)
                        + ((1 - alpha) * gt_overlay + alpha * gt_color) * gt_mask_3
                    )
                    gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)

                    saved_gt_name = os.path.join(
                        save_dir, f"{case}_slice{best_slice_idx}_gt.jpg"
                    )
                    Image.fromarray(gt_overlay).save(saved_gt_name)

            # ============================================================
            # ✅ VOLUME METRICS
            # ============================================================
            case_dice, case_hd95 = evaluate_case(
                pred_vol, mask_vol,
                num_classes=num_classes,
                ignore_background=True
            )

            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice: {case_dice} | HD95: {case_hd95}")

    return results

##############################################################################
def test_model_kvasir(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/Kvasir_SEG/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/Kvasir_SEG")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, "images", case + ".jpg")
            gt_path = os.path.join(data_dir, "masks", case + ".jpg")
        
            # full paths
            """Load image and mask for .jpg format."""
            img = Image.open(img_path).convert("RGB")   # force 3-channel RGB
            mask = Image.open(gt_path).convert("L")     # grayscale mask

            # normalize image [0,1]
            img = np.array(img).astype(np.float32)
            mask = np.array(mask).astype(np.uint8)   # convert back to numpy
            # threshold: convert all non-zero values to 1
            mask = (mask > 125).astype(np.uint8) # threhold

            orig_H, orig_W = img.shape[:2]
            case_dice, case_hd95 = [], []
            pred = np.zeros_like(mask)
    
                    
            # --- preprocess with same training transform ---
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_pil  = Image.fromarray((img_norm*255).astype(np.uint8))
            img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

            # --- forward ---
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

            # --- resize prediction back to original size ---
            pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
            pred_resized = np.array(pred_resized)
            pred = pred_resized

            case_dice, case_hd95 = evaluate_case(pred, mask, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice Obj: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice Obj: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 Obj:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        f.write("\n")

    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results


def test_model_kvasir_save_fig(model, root, img_transform, save_path="", model_name="DINO-AugSeg", num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/Kvasir_SEG/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/Kvasir_SEG")
    select_cases = ["cju16ach3m1da0993r1dq3sn2", "cju8d4jgatgpj0871q2ophhkm", "cjz14qsk2wci60794un9ozwmw", 
                    "cju334jzo261t0835yqudnfs1", "cju87r56lnkyp0755hz30leew"]

    save_dir = os.path.join(root, "dino_seg/seg_result", save_path)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for case in test_cases:
            if case not in select_cases:
                continue

            img_path = os.path.join(data_dir, "images", case + ".jpg")
            gt_path = os.path.join(data_dir, "masks", case + ".jpg")

            saved_pd_name = os.path.join(save_dir, model_name + "_" + case + "_pd.jpg")
            saved_gt_name = os.path.join(save_dir, model_name + "_" + case + "_gt.jpg")
            print(f"case: {case}")

            # Load image and mask
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(gt_path).convert("L")

            img_np = np.array(img).astype(np.uint8)
            mask_np = (np.array(mask) > 125).astype(np.uint8)

            orig_H, orig_W = img_np.shape[:2]

            # Preprocess
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_tensor = img_transform(Image.fromarray((img_norm*255).astype(np.uint8))).unsqueeze(0).to(device)

            # Forward pass
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST))

            # Evaluate
            case_dice, case_hd95 = evaluate_case(pred_resized, mask_np, num_classes=num_classes, ignore_background=True)
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

            # --- Overlay masks ---
            # Define colors
            gt_color = np.array([0, 255, 0], dtype=np.uint8)    # green
            pred_color = np.array([0, 0, 255], dtype=np.uint8)  # blue

            if model_name in ["cross_guide_wt_unet"]:
                # Ground truth overlay
                gt_overlay = img_np.copy()
                gt_mask_idx = mask_np.astype(bool)
                gt_overlay[gt_mask_idx] = (0.5 * gt_overlay[gt_mask_idx] + 0.5 * gt_color).astype(np.uint8)
                Image.fromarray(gt_overlay).save(saved_gt_name)
            # Prediction overlay
            pred_overlay = img_np.copy()
            pred_mask_idx = pred_resized > 0   # make sure mask is boolean
            pred_overlay[pred_mask_idx] = (0.6 * pred_overlay[pred_mask_idx] + 0.4 * pred_color).astype(np.uint8)
            Image.fromarray(pred_overlay).save(saved_pd_name)


    return results


##############################################################################
def test_model_isic2018(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/ISIC2018/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/ISIC2018")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, "ISIC2018_Task1-2_Training_Input", case + ".jpg")
            gt_path = os.path.join(data_dir, "ISIC2018_Task1_Training_GroundTruth", case + "_segmentation.png")
        
            # full paths
            """Load image and mask for .jpg format."""
            img = Image.open(img_path).convert("RGB")   # force 3-channel RGB
            mask = Image.open(gt_path).convert("L")     # grayscale mask

            # normalize image [0,1]
            img = np.array(img).astype(np.float32)
            mask = np.array(mask).astype(np.uint8)   # convert back to numpy
            # threshold: convert all non-zero values to 1
            mask = (mask > 125).astype(np.uint8) # threhold

            orig_H, orig_W = img.shape[:2]
            case_dice, case_hd95 = [], []
            pred = np.zeros_like(mask)
    
                    
            # --- preprocess with same training transform ---
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_pil  = Image.fromarray((img_norm*255).astype(np.uint8))
            img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

            # --- forward ---
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

            # --- resize prediction back to original size ---
            pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
            pred_resized = np.array(pred_resized)
            pred = pred_resized

            case_dice, case_hd95 = evaluate_case(pred, mask, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice Obj: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice Obj: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 Obj:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        f.write("\n")

    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results


def test_model_isic2018_save_fig(model, root, img_transform, save_file="", model_name="DINO-AugSeg", num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/ISIC2018/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/ISIC2018")
    select_cases = ["ISIC_0013007", "ISIC_0000177", "ISIC_0015155", "ISIC_0000240", "ISIC_0010044"]

    save_dir = "/home/gxu/proj1/lesionSeg/dino_seg/seg_result/"+save_file
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for case in test_cases:
            if case not in select_cases:
                continue

            img_path = os.path.join(data_dir, "ISIC2018_Task1-2_Training_Input", case + ".jpg")
            gt_path = os.path.join(data_dir, "ISIC2018_Task1_Training_GroundTruth", case + "_segmentation.png")

            saved_pd_name = os.path.join(save_dir, model_name + "_" + case + "_pd.jpg")
            saved_gt_name = os.path.join(save_dir, model_name + "_" + case + "_gt.jpg")
            print(f"case: {case}")

            # Load image and mask
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(gt_path).convert("L")

            img_np = np.array(img).astype(np.uint8)
            mask_np = (np.array(mask) > 125).astype(np.uint8)

            orig_H, orig_W = img_np.shape[:2]

            # Preprocess
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_tensor = img_transform(Image.fromarray((img_norm*255).astype(np.uint8))).unsqueeze(0).to(device)

            # Forward pass
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST))

            # Evaluate
            case_dice, case_hd95 = evaluate_case(pred_resized, mask_np, num_classes=num_classes, ignore_background=True)
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

            # --- Overlay masks ---
            # Define colors
            gt_color = np.array([0, 255, 0], dtype=np.uint8)    # green
            pred_color = np.array([0, 0, 255], dtype=np.uint8)  # blue

            # Ground truth overlay
            if model_name in ["cross_guide_wt_unet"]:
                gt_overlay = img_np.copy()
                gt_mask_idx = mask_np.astype(bool)
                gt_overlay[gt_mask_idx] = (0.5 * gt_overlay[gt_mask_idx] + 0.5 * gt_color).astype(np.uint8)
                Image.fromarray(gt_overlay).save(saved_gt_name)

            # Prediction overlay
            pred_overlay = img_np.copy()
            pred_mask_idx = pred_resized.astype(bool)
            pred_overlay[pred_mask_idx] = (0.8 * pred_overlay[pred_mask_idx] + 0.2 * pred_color).astype(np.uint8)
            Image.fromarray(pred_overlay).save(saved_pd_name)

    return results
##############################################################################

def test_model_kn3k(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=2, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []

    with open(os.path.join(root, "dataset/TN3K/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/TN3K")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, "test-image", case)
            gt_path = os.path.join(data_dir, "test-mask", case)
        
            # full paths
            """Load image and mask for .jpg format."""
            img = Image.open(img_path).convert("RGB")   # force 3-channel RGB
            mask = Image.open(gt_path).convert("L")     # grayscale mask

            # normalize image [0,1]
            img = np.array(img).astype(np.float32)
            mask = np.array(mask).astype(np.uint8)   # convert back to numpy
            # threshold: convert all non-zero values to 1
            mask = (mask > 125).astype(np.uint8) # threhold

            orig_H, orig_W = img.shape[:2]
            case_dice, case_hd95 = [], []
            pred = np.zeros_like(mask)
    
                    
            # --- preprocess with same training transform ---
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_pil  = Image.fromarray((img_norm*255).astype(np.uint8))
            img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

            # --- forward ---
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

            # --- resize prediction back to original size ---
            pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
            pred_resized = np.array(pred_resized)
            pred = pred_resized

            # # temp to training size 512x512
            # mask = Image.fromarray(mask.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
            # mask = np.array(mask)
            # pred_resized = np.array(pred)
            # pred = pred_resized


            case_dice, case_hd95 = evaluate_case(pred, mask, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

    # --- save results ---

    with open(save_file, "w") as f:
        for case, d, h in results:
            f.write(f"{case} | Dice Obj: {d} | HD95: {h}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        # overall mean across test set
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)
        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write(f"Mean Dice Obj: \n")
        for d in mean_dice:
            f.write(f"{d*100:.2f}\t")
        f.write("\n")
        # write the mean Dice
        f.write("Mean Dice\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\n")
        f.write(f"Mean HD95 Obj:\n")
        for h in mean_hd95:
            f.write(f"{h:.2f}\t")
        f.write("\n")
        # write the mean HD95
        f.write("Mean HD95\n")
        f.write(f"{np.mean(mean_hd95):.2f}\n")
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        f.write("\n")

    print(f"✅ Results saved to {save_file_all}")
    print(f"{np.mean(mean_dice)*100:.2f}")
    return results

def test_model_kn3k_save_fig(model, root, img_transform, save_path="", model_name="DINO-AugSeg", num_classes=2, device="cuda"):

    model = model.to(device)
    model.eval()
    results = []

    with open(os.path.join(root, "dataset/TN3K/test_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, "dataset/TN3K")
    # Optional: you can define a small subset to save
    select_cases = ["0021.jpg", "0084.jpg", "0043.jpg", "0066.jpg", "0131.jpg"]  # replace with your cases or leave empty

    save_dir = os.path.join(root, "dino_seg/seg_result", save_path)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for case in test_cases:
            if select_cases and case not in select_cases:
                continue

            img_path = os.path.join(data_dir, "test-image", case)
            gt_path = os.path.join(data_dir, "test-mask", case)

            saved_pd_name = os.path.join(save_dir, model_name + "_" + case.split(".jpg")[0] + "_pd.jpg")
            saved_gt_name = os.path.join(save_dir, model_name + "_" + case.split(".jpg")[0] + "_gt.jpg")
            print(f"case: {case}")

            # Load image and mask
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(gt_path).convert("L")

            img_np = np.array(img).astype(np.uint8)
            mask_np = (np.array(mask) > 125).astype(np.uint8)

            orig_H, orig_W = img_np.shape[:2]

            # Preprocess
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_tensor = img_transform(Image.fromarray((img_norm*255).astype(np.uint8))).unsqueeze(0).to(device)

            # Forward pass
            pred = model(img_tensor)
            pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST))

            # Evaluate
            case_dice, case_hd95 = evaluate_case(pred_resized, mask_np, num_classes=num_classes, ignore_background=True)
            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice Obj: {case_dice} | HD95: {case_hd95}")

            # --- Overlay masks ---
            gt_color = np.array([0, 255, 0], dtype=np.uint8)    # green
            pred_color = np.array([0, 0, 255], dtype=np.uint8)  # red

            if model_name in ["cross_guide_wt_unet"]:
                # Ground truth overlay
                gt_overlay = img_np.copy()
                gt_mask_idx = mask_np.astype(bool)
                gt_overlay[gt_mask_idx] = (0.5 * gt_overlay[gt_mask_idx] + 0.5 * gt_color).astype(np.uint8)
                Image.fromarray(gt_overlay).save(saved_gt_name)

            # Prediction overlay
            pred_overlay = img_np.copy()
            pred_mask_idx = pred_resized > 0
            pred_overlay[pred_mask_idx] = (0.8 * pred_overlay[pred_mask_idx] + 0.2 * pred_color).astype(np.uint8)
            Image.fromarray(pred_overlay).save(saved_pd_name)

    return results



##############################################################
def test_model_synapse(model, root, img_transform, save_file="results.txt", img_size=768, num_classes=9, device="cuda", split="test"):
    """
    num_classes = 9 for Synapse (0=background, 1–8 organs)
    """
    model = model.to(device)
    model.eval()

    results = []

    # Load test list
    with open(os.path.join(root, f"dataset/Synapse/{split}_dino.txt"), 'r') as f:
        lines = f.readlines()
    test_cases = [line.strip() for line in lines]

    data_dir = os.path.join(root, f"dataset/Synapse/{split}_nii")

    with torch.no_grad():
        for case in test_cases:
            img_path = os.path.join(data_dir, case + "_ct.nii.gz")
            mask_path = os.path.join(data_dir, case + "_gt.nii.gz")

            img_vol = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()
            orig_H, orig_W, num_slices = img_vol.shape

            pred_vol = np.zeros_like(mask_vol)

            for s in range(num_slices):
                img = img_vol[:, :, s]
                # normalize
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb  = np.stack([img_norm]*3, axis=-1)
                img_pil  = Image.fromarray((img_rgb*255).astype(np.uint8))
                img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

                # forward
                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                # resize prediction back
                pred_resized = Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), resample=Image.NEAREST)
                pred_resized = np.array(pred_resized)
                pred_vol[:, :, s] = pred_resized

            # --- metrics for the whole volume (ignore background=0) ---
            case_dice, case_hd95 = evaluate_case(pred_vol, mask_vol, num_classes=num_classes, ignore_background=True)     
            results.append((case, case_dice, case_hd95))

            dice_str = " | ".join([f"{d:.3f}" for d in case_dice])
            print(f"[{case}] Dice (8 organs): {dice_str} | HD95: {case_hd95}")

    # --- save results ---
    with open(save_file, "w") as f:
        for case, d, h in results:
            d_str = "\t".join([f"{x:.3f}" for x in d])
            h_str = "\t".join([f"{x:.2f}" for x in h])
            f.write(f"{case} | Dice: {d_str} | HD95: {h_str}\n")

    save_file_all = save_file.replace(".txt", "_all.txt")
    with open(save_file_all, "w") as f:
        mean_dice = np.mean([d for _, d, _ in results], axis=0)
        mean_hd95 = np.mean([h for _, _, h in results], axis=0)

        f.write("\n=== Overall Averages (ignore background) ===\n")
        f.write("Mean Dice per organ (%):\n")
        f.write("\t".join([f"{d*100:.2f}" for d in mean_dice]) + "\n")
        f.write(f"Average Dice (mean of organs): {np.mean(mean_dice)*100:.2f}\n\n")

        f.write("Mean HD95 per organ:\n")
        f.write("\t".join([f"{h:.2f}" for h in mean_hd95]) + "\n")
        f.write(f"Average HD95: {np.mean(mean_hd95):.2f}\n\n")

        # Copy-paste friendly
        f.write("Copy directly\n")
        f.write(f"{np.mean(mean_dice)*100:.2f}\t{np.mean(mean_hd95):.2f}\t")
        f.write("\t".join([f"{d*100:.2f}" for d in mean_dice]) + "\t")
        f.write("\t".join([f"{h:.2f}" for h in mean_hd95]) + "\n")

    print(f"✅ Results saved to {save_file_all}")
    print(f"📊 Mean Dice (8 organs): {np.mean(mean_dice)*100:.2f}")
    return results

def test_model_synapse_save_fig(
    model,
    root,
    img_transform,
    split="test",
    save_path="synapse_vis",
    model_name="DINO-AugSeg",
    num_classes=9,   # background + 8 organs
    device="cuda"
):
    import os, numpy as np, torch, nibabel as nib
    from PIL import Image

    model = model.to(device)
    model.eval()
    results = []

    # ✅ Load test list (YOUR FORMAT)
    with open(os.path.join(root, f"dataset/Synapse/{split}_dino.txt"), "r") as f:
        test_cases = [line.strip() for line in f.readlines()]

    data_dir = os.path.join(root, f"dataset/Synapse/{split}_nii")
    save_dir = os.path.join(root, "dino_seg/seg_result", save_path)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Shared best-slice record
    slice_record_file = os.path.join(save_dir, "best_slices.txt")

    def load_slice_record():
        d = {}
        if os.path.exists(slice_record_file):
            with open(slice_record_file, "r") as f:
                for line in f:
                    case, idx = line.strip().split(",")
                    d[case] = int(idx)
        return d

    def save_slice_record(d):
        with open(slice_record_file, "w") as f:
            for k, v in d.items():
                f.write(f"{k},{v}\n")

    slice_dict = load_slice_record()

    # ✅ Synapse 8-Class Colors
    class_colors = {
        1: np.array([255,   0,   0], dtype=np.uint8),  # Spleen
        2: np.array([  0, 255,   0], dtype=np.uint8),  # Right Kidney
        3: np.array([  0,   0, 255], dtype=np.uint8),  # Left Kidney
        4: np.array([255, 255,   0], dtype=np.uint8),  # Gallbladder
        5: np.array([255,   0, 255], dtype=np.uint8),  # Esophagus
        6: np.array([  0, 255, 255], dtype=np.uint8),  # Liver
        7: np.array([128,   0, 128], dtype=np.uint8),  # Stomach
        8: np.array([255, 128,   0], dtype=np.uint8),  # Pancreas
    }

    alpha = 0.35

    def overlay_multiclass(base_img, label_map):
        out = base_img.astype(np.float32)
        for c, color in class_colors.items():
            mask_c = (label_map == c)
            if mask_c.sum() == 0:
                continue
            mask_3 = np.stack([mask_c] * 3, axis=-1)
            out = out * (~mask_3) + ((1 - alpha) * out + alpha * color) * mask_3
        return np.clip(out, 0, 255).astype(np.uint8)

    with torch.no_grad():
        for case in test_cases:
            # ✅ YOUR PATH FORMAT
            img_path  = os.path.join(data_dir, case + "_ct.nii.gz")
            mask_path = os.path.join(data_dir, case + "_gt.nii.gz")

            img_vol  = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()
            orig_H, orig_W, num_slices = img_vol.shape

            pred_vol = np.zeros_like(mask_vol, dtype=np.uint8)

            print(f"\n✅ Processing Synapse case: {case}")

            # ============================================
            # ✅ Best Slice Selection (Reference Model)
            # ============================================
            if model_name == "cross_guide_wt_unet":
                best_dice = -1
                best_slice_idx = -1

                for s in range(num_slices):
                    img  = img_vol[:, :, s]
                    mask = mask_vol[:, :, s]

                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img_rgb  = np.stack([img_norm] * 3, axis=-1)

                    img_tensor = img_transform(
                        Image.fromarray((img_rgb * 255).astype(np.uint8))
                    ).unsqueeze(0).to(device)

                    pred = model(img_tensor)
                    pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                    pred_resized = np.array(
                        Image.fromarray(pred.astype(np.uint8)).resize(
                            (orig_W, orig_H), resample=Image.NEAREST
                        )
                    )

                    pred_vol[:, :, s] = pred_resized

                    slice_dice_list = evaluate_slice_dice(
                        pred_resized, mask,
                        num_classes=num_classes,
                        ignore_background=True
                    )
                    slice_dice = float(np.mean(slice_dice_list))

                    if slice_dice > best_dice:
                        best_dice = slice_dice
                        best_slice_idx = s

                slice_dict[case] = best_slice_idx
                save_slice_record(slice_dict)

                print(f"✅ Best slice selected: {best_slice_idx} | Dice={best_dice:.4f}")

            # ============================================
            # ✅ Load Best Slice for Other Models
            # ============================================
            else:
                best_slice_idx = slice_dict.get(case, None)
                if best_slice_idx is None:
                    print(f"⚠️ No saved best slice for {case}, skipping overlay.")

            # ============================================
            # ✅ Full Volume Prediction
            # ============================================
            for s in range(num_slices):
                img = img_vol[:, :, s]
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_rgb = np.stack([img_norm] * 3, axis=-1)

                img_tensor = img_transform(
                    Image.fromarray((img_rgb * 255).astype(np.uint8))
                ).unsqueeze(0).to(device)

                pred = model(img_tensor)
                pred = torch.softmax(pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()

                pred_resized = np.array(
                    Image.fromarray(pred.astype(np.uint8)).resize(
                        (orig_W, orig_H), resample=Image.NEAREST
                    )
                )

                pred_vol[:, :, s] = pred_resized

            # ============================================
            # ✅ Save Overlay Images
            # ============================================
            if best_slice_idx is not None and best_slice_idx >= 0:
                s = best_slice_idx

                img  = img_vol[:, :, s]
                mask = mask_vol[:, :, s]
                pred_slice = pred_vol[:, :, s]

                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                best_img_rgb = (np.stack([img_norm] * 3, axis=-1) * 255).astype(np.uint8)

                pred_overlay = overlay_multiclass(best_img_rgb, pred_slice)

                pd_name = os.path.join(
                    save_dir, f"{model_name}_{case}_slice{best_slice_idx}_pd.jpg"
                )
                # Rotate prediction overlay 90° counter-clockwise
                pred_overlay_rot = Image.fromarray(pred_overlay).rotate(90, expand=True)
                pred_overlay_rot.save(pd_name)

                # ✅ Save GT only once
                if model_name == "cross_guide_wt_unet":
                    gt_overlay = overlay_multiclass(best_img_rgb, mask)
                    gt_name = os.path.join(
                        save_dir, f"{case}_slice{best_slice_idx}_gt.jpg"
                    )
                    gt_overlay_rot = Image.fromarray(gt_overlay).rotate(90, expand=True)
                    gt_overlay_rot.save(gt_name)

            # ============================================
            # ✅ Volume Metrics
            # ============================================
            case_dice, case_hd95 = evaluate_case(
                pred_vol, mask_vol,
                num_classes=num_classes,
                ignore_background=True
            )

            results.append((case, case_dice, case_hd95))
            print(f"[{case}] Dice: {case_dice} | HD95: {case_hd95}")

    return results
