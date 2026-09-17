"""
Solar Filament Segmentation Algorithm
======================================
Author  : [Your Name]
Dataset : H-alpha chromospheric images, Jan–Mar 2011
Task    : Pixel-level instance segmentation of solar filaments
Metric  : Panoptic Quality (PQ = SQ × RQ) + Dice Score

What are solar filaments?
  Dark, elongated structures visible on the solar disk in H-alpha images.
  They are dense, cool plasma suspended in the hot corona by magnetic fields.
  They appear as long dark threads against the brighter chromosphere.

Pipeline overview:
  1. Disk extraction     — isolate the solar disk from black background
  2. Limb darkening fix  — correct the brightness gradient across the disk
  3. Local contrast map  — filaments are darker than their local surroundings
  4. Adaptive threshold  — separate filament pixels from normal chromosphere
  5. Morphological clean — remove noise, keep elongated structures
  6. Instance labelling  — assign unique ID to each individual filament
  7. Metrics             — Dice score + Panoptic Quality evaluation

Dependencies:
  pip install opencv-python-headless scikit-image scipy numpy matplotlib
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from scipy import ndimage
from skimage import morphology, measure, filters


# ══════════════════════════════════════════════════════════════
# STEP 1: SOLAR DISK EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_solar_disk(img: np.ndarray) -> tuple[np.ndarray, object]:
    """
    Find and isolate the solar disk from the black background.

    The background is pure black (pixel = 0). The disk is a bright,
    roughly circular region. We threshold and find the largest connected
    component — that's the Sun.

    Args:
        img: grayscale H-alpha image (uint8, 2048×2048)

    Returns:
        disk_mask: binary mask (255 = inside disk, 0 = outside)
        disk_prop: region properties of the disk (centroid, area, etc.)
    """
    # Simple threshold to separate bright disk from dark background
    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    # Fill any interior holes (e.g. dark sunspots that might break the disk)
    binary_filled = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Find connected components and keep only the largest = the solar disk
    labels = measure.label(binary_filled)
    props  = measure.regionprops(labels)

    if not props:
        return binary_filled, None

    disk_prop = max(props, key=lambda p: p.area)
    disk_mask = (labels == disk_prop.label).astype(np.uint8) * 255

    return disk_mask, disk_prop


# ══════════════════════════════════════════════════════════════
# STEP 2: LIMB DARKENING CORRECTION
# ══════════════════════════════════════════════════════════════
def correct_limb_darkening(img: np.ndarray,
                            disk_mask: np.ndarray,
                            u: float = 0.6) -> np.ndarray:
    """
    Correct limb darkening — the solar disk is brightest at the centre
    and gets darker toward the edges (the limb). This creates a brightness
    gradient that would make filament detection worse near the edges.

    We model the darkening using the standard formula:
        I(r) = I_0 × [1 - u × (1 - cos θ)]
    where cos θ ≈ sqrt(1 - (r/R)²) for a spherical disk.

    We divide the image by this model to get a uniform disk.

    Args:
        img       : raw grayscale image
        disk_mask : binary disk mask
        u         : limb darkening coefficient (0.6 typical for H-alpha)

    Returns:
        corrected : float32 image with limb darkening removed
    """
    h, w   = img.shape
    cy, cx = h // 2, w // 2

    # Distance from disk centre for every pixel
    Y, X = np.ogrid[:h, :w]
    r    = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Normalise by maximum disk radius
    r_max = r[disk_mask > 0].max() + 1e-9
    r_norm = r / r_max

    # cos(θ) where θ is the heliocentric angle
    cos_theta = np.sqrt(np.clip(1 - r_norm**2, 0, 1))

    # Limb darkening model
    ld_model = 1 - u * (1 - cos_theta)
    ld_model = np.where(disk_mask > 0, ld_model, 1.0)

    # Divide out the model to get a flat disk
    corrected = np.clip(img.astype(float) / (ld_model + 1e-9), 0, 255)
    return corrected.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# STEP 3–6: FILAMENT DETECTION & INSTANCE SEGMENTATION
# ══════════════════════════════════════════════════════════════
def detect_filaments(img_raw: np.ndarray,
                     disk_mask: np.ndarray,
                     contrast_sigma: float = 0.8,
                     min_area_px: int = 50,
                     min_elongation: float = 2.0,
                     large_blob_area: int = 500) -> tuple:
    """
    Main filament detection function.

    Filaments are DARK, ELONGATED structures on the solar disk.
    Key observation: filaments are darker than their LOCAL neighbourhood,
    not just darker than the disk average. This means we need local
    contrast, not global thresholding.

    Steps:
      a) Correct limb darkening (Step 2)
      b) Enhance local contrast with CLAHE
      c) Compute local brightness difference: blur(large) - local
         → dark regions get large positive values in this map
      d) Adaptive threshold on the difference map
      e) Remove small noise (<50 px) and close small gaps
      f) Filter by elongation (filaments are long, not round)
      g) Label each remaining connected component as one filament

    Args:
        img_raw         : raw grayscale image
        disk_mask       : binary disk mask
        contrast_sigma  : threshold = mean + sigma*std of difference map
        min_area_px     : discard objects smaller than this
        min_elongation  : minimum major/minor axis ratio to keep
        large_blob_area : keep blobs larger than this regardless of shape

    Returns:
        filament_mask   : binary mask (True = filament pixel)
        labeled_mask    : integer label per filament instance (0 = background)
        filament_info   : list of dicts with per-filament properties
        diff_img        : the local contrast map (useful for debugging)
    """

    # ── Step a: Limb darkening correction ──────────────────────
    corrected    = correct_limb_darkening(img_raw, disk_mask)
    corrected_u8 = np.clip(corrected, 0, 255).astype(np.uint8)

    # ── Step b: CLAHE — Contrast Limited Adaptive Histogram Equalization
    # This boosts local contrast without amplifying noise globally.
    # tileGridSize=(64,64) means we operate on ~32×32 pixel tiles
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    enhanced = clahe.apply(corrected_u8)

    # ── Step c: Local contrast map ─────────────────────────────
    # Large Gaussian blur estimates the local background brightness.
    # Subtracting the original from the blur gives us dark regions
    # (filaments) as bright spots in the difference map.
    blur_large = cv2.GaussianBlur(enhanced.astype(float), (101, 101), 0)
    diff       = blur_large - enhanced.astype(float)

    # ── Step d: Adaptive threshold ─────────────────────────────
    # Compute threshold as mean + k*std of difference values inside disk.
    # Higher k = fewer detections but higher precision.
    disk_vals = diff[disk_mask > 0]
    threshold = np.mean(disk_vals) + contrast_sigma * np.std(disk_vals)
    candidate = (diff > threshold) & (disk_mask > 0)

    # ── Step e: Morphological cleanup ──────────────────────────
    # Remove tiny isolated bright pixels (camera noise, speckles)
    candidate = morphology.remove_small_objects(
        candidate, min_size=min_area_px, connectivity=2)

    # Close small breaks in filament threads
    # (a filament that has a thin gap should still be one object)
    selem     = morphology.disk(2)
    candidate = morphology.closing(candidate, selem)

    # ── Step f & g: Filter by elongation + label instances ─────
    labeled   = measure.label(candidate)
    props     = measure.regionprops(labeled)

    filament_mask = np.zeros_like(img_raw, dtype=bool)
    filament_info = []

    for prop in props:
        if prop.area < min_area_px:
            continue

        # Get axis lengths (handle both old and new skimage API)
        try:
            mj = prop.axis_major_length
            mn = prop.axis_minor_length
        except AttributeError:
            mj = prop.major_axis_length
            mn = prop.minor_axis_length

        if mn < 1:
            continue

        elongation = mj / (mn + 1e-9)

        # Keep if elongated (filament-like) or simply large
        if elongation >= min_elongation or prop.area >= large_blob_area:
            filament_mask[labeled == prop.label] = True
            filament_info.append({
                "label":       int(prop.label),
                "area_px":     int(prop.area),
                "length_px":   float(mj),
                "width_px":    float(mn),
                "elongation":  float(elongation),
                "centroid_y":  float(prop.centroid[0]),
                "centroid_x":  float(prop.centroid[1]),
                "orientation": float(prop.orientation),
                "bbox":        list(prop.bbox),  # min_row, min_col, max_row, max_col
            })

    # Re-label the final clean mask so labels are contiguous 1..N
    final_labeled = measure.label(filament_mask)

    return filament_mask, final_labeled, filament_info, diff


# ══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════
def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice coefficient = 2×|A∩B| / (|A|+|B|)

    Perfect overlap → 1.0. No overlap → 0.0.
    This is a pixel-level metric — it doesn't distinguish instances.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom        = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


def compute_panoptic_quality(pred_labeled: np.ndarray,
                              gt_labeled: np.ndarray,
                              iou_thresh: float = 0.5) -> tuple[float, float, float]:
    """
    Panoptic Quality (PQ) = Segmentation Quality (SQ) × Recognition Quality (RQ)

    From Kirillov et al. (CVPR 2019) https://doi.org/10.1109/CVPR.2019.00963

    SQ = mean IoU of matched (pred, gt) pairs
       = measures how well we segment each filament
    RQ = F1 score of instance-level detection
       = measures how many filaments we found vs missed vs invented
    PQ = SQ × RQ = combined metric

    Matching: a predicted instance is matched to a GT instance if their
    IoU exceeds iou_thresh (typically 0.5). Each GT/pred can only be
    matched once (Hungarian-like greedy matching by best IoU).

    Args:
        pred_labeled : integer label map for predictions (0=background)
        gt_labeled   : integer label map for ground truth (0=background)
        iou_thresh   : minimum IoU to count as a match (default 0.5)

    Returns:
        PQ, SQ, RQ
    """
    pred_ids = np.unique(pred_labeled[pred_labeled > 0])
    gt_ids   = np.unique(gt_labeled[gt_labeled   > 0])

    # Edge cases
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0, 0.0, 0.0

    matched_pairs = []
    matched_gt    = set()
    matched_pred  = set()

    # For each GT instance, find the best matching prediction
    for gt_id in gt_ids:
        gt_region = (gt_labeled == gt_id)
        best_iou, best_pred = 0.0, None

        for pred_id in pred_ids:
            if pred_id in matched_pred:
                continue
            pred_region  = (pred_labeled == pred_id)
            intersection = np.logical_and(gt_region, pred_region).sum()
            union        = np.logical_or(gt_region, pred_region).sum()
            iou          = intersection / (union + 1e-9)
            if iou > best_iou:
                best_iou, best_pred = iou, pred_id

        if best_iou >= iou_thresh and best_pred is not None:
            matched_pairs.append((gt_id, best_pred, best_iou))
            matched_gt.add(gt_id)
            matched_pred.add(best_pred)

    TP = len(matched_pairs)
    FP = len(pred_ids)  - len(matched_pred)  # predicted but not in GT
    FN = len(gt_ids)    - len(matched_gt)    # in GT but not predicted

    SQ = float(np.mean([iou for _, _, iou in matched_pairs])) if matched_pairs else 0.0
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN + 1e-9)
    PQ = SQ * RQ

    return float(PQ), float(SQ), float(RQ)


# ══════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════
def save_overlay(img: np.ndarray,
                 disk_mask: np.ndarray,
                 fil_labeled: np.ndarray,
                 save_path: str,
                 filament_info: list) -> None:
    """
    Save a colour-coded overlay: original image with each filament
    instance shown in a distinct colour, plus centroid markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rgb = np.stack([img] * 3, axis=-1).astype(float)

    # Colour-code each filament instance
    n_fil   = fil_labeled.max()
    cmap    = plt.cm.hsv(np.linspace(0, 1, max(n_fil, 1), endpoint=False))
    overlay = rgb.copy()

    for i in range(1, n_fil + 1):
        mask_i = (fil_labeled == i)
        col    = cmap[i - 1][:3]
        for c_idx, c_val in enumerate(col):
            overlay[:, :, c_idx][mask_i] = c_val * 200 + 55  # bright colours

    result = (0.5 * rgb + 0.5 * overlay).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="#0d1117")
    ax.imshow(result)

    # Mark centroids
    for f in filament_info[:50]:  # limit to 50 labels for readability
        ax.plot(f["centroid_x"], f["centroid_y"], "+", color="yellow",
                markersize=4, markeredgewidth=0.7, alpha=0.8)

    ax.set_title(f"Segmented Filaments — {n_fil} instances detected",
                 color="white", fontsize=12, pad=8)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION (call this on new test images)
# ══════════════════════════════════════════════════════════════
def segment_image(image_path: str,
                  output_dir: str = "output",
                  save_mask: bool = True,
                  save_overlay_img: bool = True,
                  gt_mask_path: str = None,
                  verbose: bool = True) -> dict:
    """
    Run the full segmentation pipeline on a single H-alpha solar image.

    Args:
        image_path       : path to input .jpeg / .png / .fits image
        output_dir       : where to save masks and overlays
        save_mask        : save binary + labeled mask as PNG
        save_overlay_img : save colour overlay visualisation
        gt_mask_path     : optional path to ground-truth labeled mask (for evaluation)
        verbose          : print progress

    Returns:
        result dict with: n_filaments, dice, PQ, SQ, RQ, filament_info, runtime_s
    """
    t0   = time.time()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # ── Load image ─────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    if verbose:
        print(f"[{stem}] Loaded {img.shape[0]}×{img.shape[1]} image")

    # ── Step 1: Extract disk ───────────────────────────────────
    disk_mask, disk_prop = extract_solar_disk(img)
    disk_px = int((disk_mask > 0).sum())
    if verbose:
        print(f"[{stem}] Disk extracted: {disk_px:,} pixels")

    # ── Steps 3–6: Detect filaments ────────────────────────────
    fil_mask, fil_labeled, fil_info, diff_img = detect_filaments(img, disk_mask)
    n_fil   = fil_labeled.max()
    fil_pct = fil_mask.sum() / (disk_px + 1e-9) * 100
    runtime = time.time() - t0

    if verbose:
        print(f"[{stem}] Detected {n_fil} filament instances "
              f"({fil_pct:.2f}% disk coverage) in {runtime:.2f}s")

    # ── Save masks ─────────────────────────────────────────────
    if save_mask:
        # Binary mask: 0 = background, 255 = any filament
        cv2.imwrite(f"{output_dir}/{stem}_binary_mask.png",
                    (fil_mask.astype(np.uint8)) * 255)
        # Labelled mask: pixel value = filament instance ID
        # (stored as 16-bit so we can have >255 instances)
        cv2.imwrite(f"{output_dir}/{stem}_labeled_mask.png",
                    fil_labeled.astype(np.uint16))
        if verbose:
            print(f"[{stem}] Masks saved to {output_dir}/")

    # ── Save overlay ───────────────────────────────────────────
    if save_overlay_img:
        save_overlay(img, disk_mask, fil_labeled,
                     f"{output_dir}/{stem}_overlay.jpg", fil_info)
        if verbose:
            print(f"[{stem}] Overlay saved")

    # ── Evaluate (if GT provided) ──────────────────────────────
    dice, PQ, SQ, RQ = None, None, None, None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
        gt_binary  = gt > 0
        dice       = compute_dice(fil_mask, gt_binary)
        PQ, SQ, RQ = compute_panoptic_quality(fil_labeled, gt.astype(int))
        if verbose:
            print(f"[{stem}] Dice={dice:.4f}  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}")

    result = {
        "filename":       os.path.basename(image_path),
        "stem":           stem,
        "n_filaments":    int(n_fil),
        "filament_pct":   round(float(fil_pct), 4),
        "disk_px":        disk_px,
        "filament_px":    int(fil_mask.sum()),
        "runtime_s":      round(runtime, 3),
        "dice":           round(dice, 6) if dice is not None else None,
        "PQ":             round(PQ, 6)   if PQ   is not None else None,
        "SQ":             round(SQ, 6)   if SQ   is not None else None,
        "RQ":             round(RQ, 6)   if RQ   is not None else None,
        "filament_info":  fil_info,
    }
    return result


# ══════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════
def run_batch(input_dir: str,
              output_dir: str = "output",
              gt_dir: str = None,
              extensions: tuple = (".jpeg", ".jpg", ".png", ".fits")) -> list:
    """
    Process all solar images in a directory.

    Args:
        input_dir  : folder containing H-alpha images
        output_dir : where to save results
        gt_dir     : optional folder of ground-truth masks
        extensions : accepted file extensions

    Returns:
        list of result dicts (one per image)
    """
    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not files:
        print(f"No images found in {input_dir}")
        return []

    print(f"Found {len(files)} images in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for fname in files:
        image_path = os.path.join(input_dir, fname)
        stem       = os.path.splitext(fname)[0]
        gt_path    = os.path.join(gt_dir, f"{stem}_labeled_mask.png") \
                     if gt_dir else None
        try:
            result = segment_image(
                image_path     = image_path,
                output_dir     = output_dir,
                save_mask      = True,
                save_overlay_img = True,
                gt_mask_path   = gt_path,
                verbose        = True,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on {fname}: {e}")
            continue

    # Save summary JSON
    summary_path = os.path.join(output_dir, "batch_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Images processed : {len(all_results)}")
    print(f"Total filaments  : {sum(r['n_filaments'] for r in all_results)}")
    print(f"Mean per image   : {np.mean([r['n_filaments'] for r in all_results]):.1f}")
    avg_pct = np.mean([r['filament_pct'] for r in all_results])
    print(f"Mean disk coverage: {avg_pct:.2f}%")
    avg_time = np.mean([r['runtime_s'] for r in all_results])
    print(f"Mean runtime     : {avg_time:.2f}s per image")

    dice_scores = [r['dice'] for r in all_results if r['dice'] is not None]
    pq_scores   = [r['PQ']   for r in all_results if r['PQ']   is not None]
    if dice_scores:
        print(f"\nDice Score       : {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    if pq_scores:
        print(f"Panoptic Quality : {np.mean(pq_scores):.4f} ± {np.std(pq_scores):.4f}")

    print(f"\nResults saved to: {summary_path}")
    return all_results


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solar Filament Segmentation — H-alpha Images"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to a single image OR a directory of images")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--gt",     default=None,
                        help="Directory of ground-truth labeled masks (optional)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip saving overlay visualisations (faster)")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Single image mode
        result = segment_image(
            image_path       = args.input,
            output_dir       = args.output,
            save_overlay_img = not args.no_overlay,
            verbose          = True,
        )
        print(f"\nDetected {result['n_filaments']} filaments")
        print(f"Results in: {args.output}/")
    else:
        # Batch mode
        run_batch(
            input_dir  = args.input,
            output_dir = args.output,
            gt_dir     = args.gt,
        )
