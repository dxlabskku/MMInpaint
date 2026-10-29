import cv2
import numpy as np
import random
from pathlib import Path
from typing import Optional, Tuple
from scipy import interpolate
import os


class CoarseMaskGenerator:
    """
    Generates two types of masks from fine masks:
      1. Coarse Mask → output_dir/coarse/  <stem>_coarse.png
      2. BBox Mask   → output_dir/bbox/    <stem>_bbox.png

    Each mask type (including the input fine mask) can optionally be
    saved in inverted form as well:
      3. Inverted Fine   → output_dir/fine_inv/   <stem>_fine_inv.png
      4. Inverted Coarse → output_dir/coarse_inv/ <stem>_coarse_inv.png
      5. Inverted BBox   → output_dir/bbox_inv/   <stem>_bbox_inv.png
    """

    def __init__(
        self,
        sampling_interval: int = 30,
        perturbation_range: float = 3.0,
        small_threshold: Optional[int] = None,
        large_threshold: Optional[int] = None,
        dilation_ratio: float = 0.05,
        spline_weight_small: float = 0.2,
        spline_weight_large: float = 0.8,
        bbox_padding_ratio: float = 0.05,
    ):
        self.sampling_interval   = sampling_interval
        self.perturbation_range  = perturbation_range
        self.dilation_ratio      = dilation_ratio
        self.small_threshold     = small_threshold
        self.large_threshold     = large_threshold
        self.spline_weight_small = spline_weight_small
        self.spline_weight_large = spline_weight_large
        self.bbox_padding_ratio  = bbox_padding_ratio

    # ------------------------------------------------------------------
    # Threshold auto-calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        input_dir: str,
        glob_pattern: str = "*.png",
        small_percentile: float = 25.0,
        large_percentile: float = 75.0,
        verbose: bool = True,
    ) -> Tuple[int, int]:
        input_path = Path(input_dir)
        mask_files = list(input_path.glob(glob_pattern))

        if not mask_files:
            print(f"[Debug] Resolved path : {input_path.resolve()}")
            print(f"[Debug] Exists        : {input_path.exists()}")
            print(f"[Debug] Contents      :")
            for f in sorted(input_path.iterdir()):
                print(f"          {f.name}")
            raise FileNotFoundError(
                f"No files matched '{glob_pattern}' in {input_dir}"
            )

        diagonals = []
        for m_path in mask_files:
            raw  = np.fromfile(str(m_path), np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                _, _, w, h = cv2.boundingRect(cnt)
                if w > 0 and h > 0:
                    diagonals.append(np.sqrt(w ** 2 + h ** 2))

        if not diagonals:
            raise ValueError("No valid contours found across the dataset.")

        diagonals = np.array(diagonals)
        small_t   = int(np.percentile(diagonals, small_percentile))
        large_t   = int(np.percentile(diagonals, large_percentile))

        min_gap = 20
        if large_t - small_t < min_gap:
            mid     = (small_t + large_t) // 2
            small_t = max(1, mid - min_gap // 2)
            large_t = mid + min_gap // 2

        self.small_threshold = small_t
        self.large_threshold = large_t

        if verbose:
            print(f"[Calibration] {len(mask_files)} files / {len(diagonals)} contours")
            print(f"  Diagonal  min={diagonals.min():.1f}  max={diagonals.max():.1f}"
                  f"  mean={diagonals.mean():.1f}  std={diagonals.std():.1f}")
            print(f"  P{small_percentile:.0f} = {small_t}  →  small_threshold  "
                  f"(spline weight = {self.spline_weight_small:.0%})")
            print(f"  P{large_percentile:.0f} = {large_t}  →  large_threshold  "
                  f"(spline weight = {self.spline_weight_large:.0%})")

        return small_t, large_t

    def _check_thresholds(self):
        if self.small_threshold is None or self.large_threshold is None:
            raise RuntimeError(
                "Thresholds not set. Pass them to __init__ "
                "or call calibrate() first."
            )

    # ------------------------------------------------------------------
    # Inversion utility
    # ------------------------------------------------------------------
    @staticmethod
    def invert(mask: np.ndarray) -> np.ndarray:
        """
        Bitwise-inverts a binary mask: white↔black.
        Accepts grayscale (H,W) or BGR (H,W,3) arrays.
        """
        return cv2.bitwise_not(mask)

    # ------------------------------------------------------------------
    # Size-aware strategy weight
    # ------------------------------------------------------------------
    def _spline_weight_for(self, mask: np.ndarray) -> float:
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.spline_weight_small

        largest    = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest)
        diag       = np.sqrt(w ** 2 + h ** 2)

        t = np.clip(
            (diag - self.small_threshold) / (self.large_threshold - self.small_threshold),
            0.0, 1.0,
        )
        return self.spline_weight_small + t * (self.spline_weight_large - self.spline_weight_small)

    # ------------------------------------------------------------------
    # Strategy A: Spline contour perturbation
    # ------------------------------------------------------------------
    def _strategy_spline(self, mask: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask

        result = np.zeros_like(mask)
        for cnt_raw in contours:
            cnt        = cnt_raw.reshape(-1, 2)
            _, _, w, h = cv2.boundingRect(cnt_raw)
            diag       = np.sqrt(w ** 2 + h ** 2)

            ratio = np.clip(
                (diag - self.small_threshold) / (self.large_threshold - self.small_threshold),
                0.0, 1.0,
            )
            min_interval      = max(4, self.sampling_interval // 6)
            adaptive_interval = int(min_interval + ratio * (self.sampling_interval - min_interval))
            min_perturb       = max(0.5, self.perturbation_range * 0.05)
            adaptive_perturb  = min_perturb + ratio * (self.perturbation_range - min_perturb)

            if len(cnt) < adaptive_interval * 2:
                cv2.drawContours(result, [cnt_raw], -1, 255, -1)
                continue

            pts  = cnt[np.arange(0, len(cnt), adaptive_interval)].astype(float)
            pts += np.random.uniform(-adaptive_perturb, adaptive_perturb, pts.shape)
            pts  = np.vstack([pts, pts[0]])

            try:
                tck, _  = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
                new_pts = interpolate.splev(np.linspace(0, 1, 300), tck)
                cv2.fillPoly(result,
                             [np.stack(new_pts, axis=1).astype(np.int32)], 255)
            except Exception:
                cv2.drawContours(result, [cnt_raw], -1, 255, -1)

        return result

    # ------------------------------------------------------------------
    # Strategy B: Morphological dilation + Gaussian blur
    # ------------------------------------------------------------------
    def _strategy_dilation(self, mask: np.ndarray) -> np.ndarray:
        result      = mask.copy()
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            k_size, b_size = 11, 15
        else:
            _, _, w, h = cv2.boundingRect(np.concatenate(contours))
            scale  = np.sqrt(w * h)
            k_size = max(3, int(scale * self.dilation_ratio) | 1)
            b_size = max(3, int(k_size * 1.2) | 1)

        kernel = np.ones((k_size, k_size), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        result = cv2.GaussianBlur(result, (b_size, b_size), 0)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        return result

    # ------------------------------------------------------------------
    # Strategy C: BBox mask
    # ------------------------------------------------------------------
    def _strategy_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Draws a padded solid rectangle for each contour's bounding box."""
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        H, W   = mask.shape[:2]
        result = np.zeros_like(mask)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = int(np.sqrt(w * h) * self.bbox_padding_ratio)
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(W, x + w + pad)
            y2  = min(H, y + h + pad)
            cv2.rectangle(result, (x1, y1), (x2, y2), 255, -1)

        return result

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------
    def generate_coarse(self, mask: np.ndarray) -> np.ndarray:
        """Size-aware random strategy → coarse mask."""
        self._check_thresholds()
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        spline_w = self._spline_weight_for(mask)
        strategy = random.choices(["spline", "dilation"],
                                  weights=[spline_w, 1.0 - spline_w], k=1)[0]
        return (self._strategy_spline(mask)
                if strategy == "spline"
                else self._strategy_dilation(mask))

    def generate_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Padded bounding-box mask."""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return self._strategy_bbox(mask)

    # ------------------------------------------------------------------
    # Internal save helper
    # ------------------------------------------------------------------
    @staticmethod
    def _save(directory: Path, filename: str, img: np.ndarray):
        """Encodes and writes a PNG, supporting non-ASCII paths."""
        _, enc = cv2.imencode(".png", img)
        enc.tofile(str(directory / filename))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        input_dir: str,
        output_dir: str,
        glob_pattern: str = "*.png",
        auto_calibrate: bool = True,
        small_percentile: float = 25.0,
        large_percentile: float = 75.0,
        # --- output switches ---
        save_coarse: bool = True,
        save_bbox: bool = True,
        save_fine_inv: bool = True,
        save_coarse_inv: bool = True,
        save_bbox_inv: bool = True,
    ):
        """
        Output structure (each subfolder created only when its switch is ON):

            output_dir/
              coarse/      <stem>_coarse.png
              bbox/        <stem>_bbox.png
              fine_inv/    <stem>_fine_inv.png
              coarse_inv/  <stem>_coarse_inv.png
              bbox_inv/    <stem>_bbox_inv.png
        """
        if auto_calibrate:
            self.calibrate(input_dir, glob_pattern, small_percentile, large_percentile)
        else:
            self._check_thresholds()

        input_path  = Path(input_dir)
        output_path = Path(output_dir)

        dirs = {}
        spec = {
            "coarse":     save_coarse,
            "bbox":       save_bbox,
            "fine_inv":   save_fine_inv,
            "coarse_inv": save_coarse_inv,
            "bbox_inv":   save_bbox_inv,
        }
        for name, enabled in spec.items():
            if enabled:
                d = output_path / name
                d.mkdir(parents=True, exist_ok=True)
                dirs[name] = d

        mask_files = list(input_path.glob(glob_pattern))
        print(f"\n[Pipeline] {len(mask_files)} files")
        print(f"  Active outputs: {[k for k, v in spec.items() if v]}")

        for m_path in mask_files:
            raw       = np.fromfile(str(m_path), np.uint8)
            fine_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if fine_mask is None:
                print(f"  [SKIP] {m_path.name}")
                continue

            stem   = m_path.stem
            prefix = stem.replace("_mask_fine", "")

            coarse_mask = self.generate_coarse(fine_mask) if (save_coarse or save_coarse_inv) else None
            bbox_mask   = self.generate_bbox(fine_mask)   if (save_bbox   or save_bbox_inv)   else None

            if save_coarse:
                self._save(dirs["coarse"], f"{prefix}_mask_coarse.png", coarse_mask)
            if save_bbox:
                self._save(dirs["bbox"], f"{prefix}_mask_bbox.png", bbox_mask)
            if save_fine_inv:
                self._save(dirs["fine_inv"],   f"{prefix}_mask_fine_inv.png",   self.invert(fine_mask))
            if save_coarse_inv:
                self._save(dirs["coarse_inv"], f"{prefix}_mask_coarse_inv.png", self.invert(coarse_mask))
            if save_bbox_inv:
                self._save(dirs["bbox_inv"],   f"{prefix}_mask_bbox_inv.png",   self.invert(bbox_mask))

        print(f"[Pipeline] Done → {output_path}")
        for name, d in dirs.items():
            print(f"  {name:12s} → {d}")


# ------------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------------
from PIL import Image

def highlight_mask_region(img_path, mask_path, output_path, alpha=0.6, color=(0, 0, 255)):
    """Overlays the mask's white region as a semi-transparent colored highlight on the original image."""
    image = Image.open(img_path).convert("RGBA")
    mask  = Image.open(mask_path).convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    binary_mask = mask.point(lambda p: 255 if p > 128 else 0)

    overlay = Image.new("RGBA", image.size, (*color, 0))
    overlay_alpha = binary_mask.point(lambda p: int(255 * alpha) if p > 0 else 0)
    overlay.putalpha(overlay_alpha)

    result = Image.alpha_composite(image, overlay)
    result.save(output_path)
    print(f"  [OK] {Path(output_path).name}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --- Mask generation ---
    input_dir  = 'coco/mask_fine'
    output_dir = 'coco/mask_generated_output'

    generator = CoarseMaskGenerator(
        sampling_interval   = 30,
        perturbation_range  = 3.0,
        dilation_ratio      = 0.05,
        spline_weight_small = 0.2,
        spline_weight_large = 0.8,
        bbox_padding_ratio  = 0.05,
    )

    generator.run_pipeline(
        input_dir        = input_dir,
        output_dir       = output_dir,
        glob_pattern     = "*.png",
        auto_calibrate   = True,
        small_percentile = 25.0,
        large_percentile = 75.0,
        save_coarse      = True,
        save_bbox        = True,
        save_fine_inv    = True,
        save_coarse_inv  = True,
        save_bbox_inv    = True,
    )

    # --- Mask visualization ---
    img_dir    = Path('coco/image')
    mask_dir   = Path('coco/mask_coarse')
    vis_output = Path('coco/mask_generated_view')
    vis_output.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    print(f"\nFound {len(img_files)} images.\n")

    for img_path in img_files:
        stem      = img_path.stem.removesuffix("_mask")
        mask_path = mask_dir / f"{stem}_mask_coarse.png"

        if not mask_path.exists():
            print(f"  [SKIP] {stem} — mask not found: {mask_path.name}")
            continue

        out_path = vis_output / f"{stem}_coarse_preview.png"
        highlight_mask_region(
            img_path    = str(img_path),
            mask_path   = str(mask_path),
            output_path = str(out_path),
            alpha       = 0.6,
        )

    print(f"\nDone → {vis_output}")
