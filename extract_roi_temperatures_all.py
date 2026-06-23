import os
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
 
# File and directory settings
SUBJECTS         = [f"S{i}" for i in range(1, 11)]   
CSV_DIR          = "."                               
IMAGE_ROOT       = "Charlotte-ThermalFace"          
COMBINED_OUT     = "all_subjects_roi_temperatures.csv"
SAVE_PER_SUBJECT = True                           
 
# Temperature linear resolution
TLR = 100.0
# Mininum frontal landmarks for frontal image
FRONTAL_MIN_LANDMARKS = 60


# ROI ranges
ROI_FRONTAL = {
    "forehead": [list(range(68, 73)) + [19, 24]],
    "cheek":    [[0, 1, 2, 3, 4, 31], [12, 13, 14, 15, 16, 35]],   
    "eye":      [list(range(36, 42)), list(range(42, 48))],         
    "nose":     [list(range(27, 36))],
    "mouth":    [list(range(48, 68))],
    "chin":     [[5, 6, 7, 8, 9, 10, 11, 57]],
}
 
ROI_SIDE = {
    "forehead": [[40, 41, 42, 9, 14]],
    "cheek":    [[0, 1, 2, 3, 4, 5, 21]],
    "eye":      [list(range(22, 28))],
    "nose":     [list(range(15, 22))],
    "mouth":    [list(range(28, 40))],
    "chin":     [[6, 7, 8, 28]],
}


# ROI order
REGION_ORDER = ["forehead", "cheek", "eye", "nose", "mouth", "chin"]
# Experimental condition information
META_COLS = ["Distance", "RH", "Airflow", "env-temp", "Sensation"]
# All columns
ALL_COLS = ["Subject", "ID"] + REGION_ORDER + META_COLS
 
 

def load_thermal_images(image_path):
    raw = tifffile.imread(image_path)
    return raw.astype(np.float64) / TLR - 273.15
 
 # Not NaN points
def present_points(row, indices):
    pts = []
    for i in indices:
        x, y = row.get(f"x{i}"), row.get(f"y{i}")
        if pd.notna(x) and pd.notna(y):
            pts.append((float(x), float(y)))
    return pts
 
 # Order to draw polygon
def order_by_angle(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
 
 
def polygon_mask(pts, h, w):
    mask_img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_img).polygon(order_by_angle(pts), fill=1)
    return np.array(mask_img, dtype=bool)
 
 # Calculate region means
def region_mean(celsius, row, subregions):
    h, w = celsius.shape
    pooled = np.zeros((h, w), dtype=bool)
    used_any = False
 
    for indices in subregions:
        pts = present_points(row, indices)
        if len(pts) >= 3:
            pooled |= polygon_mask(pts, h, w)
            used_any = True
        elif pts:  # too few points to form a polygon -> patch fallback
            for x, y in pts:
                xi, yi = int(round(x)), int(round(y))
                pooled[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4] = True
                used_any = True
 
    if not used_any or pooled.sum() == 0:
        return np.nan
    return float(celsius[pooled].mean())
 
    
 # Calculate mean for all ROIs
def process_image(row, image_path):
    celsius = load_thermal_images(image_path)
    n_present = sum(pd.notna(row.get(f"x{i}")) for i in range(73))
    roi_map = ROI_FRONTAL if n_present >= FRONTAL_MIN_LANDMARKS else ROI_SIDE
    return {region: region_mean(celsius, row, roi_map[region])
            for region in REGION_ORDER}
 
 

# Find images
def image_dir_for(subject):
    return os.path.join(IMAGE_ROOT, subject)
 
 
def find_image(image_dir, img_id):
    for ext_name in (".tiff",".tif"):
        p = os.path.join(image_dir, f"{img_id}{ext_name}")
        if os.path.exists(p):
            return p
    return None
 
 
 # Do it for all subjects
def main():
    all_records = []
 
    for subj in SUBJECTS:
        csv_path = os.path.join(CSV_DIR, f"{subj}.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
 
        img_dir = image_dir_for(subj)
        df = pd.read_csv(csv_path)
        sub_records, missing = [], 0
 
        for _, row in df.iterrows():
            image_path = find_image(img_dir, row["ID"])
            if image_path is None:
                missing += 1
                continue
            temps = process_image(row, image_path)
            rec = {"Subject": subj, "ID": row["ID"], **temps}
            for c in META_COLS:
                rec[c] = row.get(c)
            sub_records.append(rec)
 
        if SAVE_PER_SUBJECT and sub_records:
            pd.DataFrame(sub_records, columns=ALL_COLS).to_csv(
                f"{subj}_roi_temperatures.csv", index=False)
 
        all_records.extend(sub_records)
        print(f"{subj}: {len(sub_records):5d} images processed "
              f"({missing} skipped - no image file)")
 
    combined = pd.DataFrame(all_records, columns=ALL_COLS)
    combined.to_csv(COMBINED_OUT, index=False)
 
    print(f"\nCombined dataset: {len(combined)} rows -> {COMBINED_OUT}")
    if len(combined):
        print("\nThermal label distribution (Sensation):")
        print(combined["Sensation"].value_counts().sort_index().to_string())
        print("\nRegion temperature means (C):")
        print(combined[REGION_ORDER].mean().round(2).to_string())
 
 
if __name__ == "__main__":
    main()