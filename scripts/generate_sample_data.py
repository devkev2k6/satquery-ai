#!/usr/bin/env python3
"""
SatQuery AI - Synthetic & Sample Data Generator
================================================
Prepares small, manageable sample subsets (< 500MB total) for:
  - data/bigearthnet/{images, labels.csv}
  - data/vrsbench/{images, annotations.json}
  - data/rsvqa/{images, qa_pairs.json}
  - data/cdvqa/{image_pairs/, qa_pairs.json}
  - data/sample/{optical/, sar/, pairs/} (synthetic fallback data)

Can run with pure Python standard library (struct + zlib PNG encoder)
or with PIL / numpy if available.
"""

import os
import sys
import zlib
import struct
import random
import json
import csv
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BIGEARTHNET_DIR = DATA_DIR / "bigearthnet"
VRSBENCH_DIR = DATA_DIR / "vrsbench"
RSVQA_DIR = DATA_DIR / "rsvqa"
CDVQA_DIR = DATA_DIR / "cdvqa"
SAMPLE_DIR = DATA_DIR / "sample"


def write_png(filepath: Path, width: int, height: int, pixels: list):
    """
    Writes a raw 24-bit RGB PNG file using pure Python (no external dependencies).
    pixels: list of (r, g, b) tuples, length width * height, row by row.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # Filter type 0 (None)
        row_offset = y * width
        for x in range(width):
            r, g, b = pixels[row_offset + x]
            raw_data.append(max(0, min(255, int(r))))
            raw_data.append(max(0, min(255, int(g))))
            raw_data.append(max(0, min(255, int(b))))

    compressed = zlib.compress(bytes(raw_data), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    with open(filepath, "wb") as f:
        # PNG Signature
        f.write(b"\x89PNG\r\n\x1a\n")
        # IHDR: width(4), height(4), bit_depth(1), color_type(1, RGB=2), compression(1), filter(1), interlace(1)
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        f.write(chunk(b"IHDR", ihdr_data))
        # IDAT
        f.write(chunk(b"IDAT", compressed))
        # IEND
        f.write(chunk(b"IEND", b""))


def generate_optical_texture(width=128, height=128, terrain_type="forest", seed=42):
    """Generates synthetic optical satellite reflectance textures."""
    random.seed(seed)
    pixels = []
    
    # Palette parameters per terrain
    if terrain_type == "forest":
        base_r, base_g, base_b = 35, 95, 45
    elif terrain_type == "water":
        base_r, base_g, base_b = 25, 60, 110
    elif terrain_type == "urban":
        base_r, base_g, base_b = 130, 130, 135
    elif terrain_type == "agriculture":
        base_r, base_g, base_b = 145, 140, 55
    elif terrain_type == "desert":
        base_r, base_g, base_b = 190, 160, 115
    elif terrain_type == "cloudy":
        base_r, base_g, base_b = 220, 225, 230
    else:
        base_r, base_g, base_b = 80, 110, 70

    for y in range(height):
        for x in range(width):
            # Gradient + noise modulation
            grad = (x + y) / (width + height) * 20
            noise = random.randint(-25, 25)
            
            # Urban grid effect
            grid_mod = 0
            if terrain_type == "urban" and (x % 16 < 3 or y % 16 < 3):
                grid_mod = -40  # Darker road grid
            
            # Agricultural patch effect
            patch_mod = 0
            if terrain_type == "agriculture":
                patch_id = ((x // 32) + (y // 32)) % 3
                patch_mod = patch_id * 15 - 15

            r = base_r + grad + noise + grid_mod + patch_mod
            g = base_g + grad + noise + grid_mod + patch_mod
            b = base_b + grad + noise + grid_mod + patch_mod

            pixels.append((r, g, b))
    return pixels


def generate_sar_texture(width=128, height=128, surface="land", seed=42):
    """
    Generates synthetic Synthetic Aperture Radar (SAR) backscatter textures
    with characteristic multiplicative Rayleigh speckle noise and high-return scatterers.
    """
    random.seed(seed)
    pixels = []

    mean_val = 30 if surface == "water" else (140 if surface == "urban_scatter" else 85)

    for y in range(height):
        for x in range(width):
            # Multiplicative speckle noise (Rayleigh-like via sum of squares of gaussians)
            u1 = random.uniform(0.001, 0.999)
            u2 = random.uniform(0.001, 0.999)
            # Box-muller transform
            z = (-2 * (u1 ** 0.5) * 0.5 + 1.0)
            speckle = random.choice([0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
            
            val = mean_val * speckle + random.randint(-15, 15)

            # High-intensity point targets (corner reflectors, metallic roofs)
            if surface == "urban_scatter" and random.random() < 0.03:
                val = 245 + random.randint(0, 10)
            elif surface == "water" and random.random() < 0.005:
                val = 210  # Floating vessel/buoy

            val = max(10, min(255, int(val)))
            pixels.append((val, val, val))  # SAR single polarization grayscale as RGB
    return pixels


def create_sample_dataset():
    """Populates data/sample/ with optical, sar, and multitemporal pairs."""
    print("[*] Generating data/sample/ synthetic fallback data...")
    optical_dir = SAMPLE_DIR / "optical"
    sar_dir = SAMPLE_DIR / "sar"
    pairs_dir = SAMPLE_DIR / "pairs"

    for d in [optical_dir, sar_dir, pairs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    terrains = ["forest", "water", "urban", "agriculture", "desert", "cloudy"]
    
    # 1. Optical tiles
    for i in range(1, 16):
        t_type = terrains[(i - 1) % len(terrains)]
        img_name = f"sample_optical_{i:03d}.png"
        meta_name = f"sample_optical_{i:03d}.json"
        
        px = generate_optical_texture(128, 128, terrain_type=t_type, seed=100 + i)
        write_png(optical_dir / img_name, 128, 128, px)
        
        meta = {
            "image_id": f"sample_optical_{i:03d}",
            "modality": "optical",
            "terrain": t_type,
            "labels": [t_type, "synthetic_satellite"],
            "question": f"What terrain type is depicted in this optical satellite observation?",
            "answer": f"Simulated {t_type} land cover.",
        }
        with open(optical_dir / meta_name, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # 2. SAR tiles
    surfaces = ["land", "water", "urban_scatter"]
    for i in range(1, 16):
        s_type = surfaces[(i - 1) % len(surfaces)]
        img_name = f"sample_sar_{i:03d}.png"
        meta_name = f"sample_sar_{i:03d}.json"
        
        px = generate_sar_texture(128, 128, surface=s_type, seed=200 + i)
        write_png(sar_dir / img_name, 128, 128, px)

        meta = {
            "image_id": f"sample_sar_{i:03d}",
            "modality": "sar",
            "surface_type": s_type,
            "labels": [f"sar_{s_type}", "radar_backscatter"],
            "question": "What are the radar backscatter characteristics of this SAR image?",
            "answer": f"Sentinel-1 style simulated backscatter indicating {s_type}.",
        }
        with open(sar_dir / meta_name, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # 3. Multitemporal Change Pairs
    changes = [
        ("urban_expansion", "forest", "urban", "New residential buildings and roads appeared."),
        ("deforestation", "forest", "agriculture", "Woodland area was cleared for arable land."),
        ("reservoir_filling", "agriculture", "water", "Low-lying area was flooded forming a water reservoir."),
        ("industrial_development", "desert", "urban", "Industrial units and storage facilities were erected."),
        ("vegetation_recovery", "desert", "forest", "Afforestation and vegetation regrowth occurred."),
    ]
    for i in range(1, 11):
        ch_type, t1_type, t2_type, desc = changes[(i - 1) % len(changes)]
        t1_img = f"sample_pair_{i:03d}_t1.png"
        t2_img = f"sample_pair_{i:03d}_t2.png"
        meta_name = f"sample_pair_{i:03d}_meta.json"

        px_t1 = generate_optical_texture(128, 128, terrain_type=t1_type, seed=300 + i)
        px_t2 = generate_optical_texture(128, 128, terrain_type=t2_type, seed=400 + i)

        write_png(pairs_dir / t1_img, 128, 128, px_t1)
        write_png(pairs_dir / t2_img, 128, 128, px_t2)

        meta = {
            "pair_id": f"sample_pair_{i:03d}",
            "image_t1": t1_img,
            "image_t2": t2_img,
            "change_type": ch_type,
            "question": "What environmental or infrastructure change occurred between Time 1 and Time 2?",
            "answer": desc,
        }
        with open(pairs_dir / meta_name, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("[+] Created data/sample/ (15 optical, 15 SAR, 10 pairs).")


def create_bigearthnet_sample():
    """Populates data/bigearthnet/ with images and labels.csv."""
    print("[*] Generating data/bigearthnet/ sample data...")
    images_dir = BIGEARTHNET_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_file = BIGEARTHNET_DIR / "labels.csv"

    # BigEarthNet 19 CLC classes
    clc_presets = [
        (["Coniferous forest", "Mixed forest"], "optical", "train"),
        (["Discontinuous urban fabric", "Industrial or commercial units"], "optical", "train"),
        (["Arable land", "Pastures"], "optical", "train"),
        (["Water bodies", "Inland marshes"], "optical", "train"),
        (["Broad-leaved forest", "Transitional woodland-shrub"], "optical", "val"),
        (["Complex cultivation patterns", "Land principally occupied by agriculture"], "optical", "val"),
        (["Continuous urban fabric", "Road and rail networks"], "optical", "test"),
        (["Pastures", "Natural grasslands"], "optical", "test"),
        (["Water courses", "Water bodies"], "sar", "train"),
        (["Urban fabric", "Industrial units"], "sar", "train"),
        (["Coniferous forest"], "sar", "val"),
        (["Arable land"], "sar", "test"),
    ]

    rows = [["image_file", "labels", "modality", "split", "caption"]]

    for idx, (labels, mod, split) in enumerate(clc_presets, start=1):
        filename = f"S2A_MSIL2A_20170717T113321_{idx:02d}.png"
        img_path = images_dir / filename

        if mod == "optical":
            t_type = "forest" if "forest" in labels[0].lower() else (
                "water" if "water" in labels[0].lower() else (
                    "urban" if "urban" in labels[0].lower() else "agriculture"
                )
            )
            px = generate_optical_texture(128, 128, terrain_type=t_type, seed=500 + idx)
        else:
            s_type = "water" if "water" in labels[0].lower() else (
                "urban_scatter" if "urban" in labels[0].lower() else "land"
            )
            px = generate_sar_texture(128, 128, surface=s_type, seed=600 + idx)

        write_png(img_path, 128, 128, px)
        label_str = "|".join(labels)
        caption = f"A {mod} satellite observation showing {', '.join(labels)}."
        rows.append([filename, label_str, mod, split, caption])

    with open(labels_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[+] Created data/bigearthnet/ ({len(clc_presets)} patches, labels.csv).")


def create_vrsbench_sample():
    """Populates data/vrsbench/ with images and annotations.json."""
    print("[*] Generating data/vrsbench/ sample data...")
    images_dir = VRSBENCH_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annot_file = VRSBENCH_DIR / "annotations.json"

    presets = [
        {
            "id": "vrs_0001",
            "terrain": "urban",
            "caption": "A high-resolution satellite scene showing an urban downtown intersection with commercial high-rises and road networks.",
            "qa": [
                {"question": "What is the primary function of the complex in this image?", "answer": "Commercial urban center"},
                {"question": "Are there road networks visible surrounding the buildings?", "answer": "Yes, major paved arterial roads are present."}
            ],
            "grounding": [
                {"object": "commercial building", "bbox": [20, 20, 75, 75]},
                {"object": "road intersection", "bbox": [10, 80, 118, 118]}
            ]
        },
        {
            "id": "vrs_0002",
            "terrain": "water",
            "caption": "Aerial view of a coastal marina with docking berths, piers, and calm blue harbor waters.",
            "qa": [
                {"question": "What maritime infrastructure is present?", "answer": "A harbor with docking piers and berths."},
                {"question": "Is the water surface calm or rough?", "answer": "The harbor water is calm."}
            ],
            "grounding": [
                {"object": "pier dock", "bbox": [30, 40, 90, 80]}
            ]
        },
        {
            "id": "vrs_0003",
            "terrain": "agriculture",
            "caption": "A patchwork of agricultural fields showing geometric crop boundaries and irrigation channels.",
            "qa": [
                {"question": "How are the agricultural parcels arranged?", "answer": "In structured geometric rectangular parcels."},
                {"question": "Are there active cultivated plots?", "answer": "Yes, multiple vegetated and fallow plots are visible."}
            ],
            "grounding": [
                {"object": "cultivated field", "bbox": [15, 15, 60, 60]},
                {"object": "fallow field", "bbox": [65, 65, 115, 115]}
            ]
        },
        {
            "id": "vrs_0004",
            "terrain": "forest",
            "caption": "Dense canopy of coniferous forest located adjacent to a river stream.",
            "qa": [
                {"question": "What dominant vegetation type is visible?", "answer": "Coniferous forest canopy."},
                {"question": "Is there any hydrographic feature nearby?", "answer": "Yes, a river stream borders the woodland."}
            ],
            "grounding": [
                {"object": "river stream", "bbox": [5, 45, 120, 75]}
            ]
        },
        {
            "id": "vrs_0005",
            "terrain": "desert",
            "caption": "Arid terrain featuring sand dunes, dry gullies, and sparse desert vegetation.",
            "qa": [
                {"question": "What type of landscape is represented?", "answer": "Arid desert landscape with dune ridges."},
                {"question": "Is dense urban infrastructure present?", "answer": "No, the area is remote and uninhabited."}
            ],
            "grounding": [
                {"object": "sand dune ridge", "bbox": [25, 30, 100, 70]}
            ]
        },
    ]

    annotations = []
    for idx, item in enumerate(presets, start=1):
        filename = f"{item['id']}.png"
        img_path = images_dir / filename
        px = generate_optical_texture(128, 128, terrain_type=item["terrain"], seed=700 + idx)
        write_png(img_path, 128, 128, px)

        annotations.append({
            "image_id": item["id"],
            "image_file": filename,
            "caption": item["caption"],
            "qa_pairs": item["qa"],
            "grounding": item["grounding"],
        })

    with open(annot_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"[+] Created data/vrsbench/ ({len(presets)} images, annotations.json).")


def create_rsvqa_sample():
    """Populates data/rsvqa/ with images and qa_pairs.json."""
    print("[*] Generating data/rsvqa/ sample data...")
    images_dir = RSVQA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    qa_file = RSVQA_DIR / "qa_pairs.json"

    items = [
        {"id": 101, "terrain": "urban", "q": "Are there buildings present in the scene?", "a": "Yes", "type": "presence"},
        {"id": 102, "terrain": "water", "q": "Is there any body of water visible?", "a": "Yes", "type": "presence"},
        {"id": 103, "terrain": "forest", "q": "Is there an airport located in this tile?", "a": "No", "type": "presence"},
        {"id": 104, "terrain": "urban", "q": "How many major building clusters are visible?", "a": "4", "type": "count"},
        {"id": 105, "terrain": "agriculture", "q": "How many agricultural parcels are present?", "a": "6", "type": "count"},
        {"id": 106, "terrain": "forest", "q": "Is the forested area larger than the open grassland area?", "a": "Yes", "type": "comparison"},
        {"id": 107, "terrain": "urban", "q": "Does urban impervious surface cover more than 50% of the tile?", "a": "Yes", "type": "area"},
        {"id": 108, "terrain": "desert", "q": "Is agricultural vegetation present in this arid region?", "a": "No", "type": "presence"},
    ]

    qa_list = []
    for item in items:
        filename = f"rsvqa_{item['id']:04d}.png"
        img_path = images_dir / filename
        px = generate_optical_texture(128, 128, terrain_type=item["terrain"], seed=800 + item["id"])
        write_png(img_path, 128, 128, px)

        qa_list.append({
            "id": item["id"],
            "image_file": filename,
            "question": item["q"],
            "answer": item["a"],
            "type": item["type"],
        })

    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, indent=2)

    print(f"[+] Created data/rsvqa/ ({len(items)} images, qa_pairs.json).")


def create_cdvqa_sample():
    """Populates data/cdvqa/ with image_pairs/ and qa_pairs.json."""
    print("[*] Generating data/cdvqa/ sample data...")
    pairs_dir = CDVQA_DIR / "image_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    qa_file = CDVQA_DIR / "qa_pairs.json"

    scenarios = [
        {
            "pair_id": "cdvqa_001",
            "t1": "forest",
            "t2": "urban",
            "q": "Did new building structures appear between Time 1 and Time 2?",
            "a": "Yes, urban expansion replaced woodland areas.",
            "change_type": "urban_expansion"
        },
        {
            "pair_id": "cdvqa_002",
            "t1": "agriculture",
            "t2": "water",
            "q": "What hydrological change occurred between T1 and T2?",
            "a": "Inundation of farmland forming an open water reservoir.",
            "change_type": "flooding"
        },
        {
            "pair_id": "cdvqa_003",
            "t1": "forest",
            "t2": "agriculture",
            "q": "Was there deforestation observed between T1 and T2?",
            "a": "Yes, trees were cleared to establish agricultural plots.",
            "change_type": "deforestation"
        },
        {
            "pair_id": "cdvqa_004",
            "t1": "desert",
            "t2": "urban",
            "q": "What facility was built in this arid zone?",
            "a": "Industrial structures and paved access roads.",
            "change_type": "construction"
        },
        {
            "pair_id": "cdvqa_005",
            "t1": "urban",
            "t2": "urban",
            "q": "Did the road network change between the two satellite passes?",
            "a": "No significant changes occurred to the road layout.",
            "change_type": "no_change"
        },
    ]

    qa_list = []
    for idx, sc in enumerate(scenarios, start=1):
        t1_filename = f"{sc['pair_id']}_t1.png"
        t2_filename = f"{sc['pair_id']}_t2.png"

        px_t1 = generate_optical_texture(128, 128, terrain_type=sc["t1"], seed=900 + idx)
        px_t2 = generate_optical_texture(128, 128, terrain_type=sc["t2"], seed=950 + idx)

        write_png(pairs_dir / t1_filename, 128, 128, px_t1)
        write_png(pairs_dir / t2_filename, 128, 128, px_t2)

        qa_list.append({
            "pair_id": sc["pair_id"],
            "image_t1": t1_filename,
            "image_t2": t2_filename,
            "question": sc["q"],
            "answer": sc["a"],
            "change_type": sc["change_type"],
        })

    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, indent=2)

    print(f"[+] Created data/cdvqa/ ({len(scenarios)} bi-temporal pairs, qa_pairs.json).")


def main():
    print("=" * 60)
    print("SatQuery AI - Generating Sample & Benchmark Datasets")
    print("=" * 60)
    create_sample_dataset()
    create_bigearthnet_sample()
    create_vrsbench_sample()
    create_rsvqa_sample()
    create_cdvqa_sample()
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
