"""
Solar Filament Segmentation & Tracking — Streamlit App
======================================================
Interactive web UI for H-alpha solar filament instance segmentation
and multi-frame tracking.
"""

import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import time
import io
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filament_segmentation import (
    extract_solar_disk,
    detect_filaments,
    compute_dice,
    compute_panoptic_quality,
    save_overlay,
)

from filament_tracking import (
    FilamentTracker,
    labeled_mask_to_instances,
    tracks_to_summary,
    build_track_overlay,
    color_for_track,
)

# ──────────────────────────────────────────────────────────────
# Page config & style
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Filament Segmentation & Tracking",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stButton>button {
        background-color: #238636; color: white;
        border-radius: 6px; border: none;
        padding: 0.5rem 1.2rem; font-weight: 600;
    }
    .stButton>button:hover { background-color: #2ea043; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("☀️ Solar Filaments")
    mode = st.radio(
        "Mode",
        ["Single Image", "Time Series Tracking"],
        index=0,
        help="Single-image segmentation or multi-frame tracking",
    )
    st.markdown("---")

    st.subheader("⚙️ Detection Parameters")
    contrast_sigma = st.slider("Contrast σ", 0.3, 2.0, 0.8, 0.1)
    min_area_px = st.slider("Min area (px)", 20, 300, 50, 10)
    min_elongation = st.slider("Min elongation", 1.0, 5.0, 2.0, 0.1)
    large_blob_area = st.slider("Large blob override (px)", 200, 2000, 500, 50)

    if mode == "Time Series Tracking":
        st.markdown("---")
        st.subheader("🔗 Tracking Parameters")
        max_centroid_dist = st.slider(
            "Max centroid distance (px)", 10, 120, 40, 5,
            help="Maximum pixel distance for associating detections across frames",
        )
        max_frames_lost = st.slider(
            "Max frames lost", 1, 10, 3,
            help="How many consecutive frames a track may disappear before it is closed",
        )
        use_diff_rot = st.checkbox("Differential rotation correction", value=True)
        dt_minutes = st.number_input(
            "Cadence (minutes between frames)",
            min_value=1.0, max_value=1440.0, value=60.0, step=1.0,
            help="Used for differential-rotation prediction",
        )
        store_masks = st.checkbox(
            "Store instance masks (needed for IoU matching & overlays)",
            value=True,
        )

    st.markdown("---")
    st.caption(
        "Pipeline: Disk extraction → Limb darkening → Local contrast → "
        "Adaptive threshold → Morphology → Instance labelling"
        + (" → Multi-frame association" if mode == "Time Series Tracking" else "")
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_gray(uploaded) -> np.ndarray:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    uploaded.seek(0)
    return img


def run_segmentation(img, params):
    disk_mask, disk_prop = extract_solar_disk(img)
    fil_mask, fil_labeled, fil_info, diff_img = detect_filaments(
        img,
        disk_mask,
        contrast_sigma=params["contrast_sigma"],
        min_area_px=params["min_area_px"],
        min_elongation=params["min_elongation"],
        large_blob_area=params["large_blob_area"],
    )
    return disk_mask, fil_mask, fil_labeled, fil_info, diff_img


def make_overlay(img, fil_labeled, fil_info):
    rgb = np.stack([img] * 3, axis=-1).astype(float)
    n_fil = max(int(fil_labeled.max()), 1)
    cmap = plt.cm.hsv(np.linspace(0, 1, n_fil, endpoint=False))
    overlay = rgb.copy()
    for i in range(1, n_fil + 1):
        mask_i = fil_labeled == i
        col = cmap[i - 1][:3]
        for c in range(3):
            overlay[:, :, c][mask_i] = col[c] * 200 + 55
    return (0.5 * rgb + 0.5 * overlay).astype(np.uint8)


# =====================================================================
# MODE 1 — Single Image
# =====================================================================
if mode == "Single Image":
    st.title("Solar Filament Instance Segmentation")
    st.markdown(
        "Upload an H-alpha chromospheric image to detect and segment individual solar filaments."
    )

    uploaded_file = st.file_uploader(
        "Choose an H-alpha image (JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    use_gt = st.checkbox("Provide ground-truth mask for metrics", value=False)
    gt_file = None
    if use_gt:
        gt_file = st.file_uploader(
            "Ground-truth labeled mask",
            type=["png", "jpg", "jpeg"],
            key="gt_uploader",
        )

    if uploaded_file is not None:
        img = load_gray(uploaded_file)
        if img is None:
            st.error("Could not decode the image.")
            st.stop()

        st.success(f"Loaded **{uploaded_file.name}**  ·  `{img.shape[0]}×{img.shape[1]}`")
        st.image(img, caption="Input H-alpha image", use_container_width=True, clamp=True)

        if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
            params = dict(
                contrast_sigma=contrast_sigma,
                min_area_px=min_area_px,
                min_elongation=min_elongation,
                large_blob_area=large_blob_area,
            )
            with st.spinner("Running pipeline…"):
                t0 = time.time()
                disk_mask, fil_mask, fil_labeled, fil_info, diff_img = run_segmentation(img, params)
                n_fil = int(fil_labeled.max())
                disk_px = int((disk_mask > 0).sum())
                fil_pct = fil_mask.sum() / (disk_px + 1e-9) * 100
                runtime = time.time() - t0

                dice = PQ = SQ = RQ = None
                if use_gt and gt_file is not None:
                    gt = load_gray(gt_file)
                    if gt is not None:
                        if gt.shape[:2] != img.shape[:2]:
                            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        dice = compute_dice(fil_mask, gt > 0)
                        PQ, SQ, RQ = compute_panoptic_quality(fil_labeled, gt.astype(int))

            st.markdown("---")
            st.subheader("Results Summary")
            mcols = st.columns(5)
            mcols[0].metric("Filaments", n_fil)
            mcols[1].metric("Disk coverage", f"{fil_pct:.2f}%")
            mcols[2].metric("Filament px", f"{int(fil_mask.sum()):,}")
            mcols[3].metric("Runtime", f"{runtime:.2f} s")
            mcols[4].metric("Dice", f"{dice:.4f}" if dice is not None else "—")

            if PQ is not None:
                pq_cols = st.columns(3)
                pq_cols[0].metric("PQ", f"{PQ:.4f}")
                pq_cols[1].metric("SQ", f"{SQ:.4f}")
                pq_cols[2].metric("RQ", f"{RQ:.4f}")

            # Visuals
            st.markdown("---")
            st.subheader("Visualizations")
            result_overlay = make_overlay(img, fil_labeled, fil_info)
            diff_norm = np.clip(diff_img, 0, None)
            if diff_norm.max() > 0:
                diff_norm = (diff_norm / diff_norm.max() * 255).astype(np.uint8)

            vcols = st.columns(3)
            vcols[0].image(img, caption="Original", use_container_width=True, clamp=True)
            vcols[1].image(diff_norm, caption="Local contrast", use_container_width=True, clamp=True)
            vcols[2].image(result_overlay, caption=f"Overlay ({n_fil} filaments)", use_container_width=True)

            mcols2 = st.columns(2)
            mcols2[0].image((fil_mask.astype(np.uint8) * 255), caption="Binary mask", use_container_width=True, clamp=True)
            mcols2[1].image(np.where(disk_mask > 0, img, 0), caption="Solar disk", use_container_width=True, clamp=True)

            if fil_info:
                st.markdown("---")
                st.subheader(f"Detected Filaments ({len(fil_info)})")
                df = pd.DataFrame(sorted(fil_info, key=lambda x: x["area_px"], reverse=True))
                df = df[["label", "area_px", "length_px", "width_px", "elongation",
                         "centroid_y", "centroid_x", "orientation"]]
                df.columns = ["ID", "Area", "Length", "Width", "Elongation",
                              "Cy", "Cx", "Orientation"]
                for c in ["Length", "Width", "Elongation", "Cy", "Cx", "Orientation"]:
                    df[c] = df[c].round(2)
                st.dataframe(df, use_container_width=True, height=min(400, 40 + 35 * len(df)))

            # Downloads
            st.markdown("---")
            st.subheader("Download")
            with tempfile.TemporaryDirectory() as tmp:
                bin_p = os.path.join(tmp, "binary.png")
                lab_p = os.path.join(tmp, "labeled.png")
                ov_p = os.path.join(tmp, "overlay.jpg")
                cv2.imwrite(bin_p, (fil_mask.astype(np.uint8) * 255))
                cv2.imwrite(lab_p, fil_labeled.astype(np.uint16))
                save_overlay(img, disk_mask, fil_labeled, ov_p, fil_info)
                stem = Path(uploaded_file.name).stem
                dcols = st.columns(3)
                with open(bin_p, "rb") as f:
                    dcols[0].download_button("⬇️ Binary mask", f, f"{stem}_binary_mask.png", "image/png")
                with open(lab_p, "rb") as f:
                    dcols[1].download_button("⬇️ Labeled mask", f, f"{stem}_labeled_mask.png", "image/png")
                with open(ov_p, "rb") as f:
                    dcols[2].download_button("⬇️ Overlay", f, f"{stem}_overlay.jpg", "image/jpeg")
    else:
        st.info("👆 Upload an H-alpha solar image to begin.")
        with st.expander("About this algorithm"):
            st.markdown(
                """
                **Solar filaments** are dark, elongated structures of cool dense plasma
                suspended in the corona, visible in H-alpha.

                **Pipeline:** Disk extraction → Limb-darkening correction → Local contrast
                → Adaptive threshold → Morphological clean → Instance labelling → Metrics
                (Dice + Panoptic Quality).
                """
            )


# =====================================================================
# MODE 2 — Time Series Tracking
# =====================================================================
else:
    st.title("Solar Filament Time-Series Tracking")
    st.markdown(
        "Upload an ordered sequence of H-alpha images. The app segments every frame, "
        "then associates filament instances across time into persistent tracks."
    )

    uploaded_files = st.file_uploader(
        "Upload a sequence of H-alpha images (ordered by filename)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Sort by filename for temporal order
        uploaded_files = sorted(uploaded_files, key=lambda f: f.name)
        n_frames = len(uploaded_files)
        st.success(f"Loaded **{n_frames}** frames  ·  first: `{uploaded_files[0].name}`  ·  last: `{uploaded_files[-1].name}`")

        # Preview strip
        with st.expander("Preview frames", expanded=False):
            preview_idxs = [0, n_frames // 2, n_frames - 1] if n_frames >= 3 else list(range(n_frames))
            pcols = st.columns(len(preview_idxs))
            for col, idx in zip(pcols, preview_idxs):
                im = load_gray(uploaded_files[idx])
                if im is not None:
                    col.image(im, caption=f"Frame {idx}: {uploaded_files[idx].name}", use_container_width=True, clamp=True)

        if st.button("🚀 Run Tracking", type="primary", use_container_width=True):
            params = dict(
                contrast_sigma=contrast_sigma,
                min_area_px=min_area_px,
                min_elongation=min_elongation,
                large_blob_area=large_blob_area,
            )
            dt_days = float(dt_minutes) / (24.0 * 60.0)

            # ---- Segment all frames ----
            progress = st.progress(0.0, text="Segmenting frames…")
            frame_data = []  # list of dicts per frame
            t0 = time.time()

            for i, uf in enumerate(uploaded_files):
                img = load_gray(uf)
                if img is None:
                    st.warning(f"Skipping unreadable file: {uf.name}")
                    continue
                disk_mask, fil_mask, fil_labeled, fil_info, diff_img = run_segmentation(img, params)

                # Disk geometry for differential rotation
                ys, xs = np.where(disk_mask > 0)
                if len(ys) > 0:
                    disk_cy, disk_cx = float(ys.mean()), float(xs.mean())
                    disk_radius = float(np.sqrt(((ys - disk_cy) ** 2 + (xs - disk_cx) ** 2).max()))
                else:
                    disk_cy = img.shape[0] / 2
                    disk_cx = img.shape[1] / 2
                    disk_radius = min(img.shape) / 2

                instances = labeled_mask_to_instances(
                    fil_labeled, frame_idx=i, fil_info=fil_info, store_masks=store_masks
                )
                frame_data.append(
                    {
                        "idx": i,
                        "name": uf.name,
                        "img": img,
                        "disk_mask": disk_mask,
                        "fil_mask": fil_mask,
                        "fil_labeled": fil_labeled,
                        "fil_info": fil_info,
                        "instances": instances,
                        "disk_cy": disk_cy,
                        "disk_cx": disk_cx,
                        "disk_radius": disk_radius,
                    }
                )
                progress.progress((i + 1) / n_frames, text=f"Segmenting frame {i+1}/{n_frames}")

            if not frame_data:
                st.error("No valid frames could be processed.")
                st.stop()

            # ---- Track ----
            progress.progress(1.0, text="Associating tracks…")
            # Use geometry from first frame as reference
            ref = frame_data[0]
            tracker = FilamentTracker(
                max_centroid_dist=float(max_centroid_dist),
                max_frames_lost=int(max_frames_lost),
                use_differential_rotation=use_diff_rot,
                dt_days_per_frame=dt_days,
                disk_center=(ref["disk_cy"], ref["disk_cx"]),
                disk_radius=ref["disk_radius"],
            )

            for fd in frame_data:
                tracker.update(fd["instances"], frame_idx=fd["idx"])

            tracks = tracker.finalize()
            runtime = time.time() - t0
            progress.empty()

            # Store in session so scrubber works without re-running
            st.session_state["ts_frame_data"] = frame_data
            st.session_state["ts_tracks"] = tracks
            st.session_state["ts_runtime"] = runtime

        # ---- Display results if available ----
        if "ts_tracks" in st.session_state and "ts_frame_data" in st.session_state:
            frame_data = st.session_state["ts_frame_data"]
            tracks = st.session_state["ts_tracks"]
            runtime = st.session_state.get("ts_runtime", 0)
            n_frames = len(frame_data)
            summary = tracks_to_summary(tracks)
            n_tracks = len(summary)

            st.markdown("---")
            st.subheader("Tracking Summary")
            mcols = st.columns(5)
            mcols[0].metric("Frames", n_frames)
            mcols[1].metric("Tracks", n_tracks)
            lifetimes = [r["lifetime_frames"] for r in summary] if summary else [0]
            mcols[2].metric("Mean lifetime", f"{np.mean(lifetimes):.1f} fr")
            mcols[3].metric("Max lifetime", f"{max(lifetimes)} fr")
            mcols[4].metric("Runtime", f"{runtime:.1f} s")

            # Tabs
            tab_overview, tab_frames, tab_detail, tab_export = st.tabs(
                ["Overview", "Frame Explorer", "Track Detail", "Export"]
            )

            # ---- Overview ----
            with tab_overview:
                if summary:
                    df_sum = pd.DataFrame(summary)
                    st.dataframe(df_sum, use_container_width=True, height=min(450, 40 + 35 * len(df_sum)))

                    # Lifetime histogram
                    fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0d1117")
                    ax.set_facecolor("#161b22")
                    ax.hist([r["lifetime_frames"] for r in summary], bins=min(20, max(5, n_tracks)), color="#58a6ff", edgecolor="#0d1117")
                    ax.set_xlabel("Lifetime (frames)", color="#e6edf3")
                    ax.set_ylabel("Count", color="#e6edf3")
                    ax.tick_params(colors="#e6edf3")
                    ax.set_title("Track lifetime distribution", color="#e6edf3")
                    for spine in ax.spines.values():
                        spine.set_color("#30363d")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No tracks formed.")

            # ---- Frame Explorer ----
            with tab_frames:
                frame_idx = st.slider("Frame", 0, n_frames - 1, 0)
                fd = frame_data[frame_idx]
                overlay = build_track_overlay(fd["img"], tracks, frame_idx)

                c1, c2 = st.columns(2)
                c1.image(fd["img"], caption=f"Original — {fd['name']}", use_container_width=True, clamp=True)
                c2.image(overlay, caption="Tracks (persistent colours)", use_container_width=True, clamp=True)

                # List tracks present in this frame
                present = []
                for t in tracks:
                    for inst in t.instances:
                        if inst.frame_idx == frame_idx:
                            present.append(
                                {
                                    "track_id": t.track_id,
                                    "area_px": inst.area_px,
                                    "centroid_y": round(inst.centroid_y, 1),
                                    "centroid_x": round(inst.centroid_x, 1),
                                    "elongation": round(inst.elongation, 2),
                                }
                            )
                            break
                if present:
                    st.markdown(f"**{len(present)} tracks in this frame**")
                    st.dataframe(pd.DataFrame(present), use_container_width=True)

            # ---- Track Detail ----
            with tab_detail:
                if not summary:
                    st.info("No tracks to inspect.")
                else:
                    track_ids = [r["track_id"] for r in summary]
                    selected_id = st.selectbox("Select track ID", track_ids)
                    track = next(t for t in tracks if t.track_id == selected_id)

                    # Metrics
                    tc = st.columns(4)
                    tc[0].metric("Lifetime", f"{track.lifetime_frames} frames")
                    tc[1].metric("Observations", len(track.instances))
                    tc[2].metric("Max area", f"{track.max_area:,} px")
                    tc[3].metric("Mean area", f"{track.mean_area:.0f} px")

                    # Area time series
                    series = track.area_series()
                    if series:
                        frames_s, areas_s = zip(*series)
                        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
                        ax.set_facecolor("#161b22")
                        ax.plot(frames_s, areas_s, "o-", color="#58a6ff", markersize=4)
                        ax.set_xlabel("Frame", color="#e6edf3")
                        ax.set_ylabel("Area (px)", color="#e6edf3")
                        ax.set_title(f"Track {selected_id} — area evolution", color="#e6edf3")
                        ax.tick_params(colors="#e6edf3")
                        for spine in ax.spines.values():
                            spine.set_color("#30363d")
                        st.pyplot(fig)
                        plt.close(fig)

                    # Thumbnail strip of this track
                    st.markdown("**Appearances across frames**")
                    show_idxs = [inst.frame_idx for inst in track.instances]
                    # Limit to avoid too many images
                    if len(show_idxs) > 12:
                        step = len(show_idxs) // 12
                        show_idxs = show_idxs[::step][:12]
                    tcols = st.columns(min(6, len(show_idxs)))
                    for col_i, fidx in enumerate(show_idxs):
                        fd = frame_data[fidx]
                        # Crop around the instance
                        inst = next(i for i in track.instances if i.frame_idx == fidx)
                        r0, c0, r1, c1 = inst.bbox
                        pad = 20
                        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
                        r1 = min(fd["img"].shape[0], r1 + pad)
                        c1 = min(fd["img"].shape[1], c1 + pad)
                        crop = fd["img"][r0:r1, c0:c1]
                        tcols[col_i % len(tcols)].image(
                            crop, caption=f"fr {fidx}", use_container_width=True, clamp=True
                        )

            # ---- Export ----
            with tab_export:
                if summary:
                    csv_buf = io.StringIO()
                    pd.DataFrame(summary).to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Track summary CSV",
                        csv_buf.getvalue(),
                        file_name="filament_tracks_summary.csv",
                        mime="text/csv",
                    )

                    # Full per-observation table
                    rows = []
                    for t in tracks:
                        for inst in t.instances:
                            rows.append(
                                {
                                    "track_id": t.track_id,
                                    "frame": inst.frame_idx,
                                    "area_px": inst.area_px,
                                    "centroid_y": inst.centroid_y,
                                    "centroid_x": inst.centroid_x,
                                    "length_px": inst.length_px,
                                    "width_px": inst.width_px,
                                    "elongation": inst.elongation,
                                    "orientation": inst.orientation,
                                }
                            )
                    if rows:
                        csv_full = io.StringIO()
                        pd.DataFrame(rows).to_csv(csv_full, index=False)
                        st.download_button(
                            "⬇️ Full observation table CSV",
                            csv_full.getvalue(),
                            file_name="filament_tracks_full.csv",
                            mime="text/csv",
                        )
                else:
                    st.info("Nothing to export.")
    else:
        st.info(
            "👆 Upload an ordered sequence of H-alpha images (multi-select). "
            "Filenames are sorted alphabetically to establish temporal order."
        )
        with st.expander("How tracking works"):
            st.markdown(
                """
                1. **Segment** every frame with the classical filament pipeline.  
                2. **Associate** detections across consecutive frames using centroid distance
                   (optionally after differential-rotation prediction) and IoU via the
                   Hungarian algorithm.  
                3. **Maintain tracks** — a track may be briefly lost (up to *Max frames lost*)
                   before it is closed; unmatched detections start new tracks.  
                4. Explore results in the **Overview**, **Frame Explorer**, and **Track Detail** tabs.
                """
            )

# Footer
st.markdown("---")
st.caption("Solar Filament Segmentation & Tracking · H-alpha chromospheric images")
