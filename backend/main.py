import glob
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import geopandas as gpd
from shapely import Point
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pyproj import Transformer

from utils.unit_conversor import UnitConversor
from utils.yolo_model import YoloModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def create_simple_plots_video(
    plots_dir: str = 'simple_plots',
    output_video: str = 'simple_plots/simple_plots_video.mp4',
    fps: int = 5,
):
    png_files = sorted(
        glob.glob(os.path.join(plots_dir, 'simple_plot_*.png')),
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
    )

    if not png_files:
        print(f'[create_simple_plots_video] :: No PNG files found in {plots_dir}')
        return None

    print(f'[create_simple_plots_video] :: Found {len(png_files)} PNG files')

    first_frame = cv2.imread(png_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i, png_file in enumerate(png_files):
        frame = cv2.imread(png_file)
        out.write(frame)
        if (i + 1) % 10 == 0:
            print(
                f'[create_simple_plots_video] :: Processed {i + 1}/{len(png_files)} frames'
            )

    out.release()
    print(f'[create_simple_plots_video] :: Video saved to {output_video}')
    return output_video


def create_kde_plots_video(
    plots_dir: str = 'kde_plots',
    output_video: str = 'kde_plots/kde_plots_video.mp4',
    fps: int = 5,
):
    png_files = sorted(
        glob.glob(os.path.join(plots_dir, 'density_heatmap_kde_*.png')),
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
    )

    if not png_files:
        print(f'[create_kde_plots_video] :: No PNG files found in {plots_dir}')
        return None

    print(f'[create_kde_plots_video] :: Found {len(png_files)} PNG files')

    first_frame = cv2.imread(png_files[0])
    if first_frame is None:
        raise RuntimeError(f'Could not read first KDE plot image: {png_files[0]}')

    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i, png_file in enumerate(png_files):
        frame = cv2.imread(png_file)
        if frame is None:
            raise RuntimeError(f'Could not read KDE plot image: {png_file}')
        out.write(frame)
        if (i + 1) % 10 == 0:
            print(
                f'[create_kde_plots_video] :: Processed {i + 1}/{len(png_files)} frames'
            )

    out.release()
    print(f'[create_kde_plots_video] :: Video saved to {output_video}')
    return output_video

@app.post("/inference")
async def run_inference(
    model: UploadFile = File(...),
    frame: UploadFile = File(...),
    inference_mode: str = Form(...),
    density_threshold: float = Form(...),
):
    model_contents = await model.read()
    frame_contents = await frame.read()

    tiled_mode = inference_mode.lower() == "tiled"

    with tempfile.TemporaryDirectory(prefix="carla_inference_") as temp_dir:
        temp_path = Path(temp_dir)
        model_path = temp_path / (model.filename or "model.pt")
        frame_path = temp_path / (frame.filename or "input.mp4")
        predictions_path = temp_path / "output.pred"

        model_path.write_bytes(model_contents)
        frame_path.write_bytes(frame_contents)

        is_video = _is_video_suffix(frame_path.suffix or ".mp4")
        simple_plots_dir = temp_path / "simple_plots"
        simple_plots_dir.mkdir(parents=True, exist_ok=True)
        frame_densities = []
        frame_indices = []

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_path)
            model_runner = YoloModel(
                model=str(model_path),
                tiled=tiled_mode,
                input_data=str(frame_path),
                log_files=str(predictions_path),
            )

            r_boxes = model_runner.inference() or {}

            # Get video FPS for matching simple_plots video duration
            video_fps = 30.0  # Default FPS
            if is_video:
                cap = cv2.VideoCapture(str(frame_path))
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

            if r_boxes:
                if is_video:
                    cap = cv2.VideoCapture(str(frame_path))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                else:
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is None:
                        raise RuntimeError("Could not read input image")
                    height, width, _ = frame_img.shape

                def _frame_sort_key(value):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return str(value)

                for k in sorted(r_boxes.keys(), key=_frame_sort_key):
                    boxes_array = r_boxes.get(k, []) or []
                    try:
                        frame_idx = int(k)
                    except (TypeError, ValueError):
                        frame_idx = len(frame_indices)
                    frame_indices.append(frame_idx)

                    coverage = 0.0
                    if not boxes_array:
                        frame_densities.append(coverage)
                        continue

                    boxes = torch.tensor(boxes_array, dtype=torch.int16)
                    drone_pos = (-33.253099, -54.504020)
                    rel_alt = 10.09

                    frame_mask = np.zeros((height, width), dtype=np.uint8)
                    for x1, y1, x2, y2 in boxes.tolist():
                        frame_mask[y1:y2, x1:x2] = 1
                    coverage = 100.0 * frame_mask.sum() / (width * height)
                    frame_densities.append(float(coverage))

                    conversor = UnitConversor(
                        rel_altitude=rel_alt,
                        boxes=boxes,
                        drone_pos=drone_pos,
                        gb_yaw=29.5,
                        resolution=(width, height),
                    )

                    lats, lons = conversor.calc_rw_positions_boxes()

                    corner_boxes = torch.stack(
                        [boxes[:, 2], boxes[:, 3], boxes[:, 2], boxes[:, 3]],
                        dim=1,
                    )
                    conversor = UnitConversor(
                        rel_altitude=rel_alt,
                        boxes=corner_boxes,
                        drone_pos=drone_pos,
                        gb_yaw=29.5,
                        resolution=(width, height),
                    )

                    # transform positions into lat, lon
                    lats, lons = conversor.calc_rw_positions_boxes()
                    
                    # (no drone pos included) - could be useful in order to see the evolution
                    # of the flight
                    geometries = [Point(lon, lat) for lat, lon in zip(lats, lons)]
                    gdf = gpd.GeoDataFrame(
                       {'type': ['detection'] * len(geometries)},
                       geometry=geometries,
                       crs='EPSG:4326',
                    ).to_crs(epsg=3857)
                    
                    model_runner.generate_density_heatmap(
                        gdf_metric=gdf,
                        bandwidths=[1, 2],
                        frame=frame_idx,
                    )

                    # circle using box center and opposite corner as radius
                    # ref_lats, ref_lons = conversor.calc_rw_positions_pixels(
                    #     boxes[:, 2], boxes[:, 3]
                    # )
                    # model_runner.generate_simple_circle_plot(
                    #     lats,
                    #     lons,
                    #     ref_lats,
                    #     ref_lons,
                    #     frame=k,
                    #     output_dir=str(simple_plots_dir),
                    #     coverage=coverage,
                    # )

            kde_plots_video_path = create_kde_plots_video(
                plots_dir='kde_plots',
                output_video='kde_plots/kde_plots_video.mp4',
                fps=int(video_fps),
            )

            simple_plots_video_path = create_simple_plots_video(
                plots_dir=str(simple_plots_dir),
                output_video=str(simple_plots_dir / "simple_plots_video.mp4"),
                fps=int(video_fps),
            )

            if is_video:
                output_media_path = temp_path / "output.mp4"
                
                simple_plots_media_path = (
                    Path(simple_plots_video_path)
                    if simple_plots_video_path is not None
                    else temp_path / "output.mp4"
                )
                kde_plots_media_path = (
                    Path(kde_plots_video_path)
                    if kde_plots_video_path is not None
                    else temp_path / "output.mp4"
                )
            else:
                output_media_path = temp_path / "output.jpg"
                simple_plot_img = simple_plots_dir / "simple_plot_0.png"
                if simple_plot_img.exists():
                    simple_plots_media_path = simple_plot_img
                else:
                    fallback = simple_plots_dir / "simple_plot_0.png"
                    shutil.copyfile(output_media_path, fallback)
                    simple_plots_media_path = fallback

                kde_plot_img = Path('kde_plots') / 'density_heatmap_kde_0.png'
                if kde_plot_img.exists():
                    kde_plots_media_path = kde_plot_img
                else:
                    kde_plots_media_path = output_media_path

            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            saved_dir = Path(__file__).resolve().parent / "generated_runs" / run_id
            saved_dir.mkdir(parents=True, exist_ok=True)

            if is_video:
                kde_plots_video_path = create_kde_plots_video(
                    plots_dir='kde_plots',
                    output_video=str(saved_dir / "kde_plots_video.mp4"),
                    fps=int(video_fps),
                )
                kde_plots_media_path = (
                    Path(kde_plots_video_path)
                    if kde_plots_video_path is not None
                    else temp_path / "output.mp4"
                )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            os.chdir(original_cwd)

        if not output_media_path.exists():
            raise HTTPException(status_code=500, detail="processed output media was not generated")

        if not simple_plots_media_path.exists():
            raise HTTPException(status_code=500, detail="simple_plots media was not generated")

        if not kde_plots_media_path.exists():
            raise HTTPException(status_code=500, detail="kde_plots media was not generated")

        saved_output = saved_dir / output_media_path.name
        saved_plots = saved_dir / simple_plots_media_path.name
        saved_kde_plots = saved_dir / kde_plots_media_path.name
        shutil.copy2(output_media_path, saved_output)
        shutil.copy2(simple_plots_media_path, saved_plots)
        if kde_plots_media_path.resolve() != saved_kde_plots.resolve():
            shutil.copy2(kde_plots_media_path, saved_kde_plots)
        if predictions_path.exists():
            shutil.copy2(predictions_path, saved_dir / predictions_path.name)

        window_frames = max(1, int(round(float(video_fps) * 3)))
        logs = []

        def _fmt_time(seconds_value: int) -> str:
            mm, ss = divmod(max(0, int(seconds_value)), 60)
            return f"{mm:02d}:{ss:02d}"

        if frame_densities:
            # Build all windows >= threshold first, then aggregate by second to avoid
            # emitting almost identical logs for each frame.
            triggered_windows = []

            if len(frame_densities) < window_frames:
                window_densities = frame_densities
                density_sum = float(sum(window_densities))
                avg_density = density_sum / max(1, len(window_densities))
                if avg_density >= density_threshold:
                    end_frame_idx = frame_indices[-1] if frame_indices else 0
                    end_seconds = int(end_frame_idx / max(float(video_fps), 1.0))
                    triggered_windows.append(
                        {
                            "end_seconds": end_seconds,
                            "end_frame_idx": end_frame_idx,
                            "density_sum": density_sum,
                            "avg_density": avg_density,
                            "frames": len(window_densities),
                        }
                    )
            else:
                for i in range(window_frames - 1, len(frame_densities)):
                    window_densities = frame_densities[i - window_frames + 1:i + 1]
                    density_sum = float(sum(window_densities))
                    avg_density = density_sum / len(window_densities)
                    if avg_density >= density_threshold:
                        end_frame_idx = frame_indices[i] if i < len(frame_indices) else i
                        end_seconds = int(end_frame_idx / max(float(video_fps), 1.0))
                        triggered_windows.append(
                            {
                                "end_seconds": end_seconds,
                                "end_frame_idx": end_frame_idx,
                                "density_sum": density_sum,
                                "avg_density": avg_density,
                                "frames": len(window_densities),
                            }
                        )

            # Keep one representative log per second with the max density in that second.
            grouped_by_second = {}
            for window in triggered_windows:
                second_key = window["end_seconds"]
                if second_key not in grouped_by_second:
                    grouped_by_second[second_key] = {
                        "count": 0,
                        "best": window,
                    }
                grouped_by_second[second_key]["count"] += 1
                if window["avg_density"] > grouped_by_second[second_key]["best"]["avg_density"]:
                    grouped_by_second[second_key]["best"] = window

            for second_key in sorted(grouped_by_second.keys()):
                group = grouped_by_second[second_key]
                best = group["best"]
                start_seconds = max(0, second_key - 3)
                logs.append(
                    f"En {_fmt_time(second_key)} del video (ventana {_fmt_time(start_seconds)}-{_fmt_time(second_key)}) "
                    f"la densidad media maxima fue {best['avg_density']:.2f}% de ocupacion "
                    f"(suma de densidades: {best['density_sum']:.2f}, frames: {best['frames']}, "
                    f"ventanas detectadas ese segundo: {group['count']})."
                )

        return {
            "results": {
                "density_threshold": density_threshold,
                "window_seconds": 3,
                "window_frames": window_frames,
                "logs": logs,
                "saved_artifacts": {
                    "run_dir": str(saved_dir),
                    "output_media_path": str(saved_output),
                    "simple_plots_media_path": str(saved_plots),
                    "kde_plots_media_path": str(saved_kde_plots),
                },
            },
        }