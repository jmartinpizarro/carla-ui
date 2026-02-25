import base64
import glob
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.utils.unit_conversor import UnitConversor
from backend.utils.yolo_model import YoloModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _file_to_base64(file_path: Path) -> str:
    with open(file_path, "rb") as file_descriptor:
        return base64.b64encode(file_descriptor.read()).decode("utf-8")


def _guess_content_type(file_path: Path, fallback: str) -> str:
    guessed_type, _ = mimetypes.guess_type(str(file_path))
    return guessed_type or fallback


def _is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def convert_to_webm(input_path: str, output_path: str) -> str:
    """Convert video to WebM with VP9 codec using ffmpeg for browser compatibility."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0',
        '-an',  # No audio
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    print(f'[convert_to_webm] :: Converted {input_path} -> {output_path}')
    return output_path


def create_simple_plots_video(
    plots_dir: str = 'simple_plots',
    output_video: str = 'simple_plots/simple_plots_video.webm',
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

    # Create a temporary MP4 first using OpenCV (reliable), then convert to WebM with ffmpeg
    temp_mp4 = output_video.replace('.webm', '_temp.mp4')
    
    first_frame = cv2.imread(png_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, fps, (width, height))

    for i, png_file in enumerate(png_files):
        frame = cv2.imread(png_file)
        out.write(frame)
        if (i + 1) % 10 == 0:
            print(
                f'[create_simple_plots_video] :: Processed {i + 1}/{len(png_files)} frames'
            )

    out.release()
    print(f'[create_simple_plots_video] :: Temp MP4 created, converting to WebM...')

    # Convert to WebM using ffmpeg
    convert_to_webm(temp_mp4, output_video)
    
    # Clean up temp file
    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)
    
    print(f'[create_simple_plots_video] :: Video saved to {output_video}')
    return output_video

@app.post("/inference")
async def run_inference(
    model: UploadFile = File(...),
    frame: UploadFile = File(...),
    inference_mode: str = Form(...),
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

                for k in r_boxes.keys():
                    boxes_array = r_boxes.get(k, [])
                    if not boxes_array:
                        continue

                    boxes = torch.tensor(boxes_array, dtype=torch.int16)
                    drone_pos = (-33.253099, -54.504020)
                    rel_alt = 10.09

                    frame_mask = np.zeros((height, width), dtype=np.uint8)
                    for x1, y1, x2, y2 in boxes.tolist():
                        frame_mask[y1:y2, x1:x2] = 1
                    coverage = 100.0 * frame_mask.sum() / (width * height)

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
                    corner_conversor = UnitConversor(
                        rel_altitude=rel_alt,
                        boxes=corner_boxes,
                        drone_pos=drone_pos,
                        gb_yaw=29.5,
                        resolution=(width, height),
                    )
                    ref_lats, ref_lons = corner_conversor.calc_rw_positions_boxes()

                    model_runner.generate_simple_circle_plot(
                        lats,
                        lons,
                        ref_lats,
                        ref_lons,
                        frame=k,
                        output_dir=str(simple_plots_dir),
                        coverage=coverage,
                    )

            simple_plots_video_path = create_simple_plots_video(
                plots_dir=str(simple_plots_dir),
                output_video=str(simple_plots_dir / "simple_plots_video.webm"),
                fps=5,
            )

            if is_video:
                # YoloModel creates output.mp4 with mp4v codec (not browser compatible)
                # Convert it to WebM with VP8 for browser playback
                yolo_output_mp4 = temp_path / "output.mp4"
                output_media_path = temp_path / "output.webm"
                if yolo_output_mp4.exists():
                    convert_to_webm(str(yolo_output_mp4), str(output_media_path))
                
                simple_plots_media_path = (
                    Path(simple_plots_video_path)
                    if simple_plots_video_path is not None
                    else temp_path / "output.webm"
                )
                simple_plots_content_type = "video/webm"
                output_content_type = "video/webm"
            else:
                output_media_path = temp_path / "output.jpg"
                simple_plot_img = simple_plots_dir / "simple_plot_0.png"
                if simple_plot_img.exists():
                    simple_plots_media_path = simple_plot_img
                else:
                    fallback = simple_plots_dir / "simple_plot_0.png"
                    shutil.copyfile(output_media_path, fallback)
                    simple_plots_media_path = fallback
                simple_plots_content_type = _guess_content_type(simple_plots_media_path, "image/png")
                output_content_type = _guess_content_type(output_media_path, "image/jpeg")

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            os.chdir(original_cwd)

        if not output_media_path.exists():
            raise HTTPException(status_code=500, detail="processed output media was not generated")

        if not simple_plots_media_path.exists():
            raise HTTPException(status_code=500, detail="simple_plots media was not generated")

        return {
            "results": {
                "simple_plots": {
                    "filename": simple_plots_media_path.name,
                    "content_type": simple_plots_content_type,
                    "content_base64": _file_to_base64(simple_plots_media_path),
                },
                "output_video": {
                    "filename": output_media_path.name,
                    "content_type": output_content_type,
                    "content_base64": _file_to_base64(output_media_path),
                },
            },
        }