"""
Contains the class YoloModel used for doing inference with a .pt model.
"""

from backend.utils.tiling_utils import process_frame_with_grids

import cv2
import numpy as np
import os
from pyproj import Geod
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class YoloModel:
    def __init__(self, model: str, tiled: bool, input_data: str, log_files: str):
        """
        :param model: str -> Route to the model
        :param tiled: bool -> The model uses tiling or not
        :param input_data: str -> Route to the file where you want to do the
        inference
        :param log_files: str -> Route where the program is going to write
        logging and the predictions
        """
        self.model: str = model
        self.tiled: bool = tiled
        self.input_data: str = input_data
        self.log_files: str = log_files

    def inference(self, conf_threshold=0.4, iou=0.75):
        try:
            YOLO_MODEL = YOLO(self.model)
        except Exception:
            print(
                f'[YoloModel] :: An error has ocurred when importing the model {self.model}\n'
            )
            return

        is_video = self.input_data.lower().endswith('.mp4')
        coverage_historic = np.array(())

        if is_video:
            # create some stuff we will need for procesing those files
            cap = cv2.VideoCapture(self.input_data)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        else:
            cap = None
            frame = cv2.imread(self.input_data)
            height, width, _ = frame.shape
            frames = [frame]

        frame_count = 0
        r_boxes = {}

        if self.log_files is not None:
            try:
                log_file = open(self.log_files, 'w')
                log_file.write(f'model:{self.model}\n')
                log_file.write(f'conf:{conf_threshold}\n')
                log_file.write(f'iou:{iou}\n')
                log_file.write('\n')
            except Exception as e:
                print(
                    f'[YoloModel] :: An error has ocurred when opening the file for writing the predictions:\n{e}\n'
                )
                return

        while True:
            if is_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if frame_count >= len(frames):
                    break
                frame = frames[frame_count].copy()

            if self.log_files is not None:
                try:
                    if is_video:
                        log_file.write(f'<{frame_count}>\n')

                    if self.tiled:
                        boxes, scores, classes = process_frame_with_grids(
                            frame, YOLO_MODEL, conf_threshold
                        )
                        frame_mask = np.zeros((height, width), dtype=np.uint8)
                        r_boxes.setdefault(frame_count, [])
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            r_boxes[frame_count].append([x1, y1, x2, y2])

                            frame_mask[y1:y2, x1:x2] = 1
                            coverage_historic = np.insert(
                                coverage_historic,
                                len(coverage_historic),
                                100.0 * frame_mask.sum() / (width * height),
                            )

                            log_file.write(f'{x1},{y1},{x2},{y2}\n')
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    else:
                        results = YOLO_MODEL(frame, conf=conf_threshold, iou=iou)

                        r_boxes.setdefault(frame_count, [])
                        for r in results:
                            frame_mask = np.zeros((height, width), dtype=np.uint8)
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                r_boxes[frame_count].append([x1, y1, x2, y2])

                                frame_mask[y1:y2, x1:x2] = 1
                                coverage_historic = np.insert(
                                    coverage_historic,
                                    len(coverage_historic),
                                    100.0 * frame_mask.sum() / (width * height),
                                )

                                log_file.write(f'{x1},{y1},{x2},{y2}\n')
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    log_file.write(f'\nMean Coverage: {np.mean(coverage_historic)}\n')
                except Exception as e:
                    print(
                        f'[YoloModel] :: An error has ocurred when writing the predictions:\n{e}\n'
                    )
                    if self.log_files is not None:
                        log_file.close()
                    return

            if is_video:
                out.write(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(
                        f'Processed {frame_count}/{total_frames} frames ({100 * frame_count / total_frames:.1f}%)'
                    )
                # cv2.imshow('YOLO Video Prediction', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                cv2.imwrite('output.jpg', frame)
                # cv2.imshow('YOLO Prediction', frame)
                cv2.waitKey(0)
                frame_count += 1

        if self.log_files is not None:
            log_file.close()

        if is_video:
            out.release()
            cap.release()

        cv2.destroyAllWindows()

        return r_boxes

    def write_predictions(self):
        pass

    def generate_density_heatmap(
        self, gdf_metric, bandwidths=[1.0, 2.0, 5.0], frame: int = 0
    ):
        """
        Generates a KDE (heatmap density map) with different physical radius. The idea is to
        group all the possible detections (positions) of a single detection (frame)
        into a plot.

        :gdf_metric: GeoDataFrame with geometry in EPSG:3857 (metric)
        :bandwidths: list with radius for plotting
        :frame: int -> by default 0, assuming it is an image. Done in order to not overwrite
        the contents and be able to iterate for making a video
        """
        # extract coordinates
        coords = np.vstack([gdf_metric.geometry.x, gdf_metric.geometry.y]).T

        print('\n[yolo_model] :: KDE Analysis config:')
        print(f'\t-Total detections: {len(coords)}')
        print(
            f'\t-X axis Range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f} meters'
        )
        print(
            f'\t-Y axis Range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f} meters'
        )
        print(f'\t-Radius: {bandwidths} meters\n')

        # create grid delimiters (5 offset works fine in this case, bigger numbers)
        # may break the plot a little bit
        x_min, x_max = coords[:, 0].min() - 5, coords[:, 0].max() + 5
        y_min, y_max = coords[:, 1].min() - 5, coords[:, 1].max() + 5

        xx, yy = np.mgrid[
            x_min:x_max:50j, y_min:y_max:50j
        ]  # Reducido de 100j a 50j para ahorrar memoria
        positions = np.vstack([xx.ravel(), yy.ravel()]).T

        # create a figure per bandwidth
        n_bw = len(bandwidths)
        fig, axes = plt.subplots(1, n_bw, figsize=(8 * n_bw, 8))
        if n_bw == 1:
            axes = [axes]

        for ax, bandwidth in zip(axes, bandwidths):
            # KDE with radius
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(coords)

            # evaluate density
            density = np.exp(kde.score_samples(positions))
            density = density.reshape(xx.shape)

            # heatmap
            levels = np.linspace(density.min(), density.max(), 15)
            contour = ax.contourf(
                xx, yy, density, levels=levels, cmap='YlOrRd', alpha=0.8
            )

            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=20,
                color='blue',
                alpha=0.5,
                edgecolors='darkblue',
                linewidth=0.5,
                label=f'Detections (n={len(coords)})',
            )

            ax.set_xlabel('X (meters, EPSG:3857)', fontsize=10)
            ax.set_ylabel('Y (meters, EPSG:3857)', fontsize=10)
            ax.set_title(f'KDE - Radius: {bandwidth}m', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)

            # Colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Density (predictions/m²)', fontsize=9)

            # Grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f'kde_plots/density_heatmap_kde_{frame}.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig)  # Libera memoria de la figura
        # plt.show()

    def generate_simple_circle_plot(
        self,
        center_lats,
        center_lons,
        ref_lats,
        ref_lons,
        frame: int = 0,
        output_dir: str = 'simple_plots',
        coverage: float = None,
    ):
        if len(center_lats) == 0:
            return

        os.makedirs(output_dir, exist_ok=True)
        geod = Geod(ellps='WGS84')

        center_lats = np.asarray(center_lats, dtype=float)
        center_lons = np.asarray(center_lons, dtype=float)
        ref_lats = np.asarray(ref_lats, dtype=float)
        ref_lons = np.asarray(ref_lons, dtype=float)

        _, _, distances = geod.inv(center_lons, center_lats, ref_lons, ref_lats)

        mean_radius = np.mean(distances) if len(distances) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            center_lons,
            center_lats,
            s=30,
            color='blue',
            alpha=0.7,
            label=f'Centers (n={len(center_lats)})',
            edgecolors='darkblue',
            linewidth=0.5,
        )

        angles = np.linspace(0, 360, 100)
        for lon, lat, radius in zip(center_lons, center_lats, distances):
            circle_lons = []
            circle_lats = []
            for angle in angles:
                lon_c, lat_c, _ = geod.fwd(lon, lat, angle, radius)
                circle_lons.append(lon_c)
                circle_lats.append(lat_c)
            ax.plot(circle_lons, circle_lats, color='red', alpha=0.6, linewidth=0.9)

        ax.set_xlabel('Longitude (EPSG:4326)', fontsize=22)
        ax.set_ylabel('Latitude (EPSG:4326)', fontsize=22)
        ax.tick_params(axis='both', labelsize=18)
        ax.tick_params(axis='x', labelrotation=45)

        title_str = (
            f'Detections with Radius (lat/lon) | Mean radius: {mean_radius:.2f} m'
        )
        if coverage is not None:
            title_str += f' | Coverage: {coverage:.2f}%'

        # ax.set_title(title_str, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=18)

        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/simple_plot_{frame}.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
