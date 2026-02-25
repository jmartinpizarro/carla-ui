import math
from typing import Tuple
import torch
from pyproj import CRS, Transformer


class UnitConversor:
    def __init__(
        self,
        rel_altitude: float,
        boxes,
        drone_pos: Tuple[float, float],
        gb_yaw,
        resolution: Tuple = (1920, 1080),
        fov=84.0,
        focal_len=24.0,
        gb_pitch=-90,
        gb_roll=0,
        gnss_error=0.5,
    ):
        """
        :param rel_altitude: float -> relative altitude (meters)
        :param boxes: -> Boxes in the form xyxy of YOLO prediction
        :param drone_pos: Tuple[float, float] -> the drone pos in the curr frame
        :param gb_yaw: float -> the yaw of the UAV
        :param resolution: specs of DJI camera
        :param fov: field of view of the camera pos. Assume x axis
        :param focal_len
        :param gb_pitch: in order to calculate if the camera is wrt to the
        ground
        :param gb_roll: in order to calculate if the camera is wrt to the
        ground
        :param gnss_error: Typical error (meters) without RTK
        """
        self.resolution = resolution
        self.boxes = boxes
        self.drone_pos = drone_pos
        self.gb_yaw = gb_yaw
        self.fov = fov
        self.focal_len = focal_len
        self.gb_pitch = gb_pitch
        self.gb_roll = gb_roll
        self.rel_altitude = rel_altitude
        self.gnss_error = gnss_error

    def update_drone_pos(self, value: Tuple[float, float]):
        self.drone_pos = value

    def _get_utm_zone(self, lon, lat):
        """
        Calculate the UTM zone EPSG code for a given lat/lon
        """
        utm_zone = int((lon + 180) / 6) + 1
        # Northern or Southern hemisphere
        if lat >= 0:
            epsg_code = 32600 + utm_zone  # WGS84 UTM North
        else:
            epsg_code = 32700 + utm_zone  # WGS84 UTM South
        return epsg_code

    def calc_rw_positions_boxes(self):
        """
        :return: Tuple[float, float] -> (lat, lon)
        """
        # the position to calculate is going to be done with respect to
        # the pixel that is considered the center of the box
        u = (self.boxes[:, 0] + self.boxes[:, 2]) / 2  # axis x
        v = (self.boxes[:, 1] + self.boxes[:, 3]) / 2  # axis y

        # Aproximated intrinsec of the FOV
        cx = self.resolution[0] / 2
        cy = self.resolution[1] / 2

        fov_x_rad = math.radians(self.fov)
        fx = cx / math.tan(fov_x_rad / 2)

        fov_y_rad = 2 * math.atan(
            (self.resolution[1] / self.resolution[0]) * math.tan(fov_x_rad / 2)
        )
        fy = cy / math.tan(fov_y_rad / 2)

        # pixel into normalised coordinates
        x_n = (u - cx) / fx
        y_n = -(v - cy) / fy

        # nadir -> if 90 degrees wrt the ground, it is possible to
        # calculate the displacement
        dx = self.rel_altitude * x_n
        dy = self.rel_altitude * y_n

        r_yaw = math.radians(self.gb_yaw)
        x = (dx * math.cos(r_yaw)) - (dy * math.sin(r_yaw))  # east (E)
        y = (dx * math.sin(r_yaw)) + (dy * math.cos(r_yaw))  # north (N)

        lat0, lon0 = self.drone_pos

        # automatic UTM depending on position
        utm_epsg = self._get_utm_zone(lon0, lat0)
        utm_crs = CRS.from_epsg(utm_epsg)

        to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)

        # drone_position -> UTM
        x0, y0 = to_utm.transform(lon0, lat0)

        # add displacement
        x_new = x0 + x
        y_new = y0 + y

        # transform again
        lon_new, lat_new = to_wgs84.transform(x_new.tolist(), y_new.tolist())

        return torch.tensor(lat_new), torch.tensor(lon_new)
