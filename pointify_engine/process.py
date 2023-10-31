import cv2
import earthpy.spatial as es
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import copy
import dialogs as dia
import numpy as np
import open3d as o3d
import os
import rasterio as rio
import shapely as sh
import shutil
import statistics
import subprocess
from shapely.geometry import Polygon
from PIL import Image, ImageOps
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal, QRunnable, QObject, QProcess, QThread
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from blend_modes import multiply, hard_light
import resources as res
from pointify_engine import simplepcv as spcv

# PARAMETERS
CC_PATH = os.path.join("C:\\", "Program Files", "CloudCompare", "CloudCompare")  # path to cloudcompare exe
POINT_LIM = 1_000_000  # the limit of point above which performing a subsampling operation for advanced computations
VOXEL_DS = 0.025  # When the point cloud is to dense, this gives the minimum spatial distance to keep between two points
MIN_RANSAC_FACTOR = 350  # (Total number of points / MIN_RANSAC_FACTOR) gives the minimum amount of points to define a
# Ransac detection
RANSAC_DIST = 0.03  # maximum distance for a point to be considered belonging to a plane

# Floor detection
BIN_HEIGHT = 0.1

# Geometric parameters
SPHERE_FACTOR = 10
SPHERE_MIN = 0.019
SPHERE_MAX = 1

# Planar analysis
SPAN = 0.03

# ODM
ODM_PATH = r'D:\Python2023\PyODM\resources\ODM\run.bat'
ODM_DEFAULT_MESH_OCTREE_LIST = [8,9,10,11,12]
ODM_DEFAULT_PC_QUALITY_LIST = ['low', 'medium', 'high', 'ultra']
ODM_DEM_RES = 0.05
# TODO: add estimation of computation time depending on quality


class ODMThread(QThread):
    outputSignal = Signal(str)
    finishedSignal = Signal()

    def __init__(self, project_path, feature_qual, pc_qual):
        super().__init__()
        self.feat_qual = feature_qual
        self.pc_qual = pc_qual
        self.project_path = project_path
        self.alt_way = True

    def run(self):
        path = ODM_PATH
        if self.alt_way:
            cmd = [path, "--project-path", self.project_path, '--feature-quality', self.feat_qual, '--pc-quality', self.pc_qual, '--mesh-octree-depth', '10', '--pc-csv', '--dsm']
        else:
            cmd = [path, "--project-path", self.project_path, '--feature-quality', self.feat_qual, '--pc-quality', self.pc_qual, '--pc-csv', '--end-with', 'odm_filterpoints']

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)

        # Read the output line by line
        for line in iter(process.stdout.readline, ""):
            self.outputSignal.emit(line)

        process.stdout.close()
        process.wait()
        self.finishedSignal.emit()

class RunnerSignals(QtCore.QObject):
    progressed = QtCore.Signal(int)
    messaged = QtCore.Signal(str)
    finished = QtCore.Signal()


class RunnerBasicProp(QtCore.QRunnable):
    def __init__(self, start, stop, pc_load, save_bound_pc=False, output_path_bound_pc='', compute_full_density=False):
        super().__init__()
        self.signals = RunnerSignals()
        self.pc_load = pc_load
        self.save_bound_pc = save_bound_pc
        self.output_path_bound_pc = output_path_bound_pc
        self.compute_full_density = compute_full_density

        self.start = start
        self.stop = stop

        self.bound = None
        self.points_bb_np = None
        self.center = None
        self.dim = None
        self.density = None
        self.n_points = None

    def run(self):
        print('go')
        step = (self.stop - self.start) / 100
        self.signals.progressed.emit(self.start + step)
        self.signals.messaged.emit('Preprocessing started')

        # bounding box
        bound = self.pc_load.get_axis_aligned_bounding_box()
        center = bound.get_center()
        dim = bound.get_extent()
        points_bb = bound.get_box_points()
        points_bb_np = np.asarray(points_bb)
        # create point cloud from bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = points_bb
        pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # write point cloud
        if self.save_bound_pc:
            o3d.io.write_point_cloud(self.output_path_bound_pc, pcd)

        # output the number of points of the point cloud
        points_list = np.asarray(self.pc_load.points)
        n_points = len(points_list[:, 1])

        # compute density
        density = []
        dim_x = dim[0]
        dim_y = dim[1]
        dim_z = dim[2]

        if not self.compute_full_density:
            pt1 = [center[0] - dim_x / 8, center[1] - dim_y / 8, center[2] - dim_z / 2]
            pt2 = [center[0] + dim_x / 8, center[1] + dim_y / 8, center[2] + dim_z / 2]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)

            crop_box = o3d.geometry.AxisAlignedBoundingBox
            crop_box = crop_box.create_from_points(points)

            point_cloud_crop = self.pc_load.crop(crop_box)
            dist = point_cloud_crop.compute_nearest_neighbor_distance()

            if dist:
                density = statistics.mean(dist)
            else:
                density = 0.01  # TODO: adapting if no points found. Idea : pick some small sample of the point cloud
        else:  # density on the whole point cloud
            dist = self.pc_load.compute_nearest_neighbor_distance()
            density = statistics.mean(dist)

        # return bound, points_bb_np, center, dim, density, n_points
        self.bound = bound
        self.points_bb_np = points_bb_np
        self.center = center
        self.dim = dim
        self.density = density
        self.n_points = n_points


class StockPileObject:
    def __init__(self):
        self.name = ''
        self.yolo_bbox = None
        self.pc = None
        self.mask = None
        self.mask_rgb = None
        self.mask_cropped = None
        self.mask_rgb_cropped = None
        self.coords = None
        self.area = 0
        self.volume = 0
        self.to_check = True # if initiated from YOLO, needs to be checked by user


class LineMeas:
    def __init__(self):
        self.main_item = None
        self.spot_items = []
        self.text_items = []
        self.coords = []
        self.data_roi = []

        self.hmin = 0
        self.hmax = 0

    def compute_data(self, img, P1, P2):
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(int), itbuffer[:, 0].astype(int)]
        self.data_roi = itbuffer[:, 2]

    def compute_extrema(self):
        self.hmax = np.amax(self.data_roi)
        self.hmin = np.amin(self.data_roi)

    def create_all_annex_infos(self):
        pass


class NokPointCloud:
    def __init__(self):
        self.name = ''
        self.path = ''

        self.pc_load = None
        self.mesh_load = None
        self.bound_pc_path = ''
        self.sub_pc_path = ''
        self.folder = ''
        self.processed_data_dir = ''
        self.img_dir = ''
        self.sub_sampled = False
        self.view_names = []
        self.view_paths = []

        self.poisson_mesh_path = ''

        # basic properties
        self.bound, self.bound_points, self.center, self.dim, self.density, self.n_points = 0, 0, 0, 0, 0, 0
        self.n_points = 0
        self.n_points_sub = 0

        # render properties
        self.res = 0
        self.standard_im_done = False
        self.height_im_done = False

        # dsm properties
        self.height_data = []
        self.high_point = 0
        self.low_point = 0

        # possible external data
        self.dsm_file_path = ''
        self.ortho_file_path = ''

    def update_dirs(self):
        self.location_dir, self.file = os.path.split(self.path)

    def do_preprocess(self):

        self.bound_pc_path = os.path.join(self.processed_data_dir, "pc_limits.ply")
        self.bound, self.bound_points, self.center, self.dim, self.density, self.n_points, self.pc_load = compute_basic_properties(
            self.path,
            save_bound_pc=True,
            output_path_bound_pc=self.bound_pc_path)

        print(f'The point cloud density is: {self.density:.3f}')

    def do_subsampling(self):
        print('subsampling...')
        sub_pc_path = os.path.join(self.location_dir,
                                   'subsampled.ply')  # path to the subsampled version of the point cloud
        sub = self.pc_load.voxel_down_sample(0.05)
        o3d.io.write_point_cloud(sub_pc_path, sub)
        self.sub_sampled = True
        self.density = 0.05

        # replace with subsampled version
        self.path = sub_pc_path
        self.update_dirs()

    def do_mesh(self):
        if self.pc_load:
            self.pc_load.estimate_normals()
            self.pc_load.orient_normals_consistent_tangent_plane(100)
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.pc_load, depth=11)

            densities = np.asarray(densities)

            print('remove low density vertices')
            vertices_to_remove = densities < np.quantile(densities, 0.025)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.compute_triangle_normals()

            self.poisson_mesh_path = os.path.join(self.processed_data_dir, "poisson_mesh.obj")

            o3d.io.write_triangle_mesh(self.poisson_mesh_path, mesh)
            self.mesh_load = o3d.io.read_triangle_mesh(self.poisson_mesh_path)


    def recompute_elevation(self):
        path_dtm = os.path.join(self.img_dir, 'dtm.tif')
        path_top_elevation = os.path.join(self.img_dir, 'elevation.png')
        path_top_hillshade = os.path.join(self.img_dir, 'hillshade.png')
        path_top_hybrid1 = os.path.join(self.img_dir, 'hybrid1.png')
        path_top_hybrid2 = os.path.join(self.img_dir, 'hybrid2.png')

        # process files
        create_elevation(path_dtm, path_top_elevation, high_limit=self.high_point, low_limit=self.low_point, type='standard')
        create_elevation(path_dtm, path_top_hillshade, high_limit=self.high_point, low_limit=self.low_point, type='hill')

        create_mixed_elevation_views(path_top_elevation, path_top_hillshade, self.view_paths[0],
                                     path_top_hybrid1, path_top_hybrid2)

    def create_height_distib(self):
        pass

    def image_selection(self):
        if self.res == 0:
            self.res = round(self.density * 4, 3) * 1000
        elif self.dsm_file_path:
            self.res = ODM_DEM_RES * 1000 # default dem resolution from ODM

        print(f'the image resolution is {self.res}')

        # create new path for dtm
        path_top = os.path.join(self.img_dir, 'top.tif')
        path_pcv = os.path.join(self.img_dir, 'pcv.tif')
        path_dtm = os.path.join(self.img_dir, 'dtm.tif')
        path_top_elevation = os.path.join(self.img_dir, 'elevation.png')
        path_top_hillshade = os.path.join(self.img_dir, 'hillshade.png')
        path_top_hybrid1 = os.path.join(self.img_dir, 'hybrid1.png')
        path_top_hybrid2 = os.path.join(self.img_dir, 'hybrid2.png')

        self.view_names.extend(
            ['top', 'pcv', 'elevation', 'hillshade', 'hybrid (hillshade/elevation)', 'hybrid (elevation/rgb)'])
        self.view_paths.extend(
            [path_top, path_pcv, path_top_elevation, path_top_hillshade, path_top_hybrid1, path_top_hybrid2])


        if not self.dsm_file_path:
            raster_top_rgb_height(self.path, self.res / 1000)
            # raster_top_rgb_height_pcv(self.path, self.res / 1000)

            # relocate image file
            img_list = generate_list('.tif', self.location_dir)
            print(f'image list {img_list}')
            os.rename(img_list[0], path_top)
            os.rename(img_list[1], path_dtm)
        else:
            # relocate ODM files
            os.rename(self.dsm_file_path, path_dtm)
            os.rename(self.ortho_file_path, path_top)

        # store height data
        with rio.open(path_dtm) as src:
            elevation = src.read(1)
            # Set masked values to np.nan
            elevation[elevation < 0] = np.nan
            self.height_data = elevation
            self.ground_data = copy.deepcopy(self.height_data)

            # call height mod
            dialog = dia.MySliderDemo(self.height_data)

            if dialog.exec_():
                self.low_point = dialog.slider_low.value()
                self.high_point = dialog.slider_high.value()

        # process files
        create_elevation(path_dtm, path_pcv, type='pcv')
        create_elevation(path_dtm, path_top_elevation, type='standard')
        create_elevation(path_dtm, path_top_hillshade, type='hill')

        create_mixed_elevation_views(path_top_elevation, path_top_hillshade, self.view_paths[0],
                                     path_top_hybrid1, path_top_hybrid2)

    def reset_ground(self):
        pass

    def standard_images(self):
        self.res = round(self.density, 3) * 1000
        print(f'the image resolution is {self.res}')
        raster_all_bound(self.path, self.res / 1000, self.bound_pc_path, xray=False)

        # create new images paths
        path_top = os.path.join(self.img_dir, 'top.tif')
        path_right = os.path.join(self.img_dir, 'right.tif')
        path_front = os.path.join(self.img_dir, 'front.tif')
        path_front_after = os.path.join(self.img_dir, 'front2.tif')
        path_back = os.path.join(self.img_dir, 'back.tif')
        path_left = os.path.join(self.img_dir, 'left.tif')

        self.view_names.extend(['top', 'right', 'front', 'back', 'left'])
        self.view_paths.extend([path_top, path_right, path_front, path_back, path_left])

        # relocate image files
        img_list = generate_list('.tif', self.location_dir)
        os.rename(img_list[0], path_right)
        os.rename(img_list[1], path_back)
        os.rename(img_list[2], path_top)
        os.rename(img_list[3], path_left)
        os.rename(img_list[4], path_front)

        # rotate right view (CloudCompare output is tilted by 90°)
        # read the images
        im_front = Image.open(path_front)
        im_back = Image.open(path_back)
        im_left = Image.open(path_left)

        # rotate image by 90 degrees and mirror if needed
        angle = 90
        # process front image

        out_f = im_front.rotate(angle, expand=True)
        out_f_mir = ImageOps.mirror(out_f)
        im_front.close()
        out_f_mir.save(path_front)
        # process back image
        out_b = im_back.rotate(angle, expand=True)
        im_back.close()
        out_b.save(path_back)
        # process left image
        out_l_mir = ImageOps.mirror(im_left)
        im_left.close()
        out_l_mir.save(path_left)

        self.standard_im_done = True

    def height_images(self):
        if self.standard_im_done:
            raster_single_bound_height(self.path, self.res / 1000, 2, self.bound_pc_path)

            # create new path for dtm
            path_dtm = os.path.join(self.img_dir, 'dtm.tif')
            path_top_elevation = os.path.join(self.img_dir, 'elevation.png')
            path_top_hillshade = os.path.join(self.img_dir, 'hillshade.png')
            path_top_hybrid1 = os.path.join(self.img_dir, 'hybrid1.png')
            path_top_hybrid2 = os.path.join(self.img_dir, 'hybrid2.png')

            self.view_names.extend(['elevation', 'hillshade', 'hybrid (hillshade/elevation)', 'hybrid (elevation/rgb)'])
            self.view_paths.extend([path_top_elevation, path_top_hillshade, path_top_hybrid1, path_top_hybrid2])

            # relocate image file
            img_list = generate_list('.tif', self.location_dir)
            os.rename(img_list[0], path_dtm)

            # store height data
            with rio.open(path_dtm) as src:
                elevation = src.read(1)
                # Set masked values to np.nan
                elevation[elevation < 0] = np.nan
                self.height_data = elevation

            # process files
            create_elevation(path_dtm, path_top_elevation, type='standard')
            create_elevation(path_dtm, path_top_hillshade, type='hill')

            create_mixed_elevation_views(path_top_elevation, path_top_hillshade, self.view_paths[0],
                                                 path_top_hybrid1, path_top_hybrid2)

        else:
            print('Process standard images first!')
            pass

    def planarity_images(self, orientation, span):
        if self.ransac_done:
            shutil.rmtree(self.ransac_cloud_folder)
            shutil.rmtree(self.ransac_obj_folder)
        self.do_ransac(min_factor=150)

        # create pcv version of subsampled cloud
        self.pcv_path = os.path.join(self.processed_data_dir, 'pcv.ply')
        create_pcv(self.sub_pc_path)
        sub_dir, _ = os.path.split(self.sub_pc_path)
        find_substring_new_path('PCV', self.pcv_path, sub_dir)
        # apply iso_transf
        mat, inv_mat = iso1_mat()
        cc_rotate_from_matrix(self.pcv_path, mat)
        self.transf_pcv_path = find_substring('pcv_TRANSFORMED', self.processed_data_dir)

        print('Launching planarity views creation...')
        # create new directory for results
        h_planes_img_dir = os.path.join(self.img_dir, 'horizontal_planes_views')
        h_planes_pc_dir = os.path.join(self.processed_data_dir, 'horizontal_planes_pc')
        new_dir(h_planes_img_dir)
        new_dir(h_planes_pc_dir)

        # create a list of detected planes
        plane_list = generate_list('obj', self.ransac_obj_folder, exclude='merged')

        # find horizontal planes
        hor_planes = find_planes(plane_list, self.ransac_cloud_folder, orientation=orientation,
                                         size_absolute='area_greater_than')
        hor_planes_loc = hor_planes[2]

        # create new_clouds from the detected planes (the original point cloud is segmented around the plane)
        n_h_elements = cc_planes_to_build_dist_list(self.path, hor_planes_loc, h_planes_pc_dir, span=span)

        # computing the properties for each new point cloud --> Useful to place the images on the entire point cloud
        new_pc_list = generate_list('.las', h_planes_pc_dir)
        # TODO: continue here

        # rendering each element
        list_h_planes_pc = generate_list('.las', h_planes_pc_dir)
        for cloud in list_h_planes_pc:
            render_planar_segment(cloud, self.res / 1000)

            # visual location
            #   rotate plane
            cc_rotate_from_matrix(cloud, mat)
            plane_rotated_path = find_substring('TRANSFORMED', h_planes_pc_dir)
            render_plane_in_cloud(plane_rotated_path, self.transf_pcv_path, self.res / 1000)

        i = 0
        for file in os.listdir(self.processed_data_dir):
            if file.endswith('.tif'):
                new_file = f'location_plane{i + 1}'
                os.rename(os.path.join(self.processed_data_dir, file), os.path.join(h_planes_img_dir, new_file))
                i += 1
        j = 0
        for file in os.listdir(h_planes_pc_dir):
            if file.endswith('.tif'):
                new_file = f'planarity{j + 1}'
                os.rename(os.path.join(h_planes_pc_dir, file), os.path.join(h_planes_img_dir, new_file))
                j += 1

        # add renders to list
        for img in os.listdir(h_planes_img_dir):
            self.view_names.append(img)
            self.view_paths.append(os.path.join(h_planes_img_dir, img))

    def do_orient(self):
        R = preproc_align_cloud(self.path, self.ransac_obj_folder, self.ransac_cloud_folder)
        print(f'The point cloud has been rotated with {R} matrix...')

        transformed_path = find_substring('TRANSFORMED', self.location_dir)
        _, trans_file = os.path.split(transformed_path)
        new_path = os.path.join(self.processed_data_dir, trans_file)
        # move transformed file
        os.rename(transformed_path, new_path)

        self.path = new_path
        self.update_dirs()

        self.bound, self.bound_points, self.center, self.dim, _, _, self.pc_load = compute_basic_properties(self.path,
                                                                                                      save_bound_pc=True,
                                                                                                      output_path_bound_pc=self.bound_pc_path)
        if self.sub_sampled:
            self.sub_pc_path

    def do_ransac(self, min_factor=MIN_RANSAC_FACTOR):
        self.sub_sampled = False
        # create RANSAC directories
        self.ransac_cloud_folder = os.path.join(self.processed_data_dir, 'RANSAC_pc')
        new_dir(self.ransac_cloud_folder)

        self.ransac_obj_folder = os.path.join(self.processed_data_dir, 'RANSAC_meshes')
        new_dir(self.ransac_obj_folder)

        # subsampling the point cloud if needed
        self.sub_pc_path = os.path.join(self.processed_data_dir,
                                        'subsampled.ply')  # path to the subsampled version of the point cloud

        if self.n_points > POINT_LIM:  # if to many points --> Subsample
            sub = self.pc_load.voxel_down_sample(VOXEL_DS)
            o3d.io.write_point_cloud(self.sub_pc_path, sub)
            self.sub_sampled = True
        else:
            shutil.copyfile(self.path, self.sub_pc_path)

        if self.sub_sampled:
            _, _, _, _, _, self.n_points_sub, _ = compute_basic_properties(self.sub_pc_path)
            print(f'The subsampled point cloud has {self.n_points_sub} points')

        # fixing RANSAC Parameters
        if not self.sub_sampled:
            points = self.n_points
        else:
            points = self.n_points_sub
        min_points = points / min_factor
        print(f'here are the minimum points {min_points}')

        self.n_planes = preproc_ransac_short(self.sub_pc_path, min_points, RANSAC_DIST, self.ransac_obj_folder,
                                                     self.ransac_cloud_folder)

        self.ransac_done = True


"""
======================================================================================
PATH FUNCTIONS
======================================================================================
"""
def find_substring(substring, folder):
    """
    Function that finds a file with a specific substring in a folder, and return its path
    @ parameters:
        substring -- substring to be looked for (string)
        folder -- input folder (string)
    """
    path = ''
    for file in os.listdir(folder):
        if substring in file:
            path = os.path.join(folder, file)
    return path


def find_substring_new_path(substring, new_path, folder):
    """
    Function that finds a file with a specific substring in a folder, and move it to a new location
    @ parameters:
        substring -- substring to be looked for (string)
        new_path -- new_path where to move the found file (string)
        folder -- input folder (string)
    """
    # rename and place in right folder

    for file in os.listdir(folder):
        if substring in file:
            os.rename(os.path.join(folder, file), new_path)


def find_substring_delete(substring, folder):
    for file in os.listdir(folder):
        if substring in file:
            os.remove(os.path.join(folder, file))


def generate_list(file_format, dir, exclude='text_to_exclude', include=''):
    """
    Function that generates the list of file with a specific extension in a folder
    :param file_format: (str) the extension to look for
    :param dir: (str) the folder to look into
    :param exclude: (str) an optional parameter to exclude files that include some text in their name
    :param include: (str) an optional parameter to specifically include files with some text in their name
    :return: (list) the list of detected files
    """
    file_list = []
    for file in os.listdir(dir):
        fileloc = os.path.join(dir, file)
        if file.endswith(file_format):
            if exclude not in file:
                if include in file:
                    file_list.append(fileloc)
    return file_list


def new_dir(dir_path):
    """
    Simple function to verify if a directory exists and if not creating it
    :param dir_path: (str) the path to check
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


"""
======================================================================================
OPEN3D FUNCTIONS
======================================================================================
"""


def basic_vis_creation(load, orientation, p_size=1.5, back_color=[1, 1, 1]):
    """A function that creates the basic environment for creating things with open3D
            @ parameters :
                pcd_load -- a point cloud loaded into open3D
                orientation -- orientation of the camera; can be 'top', ...
                p_size -- size of points
                back_color -- background color
    """
    if orientation != 'top':
        trans_init, inv_trans = name_to_matrix(orientation)
        load.transform(trans_init)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(load)
    opt = vis.get_render_option()
    opt.point_size = p_size
    opt.mesh_show_back_face = True
    opt.background_color = np.asarray(back_color)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)

    vis.poll_events()
    vis.update_renderer()
    vis.run()

    return vis, opt, ctr
def create_box_limit(pcd, output_path):
    bound = pcd.get_oriented_bounding_box()
    points_bb = bound.get_box_points()
    points_bb_np = np.asarray(points_bb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = points_bb
    pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # write point cloud
    o3d.io.write_point_cloud(output_path, pcd)


def compute_basic_properties(path, save_bound_pc=False, output_path_bound_pc='', compute_full_density=False):
    """
    A function that compiles all the basic properties of the point cloud
    :param pc_load: (open3D pc) the point cloud object, loaded into open3D
    :param save_bound_pc: (bool) Whether to save the bounding box as a point cloud
    :param output_path_bound_pc: (str) the path where to save the bounding box point cloud
    :param compute_density: (bool) whether to compute the density on the whole the point cloud
    :return:bound = the bounding box, center= the center of the bounding box, dim= the extend of the bounding box, n_points= the number of points
    """
    # bounding box
    pc_load = o3d.io.read_point_cloud(path)
    bound = pc_load.get_axis_aligned_bounding_box()
    center = bound.get_center()
    dim = bound.get_extent()

    # test for negative values
    lowest_point = center[2] - dim[2] / 2

    if lowest_point < 0:
        # shift the point cloud up
        pc_load = pc_load.translate((0,0,-lowest_point))
        bound = pc_load.get_axis_aligned_bounding_box()
        center = bound.get_center()
        dim = bound.get_extent()

        # save shifted point cloud
        o3d.io.write_point_cloud(path, pc_load)


    points_bb = bound.get_box_points()
    points_bb_np = np.asarray(points_bb)
    # create point cloud from bounding box
    pcd = o3d.geometry.PointCloud()
    pcd.points = points_bb
    pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # write point cloud
    if save_bound_pc:
        o3d.io.write_point_cloud(output_path_bound_pc, pcd)

    # output the number of points of the point cloud
    points_list = np.asarray(pc_load.points)
    n_points = len(points_list[:, 1])

    # compute density
    density = []
    dim_x = dim[0]
    dim_y = dim[1]
    dim_z = dim[2]

    if not compute_full_density:
        pt1 = [center[0] - dim_x / 8, center[1] - dim_y / 8, center[2] - dim_z / 2]
        pt2 = [center[0] + dim_x / 8, center[1] + dim_y / 8, center[2] + dim_z / 2]
        np_points = [pt1, pt2]
        points = o3d.utility.Vector3dVector(np_points)

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)

        point_cloud_crop = pc_load.crop(crop_box)
        dist = point_cloud_crop.compute_nearest_neighbor_distance()

        if dist:
            density = statistics.mean(dist)
        else:
            density = 0.01  # TODO: adapting if no points found. Idea : pick some small sample of the point cloud
    else:  # density on the whole point cloud
        dist = pc_load.compute_nearest_neighbor_distance()
        density = statistics.mean(dist)

    return bound, points_bb_np, center, dim, density, n_points, pc_load


def compute_density(pc_load, center, dim_z):
    """
    Function that uses open3D to compute the density of the point cloud
    :param pc_load: the point cloud object, loaded into open3D
    :param center: the center point
    :param dim_z:
    :return:
    """
    density = []
    pt1 = [center[0] - 2, center[1] - 2, center[2] - dim_z / 2]
    pt2 = [center[0] + 2, center[1] + 2, center[2] + dim_z / 2]
    pt3 = [center[0] - 2, center[1] + 2, center[2] + dim_z / 2]
    np_points = [pt1, pt2, pt3]
    points = o3d.utility.Vector3dVector(np_points)

    crop_box = o3d.geometry.AxisAlignedBoundingBox
    crop_box = crop_box.create_from_points(points)

    point_cloud_crop = pc_load.crop(crop_box)
    dist = point_cloud_crop.compute_nearest_neighbor_distance()

    if dist:
        density = statistics.mean(dist)
    return density


"""
======================================================================================
CLOUDCOMPARE FUNCTIONS
======================================================================================
"""


def cc_function(dest_dir, function_name, fun_txt):
    """
    Function that creates a bat file to launch CloudCompare CLI
    :param dest_dir: the folder where the bat file will be created and executed
    :param function_name: a name for the function to be executed
    :param fun_txt: the actual content of the bat file
    :return:
    """
    batpath = os.path.join(dest_dir, function_name + ".bat")
    with open(batpath, 'w') as OPATH:
        OPATH.writelines(fun_txt)
    subprocess.call([batpath])
    os.remove(batpath)


def compute_volume_clouds(cloud_path_ceiling, cloud_path_floor):
    (cloud_folder, cloud_file) = os.path.split(cloud_path_floor)
    cc_cloud_ceiling = '"' + cloud_path_ceiling + '"'
    cc_cloud_floor = '"' + cloud_path_floor + '"'
    function_name = 'volume'

    function = f' -VOLUME -GRID_STEP 0.2'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
              cc_cloud_ceiling + ' -O ' + cc_cloud_floor + function
    cc_function(cloud_folder, function_name, fun_txt)


def crop_coords(cloud_path, coords, outside=False):
    (cloud_folder, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    function_name = 'crop_complex'
    list_coords_txt = ''

    # number of vertices
    nb_ver = len(coords)

    for i in coords:
        list_coords_txt += (f'{i[0]} {i[1]} ')

    print(list_coords_txt)

    function = f' -CROP2D Z {nb_ver} ' + list_coords_txt
    if outside:
        function += ' -OUTSIDE -RASTERIZE -GRID_STEP 0.05 -EMPTY_FILL INTERP -OUTPUT_CLOUD'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
              cc_cloud + function
    cc_function(cloud_folder, function_name, fun_txt)


def cc_rotate_from_matrix(cloud_path, rot_matrix):
    """
    Function to apply a rotation to a point cloud, using a rotation matrix
    @param cloud_path:
    @param rot_matrix: as a numpy array (3,3)
    """
    (cloud_folder, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    # create the text file including the transformation matrix
    text = ''
    for row in rot_matrix:
        text = text + str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' 0' + '\n'
    text = text + '0 0 0 1'

    file_name = 'rotation.txt'
    txt_path = os.path.join(cloud_folder, file_name)
    # Write the file out again
    with open(txt_path, 'w') as file:
        file.write(text)
    cc_txt_path = '"' + txt_path + '"'

    function_name = 'rotated'
    function = ' -APPLY_TRANS ' + str(cc_txt_path)

    # prepare CloudCompare fonction
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -NO_TIMESTAMP -C_EXPORT_FMT PLY -O ' + cc_cloud + function
    cc_function(cloud_folder, function_name, fun_txt)

    os.remove(txt_path)


def cc_sphericity_linesub(cloud_path, radius, filter_min, filter_max):
    """
    A function to compute the sphericity of the point cloud, filter it according to values, and assign a color to the remaining points
    """

    (cloud_folder, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    function_name = 'sph'
    function = ' -FEATURE SPHERICITY ' + str(radius) + ' -FILTER_SF ' + str(float(filter_min)) + ' ' + str(
        float(filter_max))

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
              cc_cloud + function
    cc_function(cloud_folder, function_name, fun_txt)


# PREPROCESSING FUNCTIONS_______________________________________________________________________________________________
def preproc_ransac_short(cloud_path, min_points, dist, obj_out_dir, cloud_out_dir):
    """
    Function that detects the main planes accross the point cloud
    :param cloud_path: (str) the path to the point cloud
    :param min_points: (int) the minimum points necessary to define a plane
    :param dist: (float) the maximum distance of a point to a candidate plane to be considered as a support for that plane
    :param obj_out_dir: (str) otutput directory for obj files
    :param cloud_out_dir: (str) output directory for ply files
    :return: (int) the number of detected planes
    """
    (cloud_dir, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    function_name = 'ransac'
    function = ' -RANSAC SUPPORT_POINTS ' + str(int(min_points)) + ' EPSILON_ABSOLUTE ' + str(
        dist) + ' OUTPUT_INDIVIDUAL_PAIRED_CLOUD_PRIMITIVE -M_EXPORT_FMT OBJ -SAVE_MESHES -C_EXPORT_FMT PLY -SAVE_CLOUDS -MERGE_MESHES ' \
                '-SAVE_MESHES -MERGE_CLOUDS -SAVE_CLOUDS '

    # Prepare cloudcompare fonction
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)

    # Sort files
    i = 0
    j = 0  # number of planes
    for file in os.listdir(cloud_dir):
        if file != cloud_file:
            if file.endswith('.ply'):
                if not 'MERGED' in file:
                    if 'PLANE' in file:
                        i += 1
                        name = 'plane_' + str(i) + '.ply'
                        os.rename(os.path.join(cloud_dir, file), os.path.join(cloud_out_dir, name))
                else:
                    name = 'plane_merged' + '.ply'
                    os.rename(os.path.join(cloud_dir, file), os.path.join(cloud_out_dir, name))
            if file.endswith('.obj'):
                if not 'MERGED' in file:
                    if 'PLANE' in file:
                        j += 1
                        name = 'plane_' + str(j) + '.obj'
                        os.rename(os.path.join(cloud_dir, file), os.path.join(obj_out_dir, name))
                else:
                    name = 'plane_merged' + '.obj'
                    os.rename(os.path.join(cloud_dir, file), os.path.join(obj_out_dir, name))
            if file.endswith('.bin'):
                to_rem = os.path.join(cloud_dir, file)
                os.remove(to_rem)

    return j


def preproc_floors_from_ransac_merged(ransac_cloud_path, bin_height=0.1):
    pcd = o3d.io.read_point_cloud(ransac_cloud_path)
    points_list = np.asarray(pcd.points)

    # take the z coordinate
    dist = points_list[:, 2]

    # some statistics
    nb_pts = len(dist)
    min_z = np.min(dist)
    max_z = np.max(dist)
    range_z = max_z - min_z  # range of height
    print(f'Minimum z: {min_z} m, Maximum z: {max_z} m, Range: {range_z} m')

    # histogram computation
    nb_bins = round(range_z) / bin_height
    print(f'Number of bins: {nb_bins}')
    density, bins = np.histogram(dist, bins=int(nb_bins))
    mean_peak = np.mean(density)
    print(f'Mean population: {mean_peak} points')

    # plot histogram
    _ = plt.hist(dist, bins=int(nb_bins))
    plt.axhline(1.6 * mean_peak, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    loc = find_peaks(density, height=(1.6 * mean_peak, nb_pts))
    positions = []

    winner_list = loc[0]
    print(f'Winner list: {winner_list}')
    print(density)

    density_length = len(density)
    print(range(density_length))

    for winner in winner_list:
        for k in range(len(density)):
            if k == winner:
                print(k)

    plt.plot(positions)
    plt.show()


def preproc_align_cloud(cloud_path, ransac_obj_folder, ransac_cloud_folder, exclude_txt='MERGED'):
    def compute_rot_matrix(vector):
        a = vector
        b = [1, 0, 0]
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
        return r

    obj_list = generate_list('obj', ransac_obj_folder, exclude=exclude_txt)
    results_vert = find_planes(obj_list, ransac_cloud_folder, orientation='ver_all')
    obj_list_2 = results_vert[2]

    all_x_normals = []
    all_y_normals = []
    all_z_normals = []
    # reminder : plane_data = [cx, cy, cz, w, h, vn[0], vn[1], vn[2]]

    for num, file in enumerate(obj_list_2):
        plane_data = plane_data_light(file)
        all_x_normals.append(plane_data[5])
        all_y_normals.append(plane_data[6])
        all_z_normals.append(plane_data[7])

    hist_x = np.histogram(all_x_normals, bins=200)
    number = hist_x[0]

    m = max(number)
    index = [i for i, j in enumerate(number) if j == m]
    index = index[0]

    bins = hist_x[1]
    vx_to_find = bins[index]

    index_in_initial_list = [i for i, j in enumerate(all_x_normals) if vx_to_find - 0.1 < j < vx_to_find + 0.1]
    index_in_initial_list = index_in_initial_list[0]

    vx = all_x_normals[index_in_initial_list]
    vy = all_y_normals[index_in_initial_list]
    vector = [vx, vy, 0]
    r = compute_rot_matrix(vector)
    print(r)

    cc_rotate_from_matrix(cloud_path, r)
    return r


# TRANSFORM COLORS FUNCTIONS____________________________________________________________________________________________
def cc_recolor(cloud_path, color):
    pass

# 2D RENDERING FUNCTIONS________________________________________________________________________________________________
def raster_single_bound(cloud_path, grid_step, dir_cc, bound_pc, xray=True):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    cc_cloud_lim = '"' + bound_pc + '"'
    proj = ''
    if not xray:
        proj = ' -SF_PROJ MAX -PROJ MAX'

    function_name = 'raster'

    function = ' -AUTO_SAVE OFF -NO_TIMESTAMP -MERGE_CLOUDS -RASTERIZE' + proj + ' -VERT_DIR ' + str(
        dir_cc) + ' -GRID_STEP ' \
               + str(grid_step) + ' -OUTPUT_RASTER_RGB '

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -O ' + cc_cloud_lim + function
    cc_function(cloud_dir, function_name, fun_txt)

def raster_single_bound_height(cloud_path, grid_step, dir_cc, bound_pc):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    cc_cloud_lim = '"' + bound_pc + '"'
    proj = ' -SF_PROJ MAX -PROJ MAX'

    function_name = 'raster'

    function = ' -AUTO_SAVE OFF -NO_TIMESTAMP -MERGE_CLOUDS -RASTERIZE' + proj + ' -VERT_DIR ' + str(
        dir_cc) + ' -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_Z '

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -O ' + cc_cloud_lim + function
    cc_function(cloud_dir, function_name, fun_txt)

def raster_top_rgb_height_pcv(cloud_path, grid_step):
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    proj = ' -SF_PROJ MAX -PROJ MAX'
    add_pcv = ' -PCV -SF_CONVERT_TO_RGB FALSE -RASTERIZE' + proj + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_RGB'

    function_name = 'raster'

    function = ' -AUTO_SAVE OFF -NO_TIMESTAMP -RASTERIZE' + proj + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_RGB -RASTERIZE' + proj + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_Z' + add_pcv

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)

def raster_top_rgb_height(cloud_path, grid_step):
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    proj1 = ' -SF_PROJ MAX -PROJ MAX'
    proj2 = ' -SF_PROJ AVG -PROJ AVG'

    function_name = 'raster'

    function = ' -AUTO_SAVE OFF -NO_TIMESTAMP -RASTERIZE' + proj1 + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_RGB -RASTERIZE' + proj2 + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -EMPTY_FILL INTERP -OUTPUT_RASTER_Z'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)

def render_plane_in_cloud(plane_cloud_path, cloud_path, grid_step):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    cc_mesh = '"' + plane_cloud_path + '"'

    function_name = 'raster'
    function = ' -AUTO_SAVE OFF -SF_CONVERT_TO_RGB FALSE -MERGE_CLOUDS -RASTERIZE -GRID_STEP ' + str(
        grid_step) + ' -OUTPUT_RASTER_RGB'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -REMOVE_ALL_SFS -O  ' + cc_mesh + function
    cc_function(cloud_dir, function_name, fun_txt)


def create_pcv(cloud_path):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'

    function_name = 'pcv'
    function = ' -AUTO_SAVE OFF -C_EXPORT_FMT PLY -PCV -SF_CONVERT_TO_RGB FALSE -SAVE_CLOUDS'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)


def render_planar_segment(cloud_path, grid_step):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    proj = ' -SF_PROJ MAX -PROJ MAX'

    function_name = 'raster'

    function = ' -AUTO_SAVE OFF -BEST_FIT_PLANE -MAKE_HORIZ -SF_CONVERT_TO_RGB FALSE -RASTERIZE' + proj + ' -GRID_STEP ' \
               + str(grid_step) + ' -OUTPUT_RASTER_RGB'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)


def raster_all_bound(cloud_path, grid_step, bound_pc, xray=True, sf=False):
    """
    Function for generating images from a point cloud, from all possible points of view
    :param cloud_path: (str) path to the point cloud to render
    :param grid_step: (float) size of one pixel, in m
    :param bound_pc: (str) path to the bounding box point cloud
    :param xray: (bool) whether to render a xray view
    :param sf: (bool) whether to render one of the scalar fields
    :return:
    """
    # File names and paths
    (cloud_folder, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    cc_cloud_lim = '"' + bound_pc + '"'
    proj = ''
    if not xray:  # Note: the value is AVERAGE by default --> Thus resulting in kind of an xray view
        proj = ' -SF_PROJ MAX -PROJ MAX'

    function_name = 'raster'
    # See memo for understanding views logic
    # order of generated views is : front / right / top / back / left

    function = ' -AUTO_SAVE OFF -MERGE_CLOUDS -RASTERIZE' + proj + ' -VERT_DIR 0 -GRID_STEP ' \
               + str(grid_step) + ' -OUTPUT_RASTER_RGB -RASTERIZE' + proj + ' -VERT_DIR 1 -GRID_STEP ' \
               + str(grid_step) + ' -OUTPUT_RASTER_RGB -RASTERIZE' + proj + ' -VERT_DIR 2 -GRID_STEP ' \
               + str(grid_step) + ' -OUTPUT_RASTER_RGB'
    if not xray:  # the last two views do not need to be generated for xray
        proj = ' -SF_PROJ MIN -PROJ MIN'
        function += ' -RASTERIZE' + proj + ' -VERT_DIR 0 -GRID_STEP ' + str(grid_step) + ' -OUTPUT_RASTER_RGB ' \
                                                                                         ' -RASTERIZE' + proj + ' -VERT_DIR 1 -GRID_STEP ' + str(
            grid_step) + ' -OUTPUT_RASTER_RGB '

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -O ' + cc_cloud_lim + function
    cc_function(cloud_folder, function_name, fun_txt)


# CUT SECTIONS__________________________________________________________________________________________________________
def cut_sections_write_xml(file_name, center_x, size_x, center_y, size_y, center_z, size_z, repeat, gap, output_dir):
    # Folder params

    # get the repetition dimension and adapt for cloudcompare
    if repeat == 'x':
        param_rep = 0
    elif repeat == 'y':
        param_rep = 1
    else:
        param_rep = 2

    # Write XML file
    text = '<CloudCompare> \n <BoxThickness x="' + str(size_x) + '" y ="' + str(size_y) + '" z ="' + \
           str(size_z) + '"/> \n <BoxCenter x="' + str(center_x) + '" y ="' + str(center_y) + '" z ="' \
           + str(center_z) + '"/> \n <RepeatDim>' + str(param_rep) + '</RepeatDim> \n <RepeatGap>' + str(gap) + \
           '</RepeatGap> \n' \
           '<OutputFilePath>' + output_dir + '</OutputFilePath> \n </CloudCompare>'

    xml_path = os.path.join(output_dir, file_name)
    # Write the file out again
    with open(xml_path, 'w') as file:
        file.write(text)
    return xml_path


def cut_sections_all_dirs(cloud_path, center_x, size_x, center_y, size_y, center_z, size_z, gap,
                          output_dir_x, output_dir_y, output_dir_z):
    # Folder params
    (cloud_folder, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'

    function_txt = ''
    function_name = 'cut'

    # Write XML file
    xml_files = []

    output_folders = [output_dir_x, output_dir_y, output_dir_z]
    orientation = ['x', 'y', 'z']
    for i in range(3):
        if i == 0:
            xml_path = cut_sections_write_xml('cut_x.xml', center_x, 0.1, center_y, size_y, center_z, size_z,
                                              orientation[i],
                                              gap, output_folders[i])
        if i == 1:
            xml_path = cut_sections_write_xml('cut_y.xml', center_x, size_x, center_y, 0.1, center_z, size_z,
                                              orientation[i],
                                              gap, output_folders[i])
        if i == 2:
            xml_path = cut_sections_write_xml('cut_z.xml', center_x, size_x, center_y, size_y, center_z, 0.1,
                                              orientation[i],
                                              gap, output_folders[i])

        cc_xml = '"' + xml_path + '"'
        function_txt = function_txt + ' -CROSS_SECTION ' + cc_xml
        xml_files.append(xml_path)

    fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -NO_TIMESTAMP  -O ' \
              + cc_cloud + function_txt

    cc_function(cloud_folder, function_name, fun_txt)
    os.remove(xml_path)


# ANALYTIC FUNCTIONS____________________________________________________________________________________________________
def cc_planes_to_build_dist_list(cloud_path, obj_list, cloud_out_folder, dist=0.1, span=0.03):
    """A function that open all plane meshes in a directory, compares it to the original building cloud

    Two inputs:
        folder -- the directory containing the .obj planes
        dist -- criteria of comparison (float)

    outputs individual point planes that keep color information

    """
    (cloud_dir, cloud_file) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    function_name = 'c2m'

    function_load_compare = ''
    for i, obj_file in enumerate(obj_list):
        mesh_name = obj_file
        cc_mesh = '"' + mesh_name + '"'
        function_load_compare = ' -O ' + cc_mesh + ' -C2M_DIST -MAX_DIST ' + str(dist) + ' -FILTER_SF ' + str(
            -float(dist) + span) + ' ' + str(
            float(dist) - span) + ' -SAVE_CLOUDS -CLEAR_MESHES '

        # Prepare CloudCompare fonction
        fun_txt = 'SET MY_PATH="' + CC_PATH + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT LAS -AUTO_SAVE OFF -O ' + cc_cloud + function_load_compare

        # Execute function
        cc_function(cloud_dir, function_name, fun_txt)

    # sort files
    i = 0
    for file in os.listdir(cloud_dir):
        if file.endswith('.las'):
            if 'C2M_DIST' in file:
                i += 1
                name = 'segment_' + str(i) + '.las'
                os.rename(os.path.join(cloud_dir, file), os.path.join(cloud_out_folder, name))

    return i


def plane_data_light(obj_path):
    """
    Function to get the size, orientation and position of a plane obj
    Returns an 'plane_data' array
    cx, cy, cz -- x, y ,z coord of plane center
    w, h -- width and height of the plane
    vn -- vector representing the normal of the plane
    @parameters:
        obj_path -- path to a specific plane in obj format
    """
    v = []
    vn = []
    mylines = []
    with open(obj_path, 'rt') as myfile:
        for myline in myfile:  # For each line, read it to a string
            mylines.append(myline)
            if 'v ' in myline:
                myline = myline[2:-2]
                v.append(myline.split(' '))
            if 'vn' in myline:
                myline = myline[3:-2]
                vn = myline.split(' ')

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    v4 = v[3]

    x = [float(v1[0]), float(v2[0]), float(v3[0]), float(v4[0])]
    y = [float(v1[1]), float(v2[1]), float(v3[1]), float(v4[1])]
    z = [float(v1[2]), float(v2[2]), float(v3[2]), float(v4[2])]
    cx = (x[0] + x[2]) / 2
    cy = (y[0] + y[2]) / 2
    cz = (z[0] + z[2]) / 2
    w = math.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2)
    h = math.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2 + (z[1] - z[2]) ** 2)
    vn = [float(vn[0]), float(vn[1]), float(vn[2])]
    plane_data = [cx, cy, cz, w, h, vn[0], vn[1], vn[2]]

    return plane_data


def plane_data_complete(obj_path, cloud_path):
    """
    Function to get the size, orientation and position of a plane obj, plus additional information gathered from the
    corresponding point cloud By default, it is expected that the point cloud has the same name as the obj
    """
    (obj_folder, obj_file) = os.path.split(obj_path)
    # Read obj plane info
    v = []
    vn = []
    mylines = []
    with open(obj_path, 'rt') as myfile:
        for myline in myfile:  # For each line, read it to a string
            mylines.append(myline)
            if 'v ' in myline:
                myline = myline[2:-2]
                v.append(myline.split(' '))
            if 'vn' in myline:
                myline = myline[3:-2]
                vn = myline.split(' ')

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    v4 = v[3]

    x = [float(v1[0]), float(v2[0]), float(v3[0]), float(v4[0])]
    y = [float(v1[1]), float(v2[1]), float(v3[1]), float(v4[1])]
    z = [float(v1[2]), float(v2[2]), float(v3[2]), float(v4[2])]
    cx = (x[0] + x[2]) / 2
    cy = (y[0] + y[2]) / 2
    cz = (z[0] + z[2]) / 2
    dim1 = math.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2 + (z[0] - z[1]) ** 2)
    dim2 = math.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2 + (z[1] - z[2]) ** 2)
    area = dim1 * dim2
    vn = [float(vn[0]), float(vn[1]), float(vn[2])]
    plane_data = [x, y, z, cx, cy, cz, dim1, dim2, area, vn[0], vn[1], vn[2]]

    pcd = o3d.io.read_point_cloud(cloud_path)
    points_list = np.asarray(pcd.points)
    color_list = np.asarray(pcd.colors)
    n_points = len(points_list[:, 1])
    average_color = [math.sqrt(np.mean(color_list[:, 0] ** 2)), math.sqrt(np.mean(color_list[:, 1] ** 2)),
                     math.sqrt(np.mean(color_list[:, 2] ** 2))]

    plane_data.append(n_points)
    plane_data.extend(average_color)

    return plane_data


def find_planes(obj_list, cloud_folder, cloud_suffix='', orientation='all', size_absolute='all', size_criteria=10,
                size_comparative='all', location_relative='all', location_option='all_winners', uniformity='all'):
    """
    A central function to detect planes that respect a criterion, or a series of criteria
    :param obj_folder:
    :param
        orientation : all, hor, ver_all, ver_x, ver_y, obl
        size_absolute : area_greater_than, area_smaller_than
        size_comparative : all, area_largest, area_smallest etc.
        location_relative : all, center_extreme-x, center_extreme+x, center_extreme-z, center_and_vertex_extreme-x, ...
        relationship : all, close_..cm_parallel
        paint_level : all
        mean_point_density : all
    :return:
    """
    all_planes_data = []
    criteria_list = [orientation, size_absolute, size_comparative, location_relative, uniformity]

    # Activate search functions
    for num, criteria in enumerate(criteria_list):
        if criteria != 'all':
            criteria_list[num] = 1

    # Create empty matrices for activated functions
    if criteria_list[0] == 1:  # Orientation
        orientation_results_list = []
    if criteria_list[1]:  # Size absolute
        size_results_list = [0]
        area_list = [0]
        dim_list = [0]
    if criteria_list[2] == 1:  # Size relative
        size_results_list = [0]
        area_list = [0]
        dim_list = [0]
    if criteria_list[3] == 1:  # Location
        loc_results_list = []
        c_pos1_list = []  # list of positions of the center of candidate planes (along dim 1)
        c_pos2_list = []  # list of positions of the center of candidate planes (along dim 2)
        c_pos3_list = []  # list of positions of the center of candidate planes (along dim 3)
        v_pos1_max_list = []
        v_pos1_min_list = []
        v_pos2_max_list = []
        v_pos2_min_list = []
        v_pos3_max_list = []
        v_pos3_min_list = []

    # Get data of planes
    for num, file in enumerate(obj_list):
        (obj_folder, obj_file) = os.path.split(file)
        ply_file_name = obj_file[:-4] + cloud_suffix + 'ply'
        ply_file_loc = os.path.join(cloud_folder, ply_file_name)
        all_planes_data.append(plane_data_complete(file, ply_file_loc))

    #  Iterate through all planes
    for i, obj_file in enumerate(obj_list):
        plane_prop = all_planes_data[i]

        # Should we do orientation analyses?__________________________________
        if criteria_list[0] == 1:
            # test criteria orientation, based on the normal vector
            if orientation == 'hor':
                if abs(plane_prop[11]) > 0.998:
                    orientation_results_list.append(obj_file)
            elif orientation == 'ver_x':
                if abs(plane_prop[9]) > 0.998:
                    orientation_results_list.append(obj_file)
            elif orientation == 'ver_y':
                if abs(plane_prop[10]) > 0.998:
                    orientation_results_list.append(obj_file)
            elif orientation == 'ver_all':
                if abs(plane_prop[11]) < 0.01:
                    orientation_results_list.append(obj_file)
            elif orientation == 'obl':
                if abs(plane_prop[9]) < 0.998 and abs(plane_prop[10]) < 0.998 and abs(plane_prop[11]) < 0.998:
                    orientation_results_list.append(obj_file)

        print(orientation_results_list)
        # Should we do size analyses?_________________________________________
        if criteria_list[1] or criteria_list[2] == 1:
            # Get data from current plane
            area = plane_prop[8]  # Area
            dim1 = plane_prop[6]  # length 1
            dim2 = plane_prop[7]  # length 2

            # Should we do absolute size analysed?
            if criteria_list[1] == 1:
                if size_absolute == 'area_greater_than':
                    if area > size_criteria:
                        size_results_list.append(obj_file)
                if size_absolute == 'area_smaller_than':
                    if area < size_criteria:
                        size_results_list.append(obj_file)
                if size_absolute == 'largest_dim_larger_than':
                    if dim1 > size_criteria or dim2 > size_criteria:
                        size_results_list.append(obj_file)

            # Should we do comparative size analyses?
            if criteria_list[2] == 1:
                if size_comparative == 'area_largest':
                    if area > area_list[0]:
                        area_list[0] = area
                        size_results_list = obj_file
                if size_comparative == 'area_smallest':
                    if area < (area_list[0]):
                        area_list[0] = area
                        size_results_list = obj_file

    if criteria_list[1] and criteria_list[2] != 1:
        size_results_list = obj_list
    cross_results_list = list(set(orientation_results_list).intersection(size_results_list))

    print(cross_results_list)

    # Should we do location analysis?_________________________________________
    all_planes_data = []
    if criteria_list[3] == 1:
        reduced_obj_list = cross_results_list  # Start from the result of orientation and size analysis
        for num, file in enumerate(reduced_obj_list):
            (obj_folder, obj_file) = os.path.split(file)
            ply_file_name = obj_file[:-4] + cloud_suffix + 'ply'
            ply_file_loc = os.path.join(cloud_folder, ply_file_name)
            all_planes_data.append(plane_data_complete(file, ply_file_loc))

        # Start from reduced list of planes
        for i, obj_file in enumerate(reduced_obj_list):
            plane_prop = all_planes_data[i]
            # Position of vertices
            v_pos_x = plane_prop[0]
            v_pos_y = plane_prop[1]
            v_pos_z = plane_prop[2]

            # Position of centers
            c_pos_x = plane_prop[3]
            c_pos_y = plane_prop[4]
            c_pos_z = plane_prop[5]

            # Create lists that compile location information
            if location_relative == 'center_extreme-x':  # Only with vertical planes
                c_pos1_list.append(c_pos_x)
                c_pos2_list.append(c_pos_y)
                c_pos3_list.append(c_pos_z)
                v_pos1_max_list.append(max(v_pos_x))
                v_pos1_min_list.append(min(v_pos_x))
                v_pos2_max_list.append(max(v_pos_y))
                v_pos2_min_list.append(min(v_pos_y))
                v_pos3_max_list.append(max(v_pos_z))
                v_pos3_min_list.append(min(v_pos_z))
            if location_relative == 'center_extreme+x':  # Only with vertical planes
                c_pos1_list.append(-c_pos_x)
                c_pos2_list.append(c_pos_y)
                c_pos3_list.append(c_pos_z)
                v_pos1_max_list.append(max(v_pos_x))
                v_pos1_min_list.append(min(v_pos_x))
                v_pos2_max_list.append(max(v_pos_y))
                v_pos2_min_list.append(min(v_pos_y))
                v_pos3_max_list.append(max(v_pos_z))
                v_pos3_min_list.append(min(v_pos_z))
            if location_relative == 'center_extreme-y':  # Only with vertical planes
                c_pos1_list.append(c_pos_y)
                c_pos2_list.append(c_pos_x)
                c_pos3_list.append(c_pos_z)
                v_pos1_max_list.append(max(v_pos_y))
                v_pos1_min_list.append(min(v_pos_y))
                v_pos2_max_list.append(max(v_pos_x))
                v_pos2_min_list.append(min(v_pos_x))
                v_pos3_max_list.append(max(v_pos_z))
                v_pos3_min_list.append(min(v_pos_z))
            if location_relative == 'center_extreme+y':  # Only with vertical planes
                c_pos1_list.append(-c_pos_y)
                c_pos2_list.append(c_pos_x)
                c_pos3_list.append(c_pos_z)
                v_pos1_max_list.append(max(v_pos_y))
                v_pos1_min_list.append(min(v_pos_y))
                v_pos2_max_list.append(max(v_pos_x))
                v_pos2_min_list.append(min(v_pos_x))
                v_pos3_max_list.append(max(v_pos_z))
                v_pos3_min_list.append(min(v_pos_z))
            if location_relative == 'center_extreme-z':  # Only with horizontal planes
                c_pos1_list.append(c_pos_z)
                c_pos2_list.append(c_pos_x)
                c_pos3_list.append(c_pos_y)
                v_pos1_max_list.append(max(v_pos_z))
                v_pos1_min_list.append(min(v_pos_z))
                v_pos2_max_list.append(max(v_pos_x))
                v_pos2_min_list.append(min(v_pos_x))
                v_pos3_max_list.append(max(v_pos_y))
                v_pos3_min_list.append(min(v_pos_y))
            if location_relative == 'center_extreme+z':  # Only with horizontal planes
                c_pos1_list.append(-c_pos_z)
                c_pos2_list.append(c_pos_x)
                c_pos2_list.append(c_pos_y)
                v_pos1_max_list.append(max(v_pos_z))
                v_pos1_min_list.append(min(v_pos_z))
                v_pos2_max_list.append(max(v_pos_x))
                v_pos2_min_list.append(min(v_pos_x))
                v_pos3_max_list.append(max(v_pos_y))
                v_pos3_min_list.append(min(v_pos_y))

        # Finalize location analysis
        def compare_overlap(dim1_min_plane1, dim1_max_plane1, dim2_min_plane1, dim2_max_plane1, dim1_min_plane2,
                            dim1_max_plane2, dim2_min_plane2, dim2_max_plane2):
            common_area = (min(dim1_max_plane1, dim1_max_plane2) - max(dim1_min_plane1, dim1_min_plane2)) * \
                          (min(dim2_max_plane1, dim2_max_plane2) - max(dim2_min_plane1, dim2_min_plane2))
            overlap = common_area / ((dim1_max_plane1 - dim1_min_plane1) * (dim2_max_plane1 - dim2_min_plane1))
            return overlap

        if location_option != 'all_winners':  # Here only planes with extreme location are looked for
            min_to_beat = max(v_pos1_max_list)
            for i, obj_file in enumerate(reduced_obj_list):
                if i > 0:
                    if v_pos1_min_list[i] < min_to_beat:
                        min_to_beat = v_pos1_min_list[i]
                        loc_results_list = obj_file
        else:
            failed = 0  # If failed value = 0, then the candidate plane is rejectedµ
            print('___________________________1 Initialialized')
            for i in range(len(reduced_obj_list)):
                print('___________________________2 Plane analyzed :')
                print(reduced_obj_list[i])

                dim2_lim = [v_pos2_min_list[i], v_pos2_max_list[i]]  # If we do an analysis along x axis, for
                # example, the dim2 is y and dim 3 is z
                dim3_lim = [v_pos3_min_list[i], v_pos3_max_list[i]]

                for j in range(len(reduced_obj_list)):
                    print('___________________________2 Plane compared :')
                    print(reduced_obj_list[j])
                    dim2_lim2 = [v_pos2_min_list[j], v_pos2_max_list[j]]
                    dim3_lim2 = [v_pos3_min_list[j], v_pos3_max_list[j]]
                    if i != j:  # To reject self comparison
                        if c_pos1_list[i] < c_pos1_list[j]:
                            continue
                        else:
                            if c_pos2_list[j] < dim2_lim[0] + 0.5 or c_pos2_list[j] > dim2_lim[1] - 0.5:
                                ov = compare_overlap(dim2_lim[0], dim2_lim[1], dim3_lim[0], dim3_lim[1], dim2_lim2[0],
                                                     dim2_lim2[1], dim3_lim2[0], dim3_lim2[1])
                                if ov < 0.30:
                                    continue
                                else:
                                    failed = 1
                                    print('FAILED type A : overlap ')
                                    print(ov)
                            else:
                                if c_pos3_list[j] < dim3_lim[0] + 0.5 or c_pos3_list[j] > dim3_lim[1] - 0.5:
                                    ov = compare_overlap(dim2_lim[0], dim2_lim[1], dim3_lim[0], dim3_lim[1],
                                                         dim2_lim2[0], dim2_lim2[1], dim3_lim2[0], dim3_lim2[1])
                                    if ov < 0.30:
                                        continue
                                    else:
                                        failed = 1
                                        print('FAILED type B')
                                else:
                                    failed = 1
                                    print('FAILED type C')

                if failed == 0:  # If the candidate plane passes the test, it is added to the 'winner' list
                    loc_results_list.append(reduced_obj_list[i])
                else:
                    failed = 0
    else:
        loc_results_list = cross_results_list

    # Final compilation of results
    final_list = loc_results_list
    plane_list = [obj_list, criteria_list, final_list]
    return plane_list


"""
======================================================================================
ALGEBRA
======================================================================================
"""


# definition of rotation matrices, useful for camera operations
def rot_x_matrix(angle):
    matrix = np.asarray([[1, 0, 0, 0],
                         [0, math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                         [0, math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                         [0, 0, 0, 1]])
    return matrix


def rot_y_matrix(angle):
    matrix = np.asarray([[math.cos(math.radians(angle)), 0, math.sin(math.radians(angle)), 0],
                         [0, 1, 0, 0],
                         [-math.sin(math.radians(angle)), 0, math.cos(math.radians(angle)), 0],
                         [0, 0, 0, 1]])
    return matrix


def rot_z_matrix(angle):
    matrix = np.asarray([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0, 0],
                         [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return matrix


def front_mat():
    matrix = rot_x_matrix(-90)
    inv_matrix = rot_x_matrix(90)
    return matrix, inv_matrix


def back_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(180)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(-180)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def right_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(-90)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(90)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def left_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(90)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(-90)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def iso2_mat():
    matrix1 = rot_x_matrix(60)
    matrix2 = rot_y_matrix(-20)
    matrix3 = rot_z_matrix(190)
    final_matrix1 = matrix3 @ matrix2
    final_matrix = final_matrix1 @ matrix1
    inv_matrix1 = rot_z_matrix(-190)
    inv_matrix2 = rot_y_matrix(20)
    inv_matrix3 = rot_x_matrix(-60)
    final_inv_matrix1 = inv_matrix3 @ inv_matrix2
    final_inv_matrix = final_inv_matrix1 @ inv_matrix1
    return final_matrix, final_inv_matrix


def iso1_mat():
    matrix1 = rot_x_matrix(-60)
    matrix2 = rot_y_matrix(-20)
    matrix3 = rot_z_matrix(-10)
    final_matrix1 = matrix3 @ matrix2
    final_matrix = final_matrix1 @ matrix1
    inv_matrix1 = rot_z_matrix(10)
    inv_matrix2 = rot_y_matrix(20)
    inv_matrix3 = rot_x_matrix(60)
    final_inv_matrix1 = inv_matrix3 @ inv_matrix2
    final_inv_matrix = final_inv_matrix1 @ inv_matrix1
    return final_matrix, final_inv_matrix


def name_to_matrix(orientation):
    if orientation == 'iso_front':
        trans_init, inv_trans = iso1_mat()
    elif orientation == 'iso_back':
        trans_init, inv_trans = iso2_mat()
    elif orientation == 'left':
        trans_init, inv_trans = left_mat()
    elif orientation == 'right':
        trans_init, inv_trans = right_mat()
    elif orientation == 'front':
        trans_init, inv_trans = front_mat()
    elif orientation == 'back':
        trans_init, inv_trans = back_mat()

    return trans_init, inv_trans


"""
======================================================================================
2D OPS
======================================================================================
"""


def crop_and_save(image_path):
    image = Image.open(image_path)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    image.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))

    image.save(image_path[:-4] + 'crop.jpg')


def get_nonzero_coord(image_path):
    image = Image.open(image_path)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    xmin = np.min(x_nonzero)
    ymin = np.min(y_nonzero)
    xmax = np.max(x_nonzero)
    ymax = np.max(y_nonzero)

    return xmin, xmax, ymin, ymax


def mask_image_with_shape(original_img, mask_img):
    result_image = np.ones_like(original_img) * 255

    # Define the tolerance range
    tolerance_lower = 0
    tolerance_upper = 10

    # Create a binary mask based on the tolerance range
    interest_mask = cv2.inRange(mask_img, tolerance_lower, tolerance_upper)

    # Copy the interest region from the color image to the result image
    result_image[interest_mask == 255] = original_img[interest_mask == 255]

    return result_image


def create_pc_from_elevation_coords(elevation, rgb_path, coords, res):
    """

    :param elevation: a numpy array with elevation values (shape (X,Y))
    :param rgb_path: a path to a RGB image (shape of the resulting array (X,Y,3)
    :param coords: a list of coordinates creating a closed polygons
    :param res: resolution in m/pixel
    :return:
    """
    # 0. Read RGB Data
    image = Image.open(rgb_path)
    # convert image to numpy array
    rgb = np.asarray(image)

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    # Get the dimensions of both images
    rgb_shape = rgb.shape
    elevation_shape = elevation.shape

    # Check if the shapes are not the same
    if rgb_shape != elevation_shape:
        # Determine which image is larger
        if rgb_shape[0] > elevation_shape[0] or rgb_shape[1] > elevation_shape[1]:
            # Resize the RGB image to match the elevation image
            img_rgb = image.resize((elevation_shape[1], elevation_shape[0]))
            rgb = np.asarray(img_rgb)


    # 1. Create a path object from the polygon coordinates
    path = mpath.Path(coords)

    # 2. Create a boolean mask for the points inside the polygon
    y, x = np.mgrid[:elevation.shape[0], :elevation.shape[1]]
    coords = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(coords).reshape(elevation.shape)

    # 3. Extract the height/rgb values from the numpy array using the mask
    heights_inside_polygon = elevation[mask]
    rgb_inside_polygon = rgb[mask]

    # 4. Create the XYZ point cloud
    x_coords, y_coords = np.where(mask)

    # Convert pixel coordinates to meters
    x_coords_meters = x_coords * res
    y_coords_meters = y_coords * res

    xyz_points = np.column_stack((x_coords_meters, y_coords_meters, heights_inside_polygon))
    color_array = rgb_inside_polygon
    print(color_array)

    return xyz_points, color_array

def convert_mask_polygon(image_path, original_rgb, dest_poly_path, dest_crop_poly_path, dest_poly_rgb_path, dest_poly_rgb_crop_path):
    mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_binary = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming it's the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    contour = np.squeeze(largest_contour)

    polygon = Polygon(contour)
    print('ok buffering polygon')
    buffered_polygon = polygon.buffer(0.2, join_style=2)
    # large_buffered_polygon = polygon.buffer(0.5, join_style=2)

    coords = sh.get_coordinates(buffered_polygon)

    # create a white image with the detected shape as black
    output_image = np.ones_like(mask_image) * 255
    filled_poly = cv2.drawContours(output_image, [largest_contour], -1, 0, thickness=cv2.FILLED)

    filled_rgb_poly = mask_image_with_shape(original_rgb, filled_poly)

    # Find the bounding box of the filled contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define the desired margin
    margin = 10

    # Adjust the margin for x, y, w, h to ensure they stay within the image bounds
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, output_image.shape[1] - x)
    h = min(h + 2 * margin, output_image.shape[0] - y)

    # Crop the image to the bounding box with the margin
    cropped_poly = output_image[y:y + h, x:x + w]
    cropped_rgb_poly = filled_rgb_poly[y:y + h, x:x + w]

    cv2.imwrite(dest_poly_path, filled_poly)
    cv2.imwrite(dest_crop_poly_path, cropped_poly)
    cv2.imwrite(dest_poly_rgb_path, filled_rgb_poly)
    cv2.imwrite(dest_poly_rgb_crop_path, cropped_rgb_poly)

    # compute area (in pixels squared)
    area = count_black_pixels(cropped_poly)

    yolo_type_bbox = [x,y,x+w,y+h, 0, 0]

    return coords, area, yolo_type_bbox


def convert_mask_polygon_old(image_path, dest_path1, dest_path2):
    img_folder, _ = os.path.split(image_path)
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # get convex hull
    points = np.column_stack(np.where(img_binary.transpose() > 0))
    hull = cv2.convexHull(points)

    # draw convex hull on input image in green
    result = img_color.copy()
    cv2.polylines(result, [hull], True, (0, 0, 255), 2)

    # draw white filled hull polygon on black background
    mask = np.zeros_like(img_binary)
    result2 = cv2.fillPoly(mask, [hull], 255)

    # get the largest contour from result2
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw contour on copy of input
    contr = img_color.copy()
    contr = cv2.drawContours(contr, [big_contour], 0, (0, 0, 255), 2)

    # save result2
    cv2.imwrite(dest_path1, result)
    cv2.imwrite(dest_path2, result2)

    print(big_contour)
    contour = np.squeeze(big_contour)
    polygon = Polygon(contour)
    print('ok buffering polygon')
    buffered_polygon = polygon.buffer(0.2, join_style=2)
    # large_buffered_polygon = polygon.buffer(0.5, join_style=2)

    coords = sh.get_coordinates(buffered_polygon)

    print(f'final coords:{coords}')

    return buffered_polygon, coords


def convert_coord_img_to_cloud_topview(coords, cloud_res, center, dim):
    new_coords = []
    for coord in coords:
        x = int(coord[0]) * cloud_res / 1000
        y = int(coord[1]) * cloud_res / 1000

        new_x = center[0] - dim[0] / 2 + x
        new_y = center[1] + dim[1] / 2 - y

        new_coord = [new_x, new_y]
        new_coords.append(new_coord)

    return new_coords


def delete_elements_by_indexes(original_list, indexes_to_delete):
    # Sort the indexes in descending order to avoid index shifting during deletion
    indexes_to_delete.sort(reverse=True)

    for index in indexes_to_delete:
        if 0 <= index < len(original_list):
            del original_list[index]


def count_black_pixels(image):
    # Define the black color range (since we're using grayscale, black pixels have intensity 0)
    black_lower = 0
    black_upper = 10  # Allow a small range to account for noise

    # Count the black pixels
    black_pixel_count = 0
    for row in image:
        for pixel_value in row:
            if black_lower <= pixel_value <= black_upper:
                black_pixel_count += 1

    return black_pixel_count


def generate_summary_canva_from_inv():
    pass


def generate_summary_canva(image_list, name_list, dest_path):
    # Sort images based on pixel area
    sorted_images_and_names = sorted(zip(image_list, name_list), key=lambda x: np.sum(x[0] == 0))

    # Calculate total width and maximum height for the inventory picture
    total_width = sum(img.shape[1] for img, _ in sorted_images_and_names)
    max_height = max(img.shape[0] + 30 for img, _ in sorted_images_and_names)

    # Create a blank canvas for the inventory picture
    inventory_picture = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255  # White background
    current_x = 0

    # Place each image on the inventory picture along with its name
    for img, name in sorted_images_and_names:
        h, w = img.shape[:2]
        inventory_picture[:h, current_x:current_x + w] = img

        # Write the image name below the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)  # Black text color
        text_size = cv2.getTextSize(name, font, font_scale, 1)[0]
        text_x = current_x + (w - text_size[0]) // 2
        text_y = h + text_size[1] + 10  # Some padding below the image
        cv2.putText(inventory_picture, name, (text_x, text_y), font, font_scale, font_color, 1, cv2.LINE_AA)

        current_x += w

    cv2.imwrite(dest_path, inventory_picture)

def create_mixed_elevation_views(elevation_path, hillshade_path, rgb_path, dest_path1, dest_path2):
    img = Image.open(rgb_path)
    if img.mode == 'RGB':
        # Convert the image to RGBA format
        img_rgba = img.convert('RGBA')
    else:
        img_rgba = img

    rgb = np.asarray(img_rgba)
    elevation = np.asarray(Image.open(elevation_path))
    hillshade = np.asarray(Image.open(hillshade_path))

    # Get the dimensions of both images
    rgb_shape = rgb.shape
    elevation_shape = elevation.shape

    # Check if the shapes are not the same
    if rgb_shape != elevation_shape:
        # Determine which image is larger
        if rgb_shape[0] > elevation_shape[0] or rgb_shape[1] > elevation_shape[1]:
            # Resize the RGB image to match the elevation image
            img_rgba = img_rgba.resize((elevation_shape[1], elevation_shape[0]))
            rgb = np.asarray(img_rgba)
        else:
            # Resize the elevation image to match the RGB image
            elevation_img = Image.open(elevation_path)
            elevation_img = elevation_img.resize((rgb_shape[1], rgb_shape[0]))
            elevation = np.asarray(elevation_img)

    foreground = hillshade  # Inputs to blend_modes need to be numpy arrays.
    foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.
    background = elevation
    background_float = background.astype(float)
    blended = hard_light(background_float, foreground_float, 1)

    blended_img = np.uint8(blended)
    blended_img_raw = Image.fromarray(blended_img)
    blended_img_raw = blended_img_raw.convert('RGB')
    blended_img_raw.save(dest_path1)

    foreground = hillshade  # Inputs to blend_modes need to be numpy arrays.
    foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.
    background = rgb
    background_float = background.astype(float)
    blended = hard_light(background_float, foreground_float, 1)

    blended_img = np.uint8(blended)
    blended_img_raw = Image.fromarray(blended_img)
    blended_img_raw = blended_img_raw.convert('RGB')
    blended_img_raw.save(dest_path2)


def create_elevation(dtm_path, dest_path, high_limit=0, low_limit=0, type='standard'):
    """

    :param dtm_path: path to the DTM as a tif
    :param dest_path: path to save the image to
    :param type: can be 'standard' (only height colormap), 'hill'
    :return:
    """
    with rio.open(dtm_path) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan

    # filter values
    if low_limit != 0:
        elevation[elevation <= low_limit] = low_limit

    if low_limit != 0:
        elevation[elevation >= high_limit] = high_limit

    # interpolation


    # Plot the altitude data with the colormap
    if type =='standard':
        cmap = plt.get_cmap("terrain")
        plt.imsave(fname=dest_path, arr=elevation, cmap=cmap, vmin=np.nanmin(elevation), vmax=np.nanmax(elevation))
    elif type == 'hill':
        hillshade = es.hillshade(elevation, altitude=0)
        plt.imsave(fname=dest_path, arr=hillshade, cmap='Greys', vmin=np.nanmin(hillshade), vmax=np.nanmax(hillshade))
    elif type == 'pcv':
        visibility = spcv.compute_sky_visibility(elevation)
        result = spcv.export_results(visibility, -1,1, 2,0.2,standardize=False)
        cv2.imwrite(dest_path,result)


    # Close the figure
    plt.close()


def create_ground_map(elevation, mask_array, dest_path=''):
    tolerance_lower = 0
    tolerance_upper = 10

    # create a true intensity image
    mask_array = mask_array[:, :, 2]
    interest_mask = cv2.inRange(mask_array, tolerance_lower, tolerance_upper)

    mask_non_zero_coords = np.argwhere(interest_mask != 0)
    for coord in mask_non_zero_coords:
        x, y = coord
        elevation[x, y] = np.nan

    # Coordinates of known (non-missing) pixels
    known_coords = np.argwhere(~np.isnan(elevation))

    # Values of known pixels
    known_values = elevation[~np.isnan(elevation)]

    # Coordinates of missing pixels
    missing_coords = np.argwhere(np.isnan(elevation))

    # Perform interpolation
    interpolated_values = griddata(known_coords, known_values, missing_coords, method='nearest')

    # Assign the interpolated values to the image array
    for i in range(len(missing_coords)):
        x, y = missing_coords[i]
        elevation[x, y] = interpolated_values[i]

    # save resulting plot and new elevation
    cmap = plt.get_cmap("terrain")
    # plt.imsave(fname=dest_path, arr=elevation, cmap=cmap, vmin=np.nanmin(elevation), vmax=np.nanmax(elevation))

def compute_volume(elevation, ground, mask_array, res):
    """
    plt.figure(figsize=(10, 20))  # Set the figure size (width, height) in inches
    plt.imshow(elevation, cmap='terrain', origin='lower')  # Show the array, using a terrain colormap
    plt.colorbar(label='Elevation')  # Add a color bar on the side, labeled "Elevation"
    plt.title('Elevation Map')  # Add title
    plt.xlabel('X Coordinate')  # X-axis label
    plt.ylabel('Y Coordinate')  # Y-axis label
    plt.show()

    plt.figure(figsize=(10, 20))  # Set the figure size (width, height) in inches
    plt.imshow(ground, cmap='terrain', origin='lower')  # Show the array, using a terrain colormap
    plt.colorbar(label='Elevation')  # Add a color bar on the side, labeled "Elevation"
    plt.title('Ground Map')  # Add title
    plt.xlabel('X Coordinate')  # X-axis label
    plt.ylabel('Y Coordinate')  # Y-axis label
    plt.show()
    """
    tolerance_lower = 0
    tolerance_upper = 10

    # create a true intensity image
    mask_array = mask_array[:, :, 2]
    interest_mask = cv2.inRange(mask_array, tolerance_lower, tolerance_upper)

    mask_non_zero_coords = np.argwhere(interest_mask != 0)

    volume = 0
    for coord in mask_non_zero_coords:
        x, y = coord
        z_diff = elevation[x,y] - ground[x,y]
        value = z_diff*res*res
        volume += value

    return volume