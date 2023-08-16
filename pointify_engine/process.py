import statistics
import numpy as np
import open3d as o3d
import os
import subprocess
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image
import shapely as sh
from shapely.geometry import Polygon

import cv2

from PySide6 import QtCore, QtGui, QtWidgets


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


# PATHS FUNCTIONS_______________________________________________________________________________________________________
cc_path = os.path.join("C:\\", "Program Files", "CloudCompare", "CloudCompare")  # path to cloudcompare exe


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


# OPEN3D FUNCTIONS______________________________________________________________________________________________________
def create_box_limit(pcd, output_path):
    bound = pcd.get_oriented_bounding_box()
    points_bb = bound.get_box_points()
    points_bb_np = np.asarray(points_bb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = points_bb
    pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # write point cloud
    o3d.io.write_point_cloud(output_path, pcd)


def compute_basic_properties(pc_load, save_bound_pc=False, output_path_bound_pc='', compute_full_density=False):
    """
    A function that compiles all the basic properties of the point cloud
    :param pc_load: (open3D pc) the point cloud object, loaded into open3D
    :param save_bound_pc: (bool) Whether to save the bounding box as a point cloud
    :param output_path_bound_pc: (str) the path where to save the bounding box point cloud
    :param compute_density: (bool) whether to compute the density on the whole the point cloud
    :return:bound = the bounding box, center= the center of the bounding box, dim= the extend of the bounding box, n_points= the number of points
    """
    # bounding box
    bound = pc_load.get_axis_aligned_bounding_box()
    center = bound.get_center()
    dim = bound.get_extent()
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

    return bound, points_bb_np, center, dim, density, n_points


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


# CLOUDCOMPARE FUNCTIONS________________________________________________________________________________________________
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -NO_TIMESTAMP -C_EXPORT_FMT PLY -O ' + cc_cloud + function
    cc_function(cloud_folder, function_name, fun_txt)

    os.remove(txt_path)


def cc_recolor(cloud_path, color):
    pass


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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -O ' + \
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF -O ' + cc_cloud + function
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -O ' + cc_cloud_lim + function
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -REMOVE_ALL_SFS -O  ' + cc_mesh + function
    cc_function(cloud_dir, function_name, fun_txt)


def create_pcv(cloud_path):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'

    function_name = 'pcv'
    function = ' -AUTO_SAVE OFF -C_EXPORT_FMT PLY -PCV -SF_CONVERT_TO_RGB FALSE -SAVE_CLOUDS'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
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
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + ' -O ' + cc_cloud_lim + function
    cc_function(cloud_folder, function_name, fun_txt)


# FLOORPLANS____________________________________________________________________________________________________________
def floorplan_advanced(cloud_path, bound_pc, grid_step1, floor_level, ceiling_level, method='verticality'):
    # File names and paths
    (cloud_dir, cloudname) = os.path.split(cloud_path)
    cc_cloud = '"' + cloud_path + '"'
    cc_cloud_lim = '"' + bound_pc + '"'

    function_name = 'floor'
    function = ' -AUTO_SAVE OFF -NO_TIMESTAMP '

    if method == 'verticality':
        pass

    # projection option
    proj = ' -SF_PROJ MAX -PROJ MAX'

    # OPERATION 1: rasterize and export cloud with population count
    function += '-RASTERIZE' + proj + ' -GRID_STEP ' \
                + str(grid_step1) + ' -OUTPUT_CLOUD -SAVE_CLOUDS'

    # Prepare CloudCompare function
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -O ' + cc_cloud + function
    cc_function(cloud_dir, function_name, fun_txt)


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

    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT PLY -NO_TIMESTAMP  -O ' \
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
        fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT LAS -AUTO_SAVE OFF -O ' + cc_cloud + function_load_compare

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


# DEPRECIATED FUNCTIONS_________________________________________________________________________________________________

def preproc_z_csv(ransac_cloud_folder):  # DEPRECIATED
    for ply_file in os.listdir(ransac_cloud_folder):
        if 'merged' in ply_file:
            cloud_path = os.path.join(ransac_cloud_folder, ply_file)
    (cloud_folder, cloud_file) = os.path.split(cloud_path)
    function_name = 'dist_st'
    cc_cloud = '"' + cloud_path + '"'
    # function to create a CSV with Z coordinates of all points of a point cloud
    function = ' -COORD_TO_SF Z -SAVE_CLOUDS'

    # Prepare CloudCompare fonction
    fun_txt = 'SET MY_PATH="' + cc_path + '" \n' + '%MY_PATH% -SILENT -C_EXPORT_FMT ASC -ADD_HEADER -EXT csv -AUTO_SAVE OFF -NO_TIMESTAMP  -O ' + cc_cloud + function
    batpath = os.path.join(cloud_folder, function_name + ".bat")
    with open(batpath, 'w') as OPATH:
        OPATH.writelines(fun_txt)

    # Execute function
    subprocess.call([batpath])
    os.remove(batpath)

    # find the csv file
    csv_file = cloud_path[:-4] + '_Z_TO_SF.csv'

    return csv_file


def preproc_floors_from_csv(csv_file):  # DEPRECIATED
    # read the csv file
    data = pd.read_csv(csv_file, sep=' ')
    dist = data['Coord._Z']

    # evaluate the range of z coordinates

    # create an histogram of z coordinates
    z_hist = np.histogram(dist, bins=1000)

    # plot histogram
    _ = plt.hist(dist, bins=1000)
    plt.show()

    # compute the number of floors
    n_floors = 0  # TODO: modify to actual value

    return n_floors


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


def convert_mask_polygon(image_path, dest_poly_path, dest_crop_poly_path):
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

    # create a crop version

    # Find the bounding box of the filled contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a 10-pixel margin around the bounding box
    margin = 10
    x -= margin
    y -= margin
    w += 2 * margin
    h += 2 * margin

    # Crop the image to the bounding box with the margin
    cropped_poly = output_image[y:y + h, x:x + w]

    cv2.imwrite(dest_poly_path, filled_poly)
    cv2.imwrite(dest_crop_poly_path, cropped_poly)

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