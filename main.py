from PySide6 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageOps
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np

import os
import shutil

from pointify_engine import process
import resources as res
import widgets as wid
import test2

from ultralytics import YOLO

"""
TODO's
- Implement progress bar
- Clear viewer when changing view
- Create class for each stock pile object
- Create reporting functions
    - A giant map with all footprints

"""

# YOLO parameters
model_path = res.find('other/last.pt')
model = YOLO(model_path)  # load a custom model
threshold = 0.5
class_name_dict = {0: 'stock_pile', 1: 'vehicle', 2: 'building', 3: 'stock_with_wall', 4: 'in_construction'}

# PARAMETERS
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

class StockPileObject:
    def __init__(self):
        self.name = ''
        self.yolo_bbox = None
        self.pc = None
        self.mask = None
        self.mask_cropped = None
        self.coords = None
        self.area = 0


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

    def update_dirs(self):
        self.location_dir, self.file = os.path.split(self.path)

    def do_preprocess(self):
        self.pc_load = o3d.io.read_point_cloud(self.path)

        self.bound_pc_path = os.path.join(self.processed_data_dir, "pc_limits.ply")
        self.bound, self.bound_points, self.center, self.dim, self.density, self.n_points = process.compute_basic_properties(
            self.pc_load,
            save_bound_pc=True,
            output_path_bound_pc=self.bound_pc_path)

        print(f'The point cloud density is: {self.density:.3f}')

        if self.density < 0.05:  # if to many points --> Subsample
            sub_pc_path = os.path.join(self.location_dir,
                                       'subsampled.ply')  # path to the subsampled version of the point cloud
            sub = self.pc_load.voxel_down_sample(0.05)
            o3d.io.write_point_cloud(sub_pc_path, sub)
            self.sub_sampled = True
            self.density = 0.05

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

    def standard_images(self):
        self.res = round(self.density * 5, 3) * 1000
        process.raster_all_bound(self.path, self.res / 1000, self.bound_pc_path, xray=False)

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
        img_list = process.generate_list('.tif', self.location_dir)
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

    def planarity_images(self, orientation, span):
        if self.ransac_done:
            shutil.rmtree(self.ransac_cloud_folder)
            shutil.rmtree(self.ransac_obj_folder)
        self.do_ransac(min_factor=150)

        # create pcv version of subsampled cloud
        self.pcv_path = os.path.join(self.processed_data_dir, 'pcv.ply')
        process.create_pcv(self.sub_pc_path)
        sub_dir, _ = os.path.split(self.sub_pc_path)
        process.find_substring_new_path('PCV', self.pcv_path, sub_dir)
        # apply iso_transf
        mat, inv_mat = process.iso1_mat()
        process.cc_rotate_from_matrix(self.pcv_path, mat)
        self.transf_pcv_path = process.find_substring('pcv_TRANSFORMED', self.processed_data_dir)

        print('Launching planarity views creation...')
        # create new directory for results
        h_planes_img_dir = os.path.join(self.img_dir, 'horizontal_planes_views')
        h_planes_pc_dir = os.path.join(self.processed_data_dir, 'horizontal_planes_pc')
        process.new_dir(h_planes_img_dir)
        process.new_dir(h_planes_pc_dir)

        # create a list of detected planes
        plane_list = process.generate_list('obj', self.ransac_obj_folder, exclude='merged')

        # find horizontal planes
        hor_planes = process.find_planes(plane_list, self.ransac_cloud_folder, orientation=orientation,
                                         size_absolute='area_greater_than')
        hor_planes_loc = hor_planes[2]

        # create new_clouds from the detected planes (the original point cloud is segmented around the plane)
        n_h_elements = process.cc_planes_to_build_dist_list(self.path, hor_planes_loc, h_planes_pc_dir, span=span)

        # computing the properties for each new point cloud --> Useful to place the images on the entire point cloud
        new_pc_list = process.generate_list('.las', h_planes_pc_dir)
        # TODO: continue here

        # rendering each element
        list_h_planes_pc = process.generate_list('.las', h_planes_pc_dir)
        for cloud in list_h_planes_pc:
            process.render_planar_segment(cloud, self.res / 1000)

            # visual location
            #   rotate plane
            process.cc_rotate_from_matrix(cloud, mat)
            plane_rotated_path = process.find_substring('TRANSFORMED', h_planes_pc_dir)
            process.render_plane_in_cloud(plane_rotated_path, self.transf_pcv_path, self.res / 1000)

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
        R = process.preproc_align_cloud(self.path, self.ransac_obj_folder, self.ransac_cloud_folder)
        print(f'The point cloud has been rotated with {R} matrix...')

        transformed_path = process.find_substring('TRANSFORMED', self.location_dir)
        _, trans_file = os.path.split(transformed_path)
        new_path = os.path.join(self.processed_data_dir, trans_file)
        # move transformed file
        os.rename(transformed_path, new_path)

        self.path = new_path
        self.update_dirs()
        self.pc_load = o3d.io.read_point_cloud(self.path)
        self.bound, self.bound_points, self.center, self.dim, _, _ = process.compute_basic_properties(self.pc_load,
                                                                                                      save_bound_pc=True,
                                                                                                      output_path_bound_pc=self.bound_pc_path)
        if self.sub_sampled:
            self.sub_pc_path

    def do_ransac(self, min_factor=MIN_RANSAC_FACTOR):
        self.sub_sampled = False
        # create RANSAC directories
        self.ransac_cloud_folder = os.path.join(self.processed_data_dir, 'RANSAC_pc')
        process.new_dir(self.ransac_cloud_folder)

        self.ransac_obj_folder = os.path.join(self.processed_data_dir, 'RANSAC_meshes')
        process.new_dir(self.ransac_obj_folder)

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
            _, _, _, _, _, self.n_points_sub = process.compute_basic_properties(sub)
            print(f'The subsampled point cloud has {self.n_points_sub} points')

        # fixing RANSAC Parameters
        if not self.sub_sampled:
            points = self.n_points
        else:
            points = self.n_points_sub
        min_points = points / min_factor
        print(f'here are the minimum points {min_points}')

        self.n_planes = process.preproc_ransac_short(self.sub_pc_path, min_points, RANSAC_DIST, self.ransac_obj_folder,
                                                     self.ransac_cloud_folder)

        self.ransac_done = True


class SelectSegmentResult(QtWidgets.QDialog):
    def __init__(self, list_img, original_img_path, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'choose_with_reject'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.list_img = list_img
        print(self.list_img)
        self.n_imgs = len(self.list_img)
        self.current_img = 0

        # create custom viewer
        self.viewer_custom = wid.SimpleViewer(original_img_path, self.list_img, self)
        self.horizontalLayout_2.addWidget(self.viewer_custom)

        # connections
        self.create_connections()

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


    def create_connections(self):
        # Push buttons
        self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
        self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))

    def update_img_to_preview(self, direction):
        if direction == 'minus':
            self.current_img -= 1

        elif direction == 'plus':
            self.current_img += 1

        else:
            self.current_img = 0

        self.viewer_custom.current_img = self.current_img
        self.viewer_custom.recalculate_result()

        # change buttons
        if self.current_img == self.n_imgs - 1:
            self.pushButton_right.setEnabled(False)
        else:
            self.pushButton_right.setEnabled(True)

        if self.current_img == 0:
            self.pushButton_left.setEnabled(False)
        else:
            self.pushButton_left.setEnabled(True)


class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('What is this app about?')
        self.setFixedSize(300, 300)
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel('This app was made by Buildwise, to analyze roofs and their deformation.')
        about_text.setWordWrap(True)

        logos1 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_buildwise2.png'))
        w = self.width()
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos1.setPixmap(pixmap)

        logos2 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_pointify.png'))
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos2.setPixmap(pixmap)

        self.layout.addWidget(about_text)
        self.layout.addWidget(logos1, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(logos2, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)


class SkyStock(QtWidgets.QMainWindow):
    """
    Main Window class for the Nok-out application.
    """

    def __init__(self, parent=None):
        """
        Function to initialize the class
        :param parent:
        """
        super(SkyStock, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'volume'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        print(uifile)
        wid.loadUi(uifile, self)

        self.image_array = []
        self.image_path = ''
        self.image_loaded = False
        self.Nokclouds = []
        self.nb_roi = 0
        self.nb_seg_cloud = 0

        # add actions to action group
        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionCrop)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionSelectPoint)

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)
        self.treeView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.selmod = self.treeView.selectionModel()

        # initialize status
        self.update_progress(nb=100, text="Status: Choose point cloud!")

        # Add icons to buttons
        self.add_icon(res.find('img/cloud.png'), self.actionLoad)
        self.add_icon(res.find('img/point.png'), self.actionSelectPoint)
        self.add_icon(res.find('img/crop.png'), self.actionCrop)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        self.add_icon(res.find('img/yolo.png'), self.actionDetect)
        self.add_icon(res.find('img/magic.png'), self.actionSuperSam)
        self.add_icon(res.find('img/inventory.png'), self.actionShowInventory)

        self.add_icon(res.find('img/poly.png'), self.pushButton_show_poly)
        self.add_icon(res.find('img/square.png'), self.pushButton_show_bbox)
        self.add_icon(res.find('img/data.png'), self.pushButton_show_infos)

        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout_2.addWidget(self.viewer)

        # create connections (signals)
        self.create_connections()

    def update_progress(self, nb=None, text=''):
        self.label_status.setText(text)
        if nb is not None:
            self.progressBar.setProperty("value", nb)

            # hide progress bar when 100%
            if nb >= 100:
                self.progressBar.setVisible(False)
            elif self.progressBar.isHidden():
                self.progressBar.setVisible(True)

    def reset_parameters(self):
        """
        Reset all model parameters (image and categories)
        """

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Files')
        self.treeView.setModel(self.model)

        # reset list of stock piles

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QtGui.QIcon(img_source))

    def create_connections(self):
        """
        Link signals to slots
        """
        self.actionLoad.triggered.connect(self.get_pointcloud)
        self.actionSelectPoint.triggered.connect(self.detect_stock)
        self.actionCrop.triggered.connect(self.go_crop)
        self.actionDetect.triggered.connect(self.go_yolo)
        self.actionSuperSam.triggered.connect(self.sam_chain)
        self.actionInfo.triggered.connect(self.show_info)
        self.actionShowInventory.triggered.connect(self.inventory_canva)

        # toggle buttons
        self.pushButton_show_poly.clicked.connect(self.toggle_poly)
        self.pushButton_show_bbox.clicked.connect(self.toggle_bboxes)
        self.pushButton_show_infos.clicked.connect(self.toggle_infos)

        self.comboBox.currentIndexChanged.connect(self.on_img_combo_change)
        self.viewer.endDrawing_rect.connect(self.perform_crop)
        self.viewer.end_point_selection.connect(self.add_single_sam)

        self.selmod.selectionChanged.connect(self.on_tree_change)

    def show_info(self):
        dialog = AboutDialog()
        if dialog.exec_():
            pass

    def go_yolo(self):
        """
        Analyze the current top view of the site, and detect stock piles using YOLO algorithm
        :return:
        """
        img = self.current_cloud.view_paths[0] # the first element is the top view
        results = model(img)[0]
        count = 1

        self.stocks_inventory = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            print(x1, y1, x2, y2, score, class_id)

            if score > threshold and class_id == 0:
                print('add box to viewer!')
                # add box to the viewer
                name = f'stock pile {count}'
                # self.viewer.add_yolo_box(name, x1,y1,x2,y2)

                # create a new stock pile object
                stock_obj = StockPileObject()
                stock_obj.name = name
                stock_obj.yolo_bbox = result
                self.stocks_inventory.append(stock_obj)

                # update counter
                count +=1

        # add current inventory to viewer
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_infos(self.stocks_inventory, only_name=True)

        # enable super sam
        self.actionSuperSam.setEnabled(True)
        self.actionDetect.setEnabled(False) # TODO: allow the user to re-run YOLO

    def sam_chain(self):
        """
        Perform a series of SAM segmentation on each box detected by the YOLO algoritm
        :return:
        """
        print(r"lets get serious!")
        to_pop = []

        # take each positive yolo result, for each stock pile object, and perform a SAM segmentation in its middle
        for i, el in enumerate(self.stocks_inventory):
            x1, y1, x2, y2, score, class_id = el.yolo_bbox
            # take center of the box
            x = (x1+x2)/2
            y = (y1+y2)/2

            seg_dir = os.path.join(self.current_cloud.img_dir, 'segmentation') # create a folder to store temporary images
            process.new_dir(seg_dir)
            list_img = []

            test2.do_sam(self.current_cloud.view_paths[0], seg_dir, x, y)
            for file in os.listdir(seg_dir):
                fileloc = os.path.join(seg_dir, file)
                list_img.append(fileloc)

            dialog = SelectSegmentResult(list_img, self.current_cloud.view_paths[0])
            dialog.setWindowTitle(f"Select best output, {el.name}")

            if dialog.exec_():
                # the name is required
                text, ok = QtWidgets.QInputDialog.getText(self, 'input dialog', 'Name of the part')
                if ok:
                    el.name = text

                # the inventory element is validated and the mask is added
                print('good choice!')
                choice_im = dialog.current_img
                mask_path = list_img[choice_im]

                contour_dir = os.path.join(self.current_cloud.img_dir, 'contour')
                process.new_dir(contour_dir)
                dest_path1 = os.path.join(contour_dir, 'contour.jpg')
                dest_path2 = os.path.join(contour_dir, 'crop_contour.jpg')

                # convert SAM mask to polygon
                coords, area, _ = process.convert_mask_polygon(mask_path, dest_path1, dest_path2)

                # add infos to stock pile
                im = cv2.imread(dest_path1)
                im2 = cv2.imread(dest_path2)
                el.mask = im
                el.mask_cropped = im2
                el.coords = coords
                el.area = area*(self.current_cloud.res/1000)**2

            else:
                to_pop.append(i)

        # redraw stocks
        process.delete_elements_by_indexes(self.stocks_inventory, to_pop)

        self.viewer.clean_scene()
        self.viewer.add_list_infos(self.stocks_inventory)
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_poly(self.stocks_inventory)

        # enabled viewers buttons
        self.pushButton_show_poly.setEnabled(True)
        self.pushButton_show_bbox.setEnabled(True)
        self.pushButton_show_infos.setEnabled(True)

        self.actionShowInventory.setEnabled(True)

    def add_single_sam(self):
        # switch back to hand tool
        self.update_progress(text='Computing...')

        interest_point = self.viewer.get_selected_point()
        x = interest_point.x()
        y = interest_point.y()

        seg_dir = os.path.join(self.current_cloud.img_dir, 'segmentation')
        process.new_dir(seg_dir)
        list_img = []

        print(f'input: {self.current_cloud.view_paths[0]}, {x}, {y}')
        test2.do_sam(self.current_cloud.view_paths[0], seg_dir, x, y)

        for file in os.listdir(seg_dir):
            fileloc = os.path.join(seg_dir, file)
            list_img.append(fileloc)

        dialog = SelectSegmentResult(list_img, self.current_cloud.view_paths[0])
        dialog.setWindowTitle("Select best output")

        if dialog.exec_():
            text, ok = QtWidgets.QInputDialog.getText(self, 'input dialog', 'Name of the part')
            if ok:
                name = text
            else:
                name = f'Stock pile {len(self.stocks_inventory)}'

            print('good choice!')
            choice_im = dialog.current_img
            mask_path = list_img[choice_im]

            contour_dir = os.path.join(self.current_cloud.img_dir, 'contour')
            process.new_dir(contour_dir)
            dest_path1 = os.path.join(contour_dir, 'contour.jpg')
            dest_path2 = os.path.join(contour_dir, 'crop_contour.jpg')

            # convert SAM mask to polygon
            coords, area, yolo_type_bbox = process.convert_mask_polygon(mask_path, dest_path1, dest_path2)

            # add infos to stock pile
            im = cv2.imread(dest_path1)
            im2 = cv2.imread(dest_path2)

            # add object
            stock_obj = StockPileObject()
            stock_obj.name = name
            stock_obj.yolo_bbox = yolo_type_bbox
            self.stocks_inventory.append(stock_obj)

            stock_obj.mask = im
            stock_obj.mask_cropped = im2
            stock_obj.coords = coords
            stock_obj.area = area * (self.current_cloud.res / 1000) ** 2

        self.hand_pan()

        self.viewer.clean_scene()

        self.viewer.add_list_infos(self.stocks_inventory)
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_poly(self.stocks_inventory)

        # enabled viewers buttons
        self.pushButton_show_poly.setChecked(True)
        self.pushButton_show_bbox.setChecked(True)
        self.pushButton_show_infos.setChecked(True)


    def toggle_infos(self):
        if not self.pushButton_show_infos.isChecked():
            self.viewer.clean_scene_text()
        else:
            self.viewer.add_list_infos(self.stocks_inventory)

    def toggle_bboxes(self):
        if not self.pushButton_show_bbox.isChecked():
            self.viewer.clean_scene_rectangle()
        else:
            self.viewer.add_list_boxes(self.stocks_inventory)

    def toggle_poly(self):
        if not self.pushButton_show_poly.isChecked():
            self.viewer.clean_scene_poly()
        else:
            self.viewer.add_list_poly(self.stocks_inventory)

    def detect_stock(self, stuff_class):
        # Here the SAM model is called
        if self.actionSelectPoint.isChecked():
            self.viewer.select_point = True
            self.viewer.toggleDragMode()

            self.update_progress(text='Click on a stock!')

    def sam_process_old(self):
        """
        SAM segmentation based on user point selection
        :return:
        """
        # switch back to hand tool
        self.update_progress(text='Computing...')

        interest_point = self.viewer.get_selected_point()
        x = interest_point.x()
        y = interest_point.y()

        seg_dir = os.path.join(self.current_cloud.img_dir, 'segmentation')
        process.new_dir(seg_dir)
        list_img = []

        print(f'input: {self.current_cloud.view_paths[0]}, {x}, {y}')
        test2.do_sam(self.current_cloud.view_paths[0], seg_dir, x, y)

        for file in os.listdir(seg_dir):
            fileloc = os.path.join(seg_dir, file)
            list_img.append(fileloc)

        dialog = SelectSegmentResult(list_img, self.current_cloud.view_paths[0])
        dialog.setWindowTitle("Select best output")

        if dialog.exec_():
            print('good choice!')
            choice_im = dialog.current_img
            image_path = list_img[choice_im]

            contour_dir = os.path.join(self.current_cloud.img_dir, 'contour')
            process.new_dir(contour_dir)

            dest_path1 = os.path.join(contour_dir, 'contour1.jpg')
            dest_path2 = os.path.join(contour_dir, 'contour2.jpg')

            # convert SAM mask to polygon
            _, coords = process.convert_mask_polygon(image_path, dest_path1, dest_path2)
            self.nb_seg_cloud += 1

            text, ok = QtWidgets.QInputDialog.getText(self, 'input dialog', 'Name of the part')
            if ok:
                part_name = text
                sam_cloud_path = os.path.join(self.current_cloud.processed_data_dir, part_name + '.ply')
                sam_cloud_path_ref = os.path.join(self.current_cloud.processed_data_dir, part_name + '_large.ply')

                print('lets crop this')
                # convert image coords to point cloud coords
                new_coords = process.convert_coord_img_to_cloud_topview(coords, self.current_cloud.res,
                                                                        self.current_cloud.center,
                                                                        self.current_cloud.dim)
                process.crop_coords(self.current_cloud.path, new_coords)

                src = process.find_substring('CROPPED', self.current_cloud.location_dir)
                os.rename(src, sam_cloud_path)

                # crop and keep the outside
                process.crop_coords(self.current_cloud.path, new_coords, outside=True)
                src = process.find_substring('CROPPED_RASTER', self.current_cloud.location_dir)
                os.rename(src, sam_cloud_path_ref)

                to_delete = process.find_substring('CROPPED', self.current_cloud.location_dir)
                os.remove(to_delete)

                """
                process.crop_coords(self.current_cloud.path, new_coords2)
                src = process.find_substring('CROPPED', self.current_cloud.location_dir)
                os.rename(src, sam_cloud_path_ref)
                """

                # computing the volume
                process.compute_volume_clouds(sam_cloud_path, sam_cloud_path_ref)

                # get volume
                volume_text_file = process.find_substring('VolumeCalculationReport',
                                                          self.current_cloud.processed_data_dir)
                with open(volume_text_file) as f:
                    volume_result = f.readline()

                # create new point cloud object
                self.create_point_cloud_object(sam_cloud_path, part_name, orient=False, ransac=False, mesh=True)
                self.current_cloud = self.Nokclouds[-1]

                # launch custom viewer
                app_vis = gui.Application.instance
                app_vis.initialize()
                print(self.current_cloud.pc_load)
                print(self.current_cloud.mesh_load)

                viz = wid.Custom3dView(self.current_cloud.pc_load, self.current_cloud.mesh_load, volume_result)
                app_vis.run()

                # segm_load = o3d.io.read_point_cloud(sam_cloud_path)
                # process.basic_vis_creation(segm_load, 'top')

        self.hand_pan()


    def inventory_canva(self):
        image_list = []
        name_list = []
        for el in self.stocks_inventory:
            image_list.append(el.mask_cropped)
            name_list.append(el.name)

        inventory_dir = os.path.join(self.current_cloud.img_dir, 'inventory')
        process.new_dir(inventory_dir)
        dest_path = os.path.join(inventory_dir, 'bw_inventory.jpg')

        process.generate_summary_canva(image_list, name_list, dest_path)

    def go_crop(self):
        if self.actionCrop.isChecked():
            self.viewer.rect = True
            self.viewer.toggleDragMode()

            self.update_progress(text='Draw a box to crop!')

    def perform_crop(self):
        self.update_progress(text='Computing...')

        # get coordinates and crop cloud
        coords = self.viewer.crop_coords
        start_x = int(coords[0].x()) * self.current_cloud.res / 1000
        start_y = int(coords[0].y()) * self.current_cloud.res / 1000
        end_x = int(coords[1].x()) * self.current_cloud.res / 1000
        end_y = int(coords[1].y()) * self.current_cloud.res / 1000

        # crop the point cloud
        bound = self.current_cloud.pc_load.get_axis_aligned_bounding_box()
        center = bound.get_center()
        dim = bound.get_extent()

        if self.current_view == 'top':
            pt1 = [center[0] - dim[0] / 2 + start_x, center[1] + dim[1] / 2 - start_y, center[2] - dim[2] / 2]
            pt2 = [pt1[0] + (end_x - start_x), pt1[1] - (end_y - start_y), center[2] + dim[2] / 2]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "top"

        elif self.current_view == 'front':
            pt1 = [center[0] - dim[0] / 2 + start_x, center[1] - dim[1] / 2, center[2] + dim[2] / 2 - start_y]
            pt2 = [pt1[0] + (end_x - start_x), center[1] + dim[1] / 2, pt1[2] - (end_y - start_y)]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "front"

        elif self.current_view == 'right':
            pt1 = [center[0] - dim[0] / 2, center[1] - dim[1] / 2 + start_x, center[2] + dim[2] / 2 - start_y]
            pt2 = [center[0] + dim[0] / 2, pt1[1] + (end_x - start_x), pt1[2] - (end_y - start_y)]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "front"

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)
        point_cloud_crop = self.current_cloud.pc_load.crop(crop_box)
        self.nb_roi += 1
        roi_path = os.path.join(self.current_cloud.processed_data_dir, f"roi{self.nb_roi}.ply")
        o3d.io.write_point_cloud(roi_path, point_cloud_crop)

        # create new point cloud
        self.create_point_cloud_object(roi_path, f'roi{self.nb_roi}', orient=False, ransac=False)
        process.basic_vis_creation(point_cloud_crop, orientation)

        self.viewer.clean_scene()

        # switch back to hand tool
        self.hand_pan()

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)
        self.viewer.rect = False
        self.viewer.select_point = False
        self.viewer.toggleDragMode()

        self.update_progress(text='Yon can pan the image!')

    def get_pointcloud(self):
        """
        Get the point cloud path from the user
        :return:
        """
        try:
            pc = QtWidgets.QFileDialog.getOpenFileName(self, u"Ouverture de fichiers", "", "Point clouds (*.ply *.las)")
            print(f'the following point cloud will be loaded {pc[0]}')
        except:
            pass
        if pc[0] != '':
            # load and show new image
            self.load_main_pointcloud(pc[0])

    def load_main_pointcloud(self, path):
        """
        Load the new point cloud and reset the model
        :param path:
        :return:
        """
        original_dir, _ = os.path.split(path)

        # create specific folder for the app outputs
        self.app_dir = os.path.join(original_dir, 'SkyStock')
        process.new_dir(self.app_dir)

        self.create_point_cloud_object(path, 'Original_point_cloud')

        self.update_progress(text='Choose a functionality!')

    def create_point_cloud_object(self, path, name, orient=False, ransac=False, mesh=False):
        cloud = NokPointCloud()
        self.Nokclouds.append(cloud)  # note: self.Nokclouds[0] is always the original point cloud

        self.Nokclouds[-1].path = path
        self.Nokclouds[-1].update_dirs()
        self.Nokclouds[-1].name = name

        self.Nokclouds[-1].folder = os.path.join(self.app_dir, self.Nokclouds[-1].name)
        self.Nokclouds[-1].processed_data_dir = os.path.join(self.Nokclouds[-1].folder, 'processes')
        self.Nokclouds[-1].img_dir = os.path.join(self.Nokclouds[-1].folder, 'images')

        process.new_dir(self.Nokclouds[-1].folder)
        process.new_dir(self.Nokclouds[-1].processed_data_dir)
        process.new_dir(self.Nokclouds[-1].img_dir)
        self.process_pointcloud(self.Nokclouds[-1], orient=orient, ransac=ransac, mesh=mesh)

        # add element to treeview
        self.current_cloud = self.Nokclouds[-1]
        self.add_item_in_tree(self.model, self.Nokclouds[-1].name)  # signal a tree change

        # load image
        self.comboBox.setEnabled(True)
        self.image_loaded = True

        self.comboBox.clear()
        self.comboBox.addItems(self.current_cloud.view_names)
        self.on_img_combo_change()

        nb_pc = len(self.Nokclouds)
        build_idx = self.model.index(nb_pc - 1, 0)
        self.selmod.clearSelection()

        self.selmod.setCurrentIndex(build_idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        self.treeView.expandAll()

        # enable action(s)
        self.actionCrop.setEnabled(True)
        self.actionDetect.setEnabled(True)
        self.actionSelectPoint.setEnabled(True)
        self.actionHand_selector.setEnabled(True)

    def on_tree_change(self):
        print('CHANGED!')
        indexes = self.treeView.selectedIndexes()
        sel_item = self.model.itemFromIndex(indexes[0])
        print(indexes[0])

        for cloud in self.Nokclouds:
            if cloud.name == sel_item.text():
                self.current_cloud = cloud
                print('Current cloud name: ', self.current_cloud.name)

                self.comboBox.clear()
                self.comboBox.addItems(self.current_cloud.view_names)

    def process_pointcloud(self, pc, orient=False, ransac=False, mesh=False):
        # 1. BASIC DATA ____________________________________________________________________________________________________
        # read full high definition point cloud (using open3d)
        print('Reading the point cloud!')
        pc.do_preprocess()

        # 2. RANSAC DETECTION __________________________________________________________________________________
        if ransac:
            print('Launching RANSAC detection...')
            pc.do_ransac()

        # 3. ORIENT THE POINT CLOUD PERPENDICULAR TO AXIS_______________________________________________________
        if orient:
            print('Orienting the point cloud perpendicular to the axes...')
            pc.do_orient()

        # 4. GENERATE MESH
        if mesh:
            pc.do_mesh()

        # 4. GENERATE BASIC VIEWS_______________________________________________________
        print('Launching RGB render/exterior views creation...')
        pc.standard_images()

    def on_img_combo_change(self):
        self.actionCrop.setEnabled(True)
        i = self.comboBox.currentIndex()
        if i < 0:
            i = 0
        self.current_view = self.current_cloud.view_names[i]
        print(i)
        print(self.current_view)
        if self.current_view == 'back' or self.current_view == 'left':
            self.actionCrop.setEnabled(False)

        img_paths = self.current_cloud.view_paths
        if self.image_loaded:
            self.viewer.setPhoto(QtGui.QPixmap(img_paths[i]))

    def add_item_in_tree(self, parent, line):
        item = QtGui.QStandardItem(line)
        parent.appendRow(item)
        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Files')


def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None

    # create the application if necessary
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication(argv)
        app.setStyle('Fusion')

    # create the main window

    window = SkyStock()
    window.showMaximized()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
