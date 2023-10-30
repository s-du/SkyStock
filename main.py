from PySide6 import QtWidgets, QtGui, QtCore
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import os
from pathlib import Path
from shutil import copyfile, copytree
from pointify_engine import process
import widgets as wid
import resources as res
import test2
from ultralytics import YOLO
import dialogs as dia

"""
TODO's
- Solve 'chromatic aberration problem' --> Comes when a Odm cloud is saved in open3D
- Change data viz in center of piles
- Create stockpiles objects and show them in the treeview
- Implement progress bar
- Clear viewer when changing view
- Create class for each stock pile object
- Create reporting functions
    - A giant map with all footprints

"""
# ODM

# SAM
USE_FASTSAM = False

# YOLO parameters
model_path = res.find('other/last.pt')
model = YOLO(model_path)  # load a custom model
threshold = 0.5
class_name_dict = {0: 'stock_pile', 1: 'vehicle', 2: 'building', 3: 'stock_with_wall', 4: 'in_construction'}





class SkyStock(QtWidgets.QMainWindow):
    """
    Main Window class for the Nok-out application.
    """

    def __init__(self, is_dark_theme, parent=None):
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

        self.stocks_inventory = []

        # add actions to action group
        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionCrop)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionSelectPoint)
        ag.addAction(self.actionLineMeas)

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)
        self.treeView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.selmod = self.treeView.selectionModel()

        # initialize status
        self.update_progress(nb=100, text="Status: Choose point cloud!")

        if is_dark_theme:
            suf = '_white'
        else:
            suf = ''

        # Add icons to buttons
        self.add_icon(res.find(f'img/icons/cloud{suf}.png'), self.actionLoad)
        self.add_icon(res.find(f'img/icons/folder{suf}.png'), self.actionPhotogr)
        self.add_icon(res.find(f'img/icons/point{suf}.png'), self.actionSelectPoint)
        self.add_icon(res.find(f'img/icons/crop{suf}.png'), self.actionCrop)
        self.add_icon(res.find(f'img/icons/hand{suf}.png'), self.actionHand_selector)
        self.add_icon(res.find(f'img/icons/yolo2{suf}.png'), self.actionDetect)
        self.add_icon(res.find(f'img/icons/magic{suf}.png'), self.actionSuperSam)
        self.add_icon(res.find(f'img/icons/inventory{suf}.png'), self.actionShowInventory)
        self.add_icon(res.find(f'img/icons/profile{suf}.png'), self.actionLineMeas)
        self.add_icon(res.find(f'img/icons/alti{suf}.png'), self.actionCropHeight)
        self.add_icon(res.find(f'img/icons/poly{suf}.png'), self.actionPolySeg)

        self.add_icon(res.find(f'img/icons/poly{suf}.png'), self.pushButton_show_poly)
        self.add_icon(res.find(f'img/icons/square{suf}.png'), self.pushButton_show_bbox)
        self.add_icon(res.find(f'img/icons/data{suf}.png'), self.pushButton_show_infos)

        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout_2.addWidget(self.viewer)

        # add dual viewer
        self.dual_viewer = wid.DualViewer()
        self.verticalLayout_4.addWidget(self.dual_viewer)

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

        # reset list of stockpiles



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
        self.actionPhotogr.triggered.connect(self.odm_start)
        self.actionSelectPoint.triggered.connect(self.detect_stock)
        self.actionCrop.triggered.connect(self.go_crop)
        self.actionDetect.triggered.connect(self.go_yolo)
        self.actionSuperSam.triggered.connect(self.sam_chain)
        self.actionInfo.triggered.connect(self.show_info)
        self.actionShowInventory.triggered.connect(self.inventory_canva)
        self.actionLineMeas.triggered.connect(self.line_meas)
        self.actionCropHeight.triggered.connect(self.change_altitude_limits)
        self.actionPolySeg.triggered.connect(self.add_poly_seg)

        # toggle buttons
        self.pushButton_show_poly.clicked.connect(self.toggle_poly)
        self.pushButton_show_bbox.clicked.connect(self.toggle_bboxes)
        self.pushButton_show_infos.clicked.connect(self.toggle_infos)

        self.comboBox.currentIndexChanged.connect(self.on_img_combo_change)
        self.viewer.endDrawing_rect.connect(self.perform_crop)
        self.viewer.end_point_selection.connect(self.add_single_sam)
        self.viewer.endDrawing_line_meas.connect(self.get_ground_profile)

        self.selmod.selectionChanged.connect(self.on_tree_change)

    # ODM Related_______________________________________________
    def odm_start(self):
        # get output directory (for all files)
        out_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select output_folder"))
        while not os.path.isdir(out_dir):
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "Oops! Not a folder!")
            out_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select output_folder"))

        # create specific folder for the app outputs
        self.app_dir = os.path.join(out_dir, 'SkyStock')
        process.new_dir(self.app_dir)

        # launch odm options dialog
        dialog = dia.Pyodm(self.app_dir, process.ODM_DEFAULT_PC_QUALITY_LIST)
        if dialog.exec_():
            if len(dialog.img_list) <= 1:
                print('not enough image added')
            else:
                self.odm_img_list = dialog.img_list
                try:
                    self.odm_prepare_images()
                    idx_1 = dialog.comboBox_features.currentIndex()
                    idx_2 = dialog.comboBox_quality.currentIndex()

                    self.odm_go(idx_1, idx_2)
                except Exception as e:
                    # Print the exception to the console
                    print(f"Error: {e}")

                    # Show a warning message using PySide6
                    QtWidgets.QMessageBox.warning(None, "Warning", "Something went wrong with ODM!")

    def odm_prepare_images(self):
        # read ops details
        images = self.odm_img_list
        subfolder_name = 'code'

        # create subfolders to store results
        out_subdir = os.path.join(self.app_dir, subfolder_name)
        if not os.path.exists(out_subdir):
            os.mkdir(out_subdir)

        img_dir = os.path.join(out_subdir, 'images')
        self.results_dir = os.path.join(out_subdir, 'outputs')

        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        list_copied_images = []
        for img in images:
            _, img_file = os.path.split(img)
            dest_file = os.path.join(img_dir, img_file)
            if not os.path.exists(dest_file):
                copyfile(img, dest_file)
                list_copied_images.append(dest_file)

        self.odm_target_path = Path(self.app_dir)
        print('image prep done')

    def odm_go(self, idx_1, idx_2):
        # TODO add check if already computed

        feat_qual = process.ODM_DEFAULT_PC_QUALITY_LIST[idx_1]
        pc_qual = process.ODM_DEFAULT_PC_QUALITY_LIST[idx_2]

        # initialize status
        self.update_progress(nb=5, text="Preparing ODM")

        # launch photogrammetric reconstruction
        self.odm_thread = process.ODMThread(self.odm_target_path, feat_qual, pc_qual)
        self.odm_thread.outputSignal.connect(self.odm_update_feedback)
        self.odm_thread.finished.connect(self.odm_post_process)
        self.odm_thread.start()

    def odm_update_feedback(self, str):
        if 'Initializing ODM 3.2.1' in str:
            self.update_progress(nb=10, text='Launching ODM...')
        elif 'Running opensfm stage' in str:
            self.update_progress(nb=15, text='Launching SFM stage...')
        elif 'detect_features' in str:
            self.update_progress(nb=20, text='Detecting features...')
        elif 'match_features' in str:
            self.update_progress(nb=25, text='Matching features...')
        elif 'create_tracks' in str:
            self.update_progress(nb=30, text='Create tracks...')
        elif 'Running openmvs stage' in str:
            self.update_progress(nb=40, text='Launching MVS stage...')
        elif 'Running odm_filterpoints stage' in str:
            self.update_progress(nb=75, text='Filtering cloud...')
        elif 'ODM app finished' in str:
            self.update_progress(nb=100, text='Reconstruction finished')

    def odm_post_process(self):
        print('ok success!')
        path = os.path.join(self.app_dir, 'code', 'odm_filterpoints', 'point_cloud.ply')
        self.create_point_cloud_object(path, 'Original_point_cloud')

        self.update_progress(text='Choose a functionality!')
        self.actionLoad.setEnabled(False)

    def show_info(self):
        dialog = dia.AboutDialog()
        if dialog.exec_():
            pass

    def line_meas(self):
        if self.actionLineMeas.isChecked():
            # activate drawing tool
            self.viewer.line_meas = True
            self.viewer.toggleDragMode()

    def change_altitude_limits(self):
        dialog = dia.MySliderDemo(self.current_cloud.height_data)

        if dialog.exec_():
            # change the low and high point in the point cloud object --> It should have impact on the cropping
            self.current_cloud.low_point = dialog.slider_low.value()
            self.current_cloud.high_point = dialog.slider_high.value()

    def get_ground_profile(self):
        values = self.viewer.line_values_final
        x = np.linspace(0, len(values) * self.current_cloud.res / 1000, num=len(values))
        plt.plot(x, values, color=(1, 0, 1))
        plt.axis('equal')
        plt.ylabel('Height [m]')
        plt.xlabel('Distance [m]')
        plt.show()

        self.hand_pan()

    def add_poly_seg(self):
        """
        Allows the user to click on consecutive points to draw a polygon. When pressing 'enter' the polygon is closed
        :return:
        """
        # activate the tool on the viewer

    # AI TOOLS ----------------------------------------------------------
    def go_yolo(self):
        """
        Analyze the current top view of the site, and detect stockpiles bboxes using YOLO algorithm
        :return:
        """
        img = self.current_cloud.view_paths[0]  # the first element is the top view
        results = model(img)[0]
        count = 1

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            print(x1, y1, x2, y2, score, class_id)

            if score > threshold and class_id == 0:
                print('add box to viewer!')
                # add box to the viewer
                name = f'stock pile {count}'
                # self.viewer.add_yolo_box(name, x1,y1,x2,y2)

                # create a new stock pile object
                stock_obj = process.StockPileObject()
                stock_obj.name = name
                stock_obj.yolo_bbox = result
                self.stocks_inventory.append(stock_obj)

                # update counter
                count += 1

        # add current inventory to viewer
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_infos(self.stocks_inventory, only_name=True)

        # add objects to the triewview

        # enable super sam
        self.actionSuperSam.setEnabled(True)
        self.actionDetect.setEnabled(False)  # TODO: allow the user to re-run YOLO

    def sam_process(self, seg_dir, x, y, stock_object=None):
        list_img = []
        if not USE_FASTSAM:
            test2.do_sam(self.current_cloud.view_paths[0], seg_dir, x, y)

        # TODO Add FASTSAM option

        # list all generated mask images
        for file in os.listdir(seg_dir):
            fileloc = os.path.join(seg_dir, file)
            list_img.append(fileloc)

        # open the dialog for selecting the best segmentation (based on three top scores from SAM)
        dialog = dia.SelectSegmentResult(list_img, self.current_cloud.view_paths[0])
        dialog.setWindowTitle("Select best output")

        to_pop = False

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
            dest_path3 = os.path.join(contour_dir, 'contour_rgb.jpg')
            dest_path4 = os.path.join(contour_dir, 'crop_contour_rgb.jpg')

            # convert SAM mask to polygon
            top_view = cv2.imread(self.current_cloud.view_paths[0])
            elevation = self.current_cloud.height_data

            coords, area, yolo_type_bbox = process.convert_mask_polygon(mask_path, top_view,
                                                                        dest_path1,
                                                                        dest_path2,
                                                                        dest_path3,
                                                                        dest_path4)
            # add infos to stock pile
            im = cv2.imread(dest_path1)
            im2 = cv2.imread(dest_path2)
            im3 = cv2.imread(dest_path3)
            im4 = cv2.imread(dest_path4)

            # add stockpile class object
            if stock_object == None:
                stock_obj = process.StockPileObject()
                self.stocks_inventory.append(stock_obj)
                stock_obj.yolo_bbox = yolo_type_bbox
            else:
                stock_obj = stock_object
                stock_obj.to_check = False

            stock_obj.name = name

            stock_obj.mask = im
            stock_obj.mask_cropped = im2
            stock_obj.mask_rgb = im3
            stock_obj.mask_rgb_cropped = im4
            stock_obj.coords = coords
            stock_obj.area = area * (self.current_cloud.res / 1000) ** 2

            # compute volume
            process.create_ground_map(self.current_cloud.ground_data, stock_obj.mask)

            # compute volume of stock pile
            stock_obj.volume = process.compute_volume(self.current_cloud.height_data,
                                                      self.current_cloud.ground_data,
                                                      stock_obj.mask, self.current_cloud.res / 1000)

            # create new point cloud object
            stock_pc, color_array = process.create_pc_from_elevation_coords(elevation,
                                                                            self.current_cloud.view_paths[0],
                                                                            coords,
                                                                            self.current_cloud.res / 1000)
            pcd = o3d.geometry.PointCloud()
            color_array_normalized = color_array / 255.0
            pcd.points = o3d.utility.Vector3dVector(stock_pc)
            pcd.colors = o3d.utility.Vector3dVector(color_array_normalized)

            sam_cloud_path = os.path.join(self.current_cloud.processed_data_dir,
                                          name + '.ply')  # TODO Ensure unique names
            o3d.io.write_point_cloud(sam_cloud_path, pcd)

            self.create_point_cloud_object(sam_cloud_path, name, selection=False, gsd=False, orient=False,
                                           ransac=False, mesh=False)
            self.current_cloud = self.Nokclouds[-1]

            # o3d.visualization.draw_geometries([pcd])

            """
            # launch custom viewer
            app_vis = gui.Application.instance
            app_vis.initialize()
            print(self.current_cloud.pc_load)
            print(self.current_cloud.mesh_load)

            viz = wid.Custom3dView(self.current_cloud.pc_load, self.current_cloud.mesh_load, str(stock_obj.volume))
            app_vis.run()
            """

        else:
            to_pop = True

        return stock_obj, to_pop

    def sam_chain(self):
        """
        Perform a series of SAM segmentation on each box detected by the YOLO algoritm
        :return:
        """
        print(r"lets get serious!")
        to_pop = []

        # take each positive yolo result, for each stockpile object, and perform a SAM segmentation in its middle
        for i, el in enumerate(self.stocks_inventory):
            if el.to_check:
                x1, y1, x2, y2, score, class_id = el.yolo_bbox
                # take center of the box
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2

                seg_dir = os.path.join(self.current_cloud.img_dir,
                                       'segmentation')  # create a folder to store temporary images
                process.new_dir(seg_dir)

                # launch specific SAM method
                pop = self.sam_process(seg_dir, x, y, stock_object=el)
                if pop:
                    to_pop.append(i)

        # redraw stocks
        process.delete_elements_by_indexes(self.stocks_inventory, to_pop)

        # add data to viewer
        self.viewer.clean_scene()
        self.viewer.add_list_infos(self.stocks_inventory)
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_poly(self.stocks_inventory)

        # update treeview

        # enabled viewers buttons
        self.pushButton_show_poly.setEnabled(True)
        self.pushButton_show_bbox.setEnabled(True)
        self.pushButton_show_infos.setEnabled(True)

        self.pushButton_show_poly.setChecked(True)
        self.pushButton_show_bbox.setChecked(True)
        self.pushButton_show_infos.setChecked(True)

        self.actionShowInventory.setEnabled(True)

    def add_single_sam(self):
        # switch back to hand tool
        # TODO Add a visual feedback for the point
        self.update_progress(text='Computing...')

        interest_point = self.viewer.get_selected_point()
        x = interest_point.x()
        y = interest_point.y()

        seg_dir = os.path.join(self.current_cloud.img_dir, 'segmentation')
        process.new_dir(seg_dir)

        # launch specific SAM method
        self.sam_process(seg_dir, x, y)

        # draw all elements on 2D view
        self.viewer.clean_scene()

        self.viewer.add_list_infos(self.stocks_inventory)
        self.viewer.add_list_boxes(self.stocks_inventory)
        self.viewer.add_list_poly(self.stocks_inventory)

        self.hand_pan()
        # enabled viewers buttons
        self.pushButton_show_poly.setEnabled(True)
        self.pushButton_show_bbox.setEnabled(True)
        self.pushButton_show_infos.setEnabled(True)

        self.pushButton_show_poly.setChecked(True)
        self.pushButton_show_bbox.setChecked(True)
        self.pushButton_show_infos.setChecked(True)

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

        dialog = dia.SelectSegmentResult(list_img, self.current_cloud.view_paths[0])
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

    # VIEWER RELATED TOOLS ----------------------------------------------------------

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

        pt1 = [center[0] - dim[0] / 2 + start_x, center[1] + dim[1] / 2 - start_y, center[2] - dim[2] / 2]
        pt2 = [pt1[0] + (end_x - start_x), pt1[1] - (end_y - start_y), center[2] + dim[2] / 2]
        np_points = [pt1, pt2]
        points = o3d.utility.Vector3dVector(np_points)
        orientation = "top"

        """
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
        """

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)
        point_cloud_crop = self.current_cloud.pc_load.crop(crop_box)
        self.nb_roi += 1
        roi_path = os.path.join(self.current_cloud.processed_data_dir, f"roi{self.nb_roi}.ply")
        o3d.io.write_point_cloud(roi_path, point_cloud_crop)

        # create new point cloud
        self.create_point_cloud_object(roi_path, f'crop_{self.nb_roi}', orient=False, ransac=False, keep_previous=False)
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

    def create_point_cloud_object(self, path, name, selection=True, gsd=True, orient=False, ransac=False, mesh=False,
                                  keep_previous=True):
        cloud = process.NokPointCloud()
        if not keep_previous:
            self.Nokclouds = []

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

        # generate all basic data
        self.process_pointcloud(self.Nokclouds[-1], selection=selection, gsd=gsd, orient=orient, ransac=ransac,
                                mesh=mesh)

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

        # store pc height data
        self.height_values = self.current_cloud.height_data
        self.viewer.set_height_data(self.height_values)

        # add dual viewer images
        img1 = self.current_cloud.view_paths[0]
        img2 = self.current_cloud.view_paths[1]
        self.dual_viewer.load_images_from_path(img1, img2)

        # enable action(s)
        self.actionCrop.setEnabled(True)
        self.actionLineMeas.setEnabled(True)
        self.actionDetect.setEnabled(True)
        self.actionSelectPoint.setEnabled(True)
        self.actionHand_selector.setEnabled(True)
        self.actionLineMeas.setEnabled(True)

    def process_pointcloud(self, pc, gsd=True, selection=True, orient=False, ransac=False, mesh=False):
        # 1. BASIC DATA ____________________________________________________________________________________________________
        # read full high definition point cloud (using open3d)
        print('Reading the point cloud!')
        pc.do_preprocess()

        # let user choose GSD
        img_1 = res.find('img/2cm.png')
        img_2 = res.find('img/5cm.png')
        img_3 = res.find('img/10cm.png')
        img_4 = res.find('img/20cm.png')
        list_img = [img_1, img_2, img_3, img_4]

        density = pc.density
        if gsd:
            dialog = dia.SelectGsd(list_img, density)
            dialog.setWindowTitle("Select best output")

            if dialog.exec_():
                print('gsd chosen!')
                pc.res = dialog.value

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
        if selection:
            pc.image_selection()
        else:
            pc.standard_images()

    def on_tree_change(self):
        print('CHANGED!')
        indexes = self.treeView.selectedIndexes()

        if indexes:  # Check if the list is not empty
            sel_item = self.model.itemFromIndex(indexes[0])
            print(indexes[0])

            for cloud in self.Nokclouds:
                if cloud.name == sel_item.text():
                    self.current_cloud = cloud
                    print('Current cloud name: ', self.current_cloud.name)

                    self.comboBox.clear()
                    self.comboBox.addItems(self.current_cloud.view_names)
        else:
            print("No items selected.")

    def on_img_combo_change(self):
        self.actionCrop.setEnabled(True)
        i = self.comboBox.currentIndex()
        if i < 0:
            i = 0
        self.current_view = self.current_cloud.view_names[i]
        print(i)
        print(self.current_view)
        if self.current_view != 'top':
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

        # test if dark theme is used
        palette = app.palette()
        bg_color = palette.color(QtGui.QPalette.Window)

        is_dark_theme = bg_color.lightness() < 128
        print(is_dark_theme)

    # create the main window

    window = SkyStock(is_dark_theme)
    window.showMaximized()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
