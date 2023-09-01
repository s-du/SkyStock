import os
from PySide6 import QtWidgets, QtGui, QtCore
import resources as res
import widgets as wid
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, data=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MyCanvas, self).__init__(fig)

        self.data = data
        self.colormap = 'viridis'
        self.contour_mode = False
        self.set_data()

    def set_data(self):
        if self.contour_mode:
            self.axes.contour(self.data, cmap=self.colormap)
        else:
            self.axes.imshow(self.data, cmap=self.colormap)
        self.draw()

    def update_plot(self, low, high, colormap, contour_mode):
        self.axes.cla()
        if contour_mode:
            self.axes.contour(self.data, cmap=colormap)
            self.axes.set_ylim(self.axes.get_ylim()[::-1])
        else:
            self.axes.imshow(self.data, cmap=colormap, vmin=low, vmax=high)
        self.draw()

class MySliderDemo(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super(MySliderDemo, self).__init__(parent)

        self.data = data
        min_data, max_data = np.nanmin(data), np.nanmax(data)

        self.canvas = MyCanvas(data=self.data)

        layout = QtWidgets.QVBoxLayout()

        self.slider_low = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_low.setMinimum(int(min_data))
        self.slider_low.setMaximum(int(max_data))
        self.slider_low.setValue(int(min_data))

        self.slider_high = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_high.setMinimum(int(min_data))
        self.slider_high.setMaximum(int(max_data))
        self.slider_high.setValue(int(max_data))

        self.combo_colormap = QtWidgets.QComboBox()
        self.combo_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

        self.checkbox_contour = QtWidgets.QCheckBox('Contour Mode')
        self.checkbox_contour.setChecked(False)

        layout.addWidget(QtWidgets.QLabel(f'Low Limit ({int(min_data)} to {int(max_data)}):'))
        layout.addWidget(self.slider_low)
        layout.addWidget(QtWidgets.QLabel(f'High Limit ({int(min_data)} to {int(max_data)}):'))
        layout.addWidget(self.slider_high)
        layout.addWidget(QtWidgets.QLabel('Colormap:'))
        layout.addWidget(self.combo_colormap)
        layout.addWidget(self.checkbox_contour)
        layout.addWidget(self.canvas)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.slider_low.valueChanged.connect(self.update_plot)
        self.slider_high.valueChanged.connect(self.update_plot)
        self.combo_colormap.currentTextChanged.connect(self.update_plot)
        self.checkbox_contour.stateChanged.connect(self.update_plot)

        self.setLayout(layout)

    def update_plot(self):
        low = self.slider_low.value()
        high = self.slider_high.value()
        colormap = self.combo_colormap.currentText()
        contour_mode = self.checkbox_contour.isChecked()
        self.canvas.update_plot(low, high, colormap, contour_mode)



class SelectGsd(QtWidgets.QDialog):
    def __init__(self, list_img, density, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'gsd'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.list_img = list_img
        self.value = 0.02
        self.label_density.setText(f"<b>Point spacing in your point cloud is typically: {density*1000} mm</b>")

        # set adviced value
        new_value = density * 4 * 1000

        # add custom slider
        self.horizontalSlider = wid.CustomTickSlider(new_value, QtCore.Qt.Horizontal)
        self.horizontalSlider.setRange(20, 350)
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.horizontalSlider.setValue(new_value)

        # connections
        self.create_connections()
        self.update_image_and_label()

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


    def create_connections(self):
        # Push buttons
        self.horizontalSlider.valueChanged.connect(self.update_image_and_label)

    def update_image_and_label(self):
        self.value = self.horizontalSlider.value()
        self.text_label.setText(f"GSD: {self.value} mm/pixel")

        if self.value < 50:
            pixmap = QtGui.QPixmap(self.list_img[0])
        elif 50 <= self.value <= 100:
            pixmap = QtGui.QPixmap(self.list_img[1])
        elif 100 <= self.value <= 200:
            pixmap = QtGui.QPixmap(self.list_img[2])
        else:
            pixmap = QtGui.QPixmap(self.list_img[3])

        # Calculate the scaled pixmap to fit the label's size
        scaled_pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)

        self.image_label.setPixmap(scaled_pixmap)



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