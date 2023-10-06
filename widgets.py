# standard libraries
import logging
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
from PIL import Image
import sys
import traceback

# custom libraries
from pointify_engine import process

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)


def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message.
    If unavailable (non-console application), log an additional notice.
    """
    if QApplication.instance() is not None:
        errorbox = QMessageBox()
        errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
        errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")


class UncaughtHook(QObject):
    _exception_caught = Signal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        It is triggered each time an uncaught exception occurs.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)


# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()


class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.

    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.

    This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.

        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.

        ``customWidgets`` is a dictionary mapping from class name to class object
        for widgets that you've promoted in the Qt Designer interface. Usually,
        this should be done by calling registerCustomWidget on the QUiLoader, but
        with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = customWidgets

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets, must be a custom widget
                # this will raise KeyError if the user has not supplied the
                # relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                try:
                    widget = self.customWidgets[class_name](parent)

                except (TypeError, KeyError) as e:
                    raise Exception(
                        'No custom widget ' + class_name + ' found in customWidgets param of UiLoader __init__.')

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt4.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.
                # print(name)

            return widget


def loadUi(uifile, baseinstance=None, customWidgets=None,
           workingDirectory=None):
    """
    Dynamically load a user interface from the given ``uifile``.

    ``uifile`` is a string containing a file name of the UI file to load.

    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

    ``customWidgets`` is a dictionary mapping from class name to class object
    for widgets that you've promoted in the Qt Designer interface. Usually,
    this should be done by calling registerCustomWidget on the QUiLoader, but
    with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.

    Return ``baseinstance``, if ``baseinstance`` is not ``None``.  Otherwise
    return the newly created instance of the user interface.
    """

    loader = UiLoader(baseinstance, customWidgets)

    if workingDirectory is not None:
        loader.setWorkingDirectory(workingDirectory)

    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget


# CUSTOM SLIDER
class CustomTickSlider(QSlider):
    def __init__(self, dist_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distinctive_tick_value = dist_v  # Value where distinctive tick should be
        self.setStyleSheet("""
                    QSlider::handle:horizontal {
                        background: white; /* Set handle color */
                        width: 10px;
                        z-index: 1; /* Place handle in front of ticks */
                    }

                    QSlider::sub-page:horizontal {
                        background: #00aaff; /* Color of the filled area */
                    }

                    QSlider::groove:horizontal {
                        background: #ccc; /* Color of the unfilled area */
                        border: none;
                        height: 20px;
                    }

                    QSlider::add-page:horizontal {
                        background: #ccc; /* Color of the area beyond handle */
                    }
                """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Customize normal tick appearance
        tick_height = 5
        tick_color = Qt.darkGray

        # Customize distinctive tick appearance
        distinctive_tick_height = 50
        distinctive_tick_color = Qt.red

        handle_pos = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.value(), self.width()
        )

        for tick_value in range(self.minimum(), self.maximum() + 1):
            tick_pos = self.style().sliderPositionFromValue(
                self.minimum(), self.maximum(), tick_value, self.width()
            )
            if tick_value == self.distinctive_tick_value:

                painter.setPen(QPen(distinctive_tick_color, 2))
                painter.drawLine(tick_pos, 0, tick_pos, distinctive_tick_height)

            else:
                if tick_value % 20 == 0:  # Show normal tick every 10 values
                    painter.setPen(QPen(tick_color, 2))
                    painter.drawLine(tick_pos, 0, tick_pos, tick_height)


# CUSTOM OPEN3D VIEWER
class Custom3dView:
    def __init__(self, cloud, mesh, result):

        app = gui.Application.instance
        self.window = app.create_window("Open3D - Stock viewer", 1024, 768)
        # Since we want the label on top of the scene, we cannot use a layout,
        # so we need to manually layout the window's children.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        self.pcd_rgb = cloud
        self.mesh = mesh

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        mat.point_size = 3 * self.window.scaling
        self.widget3d.scene.add_geometry("Point Cloud RGB", self.pcd_rgb, mat)
        self.widget3d.scene.add_geometry("Mesh", self.mesh, mat)

        self.widget3d.scene.show_geometry("Point Cloud RGB", False)
        self.pc_shown = False

        # add toggle
        em = self.window.theme.font_size
        self.layout = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self.switch = gui.ToggleSwitch("Mesh/PC switch")
        self.switch.set_on_clicked(self.toggle_visibility)
        view_ctrls.add_child(self.switch)
        self.layout.add_child(view_ctrls)
        self.window.add_child(self.layout)

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center + [0, 0, 40], [0, 0, 1])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        self.info.text = result
        self.info.visible = (result != "")
        # We are sizing the info label to be exactly the right size,
        # so since the text likely changed width, we need to
        # re-layout to set the new frame.
        self.window.set_needs_layout()

    def toggle_visibility(self, is_enabled):
        print('toggle')
        if is_enabled:
            self.widget3d.scene.show_geometry("Mesh", False)
            self.widget3d.scene.show_geometry("Point Cloud RGB", True)
        else:
            self.widget3d.scene.show_geometry("Mesh", True)
            self.widget3d.scene.show_geometry("Point Cloud RGB", False)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self.layout.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)

        self.layout.frame = gui.Rect(r.get_right() - width, r.y, width,
                                     height)

        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _on_mouse_widget3d(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, event.y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    # We are sizing the info label to be exactly the right size,
                    # so since the text likely changed width, we need to
                    # re-layout to set the new frame.
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])


def QPixmapFromItem(item):
    """
    Transform a QGraphicsitem into a Pixmap
    :param item: QGraphicsItem
    :return: QPixmap
    """
    pixmap = QPixmap(item.boundingRect().size().toSize())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    # this line seems to be needed for all items except of a LineItem...
    painter.translate(-item.boundingRect().x(), -item.boundingRect().y())
    painter.setRenderHint(QPainter.Antialiasing, True)
    opt = QStyleOptionGraphicsItem()
    item.paint(painter, opt)  # here in some cases the self is needed
    return pixmap


def QPixmapToArray(pixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

    return img


class DualViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Create the main widget and layout
        self.layout = QHBoxLayout()

        # Create the QGraphicsView widgets
        self.view1 = QGraphicsView()
        self.view2 = QGraphicsView()
        self.view1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create the QGraphicsScenes for each view
        self.scene1 = QGraphicsScene()
        self.scene2 = QGraphicsScene()

        # Set the scenes for the views
        self.view1.setScene(self.scene1)
        self.view2.setScene(self.scene2)

        # Connect the view1's zoom event to the zoom_views function
        self.view1.wheelEvent = lambda event: self.zoom_views(event)
        self.view2.wheelEvent = lambda event: self.zoom_views(event)

        # Connect the view1's mouse events to the pan_views function
        self.view1.mousePressEvent = lambda event: self.pan_views(event)
        self.view1.mouseMoveEvent = lambda event: self.pan_views(event)
        self.view1.mouseReleaseEvent = lambda event: self.pan_views(event)

        # Add the views to the layout
        self.layout.addWidget(self.view1)
        self.layout.addWidget(self.view2)

        # Set the central widget
        self.setLayout(self.layout)

        # Store the zoom scale for synchronization
        self.zoom_scale = 1.0
        self.pan_origin = QPointF()

    def load_images_from_path(self, image_path1, image_path2):
        im1 = Image.open(image_path1)
        w1, _ = im1.size

        im2 = Image.open(image_path2)
        w2, _ = im2.size
        self.ratio = w2 / w1

        # Load the images from the file paths
        pixmap1 = QPixmap(image_path1)
        pixmap2 = QPixmap(image_path2)

        # Clear the scenes
        self.scene1.clear()
        self.scene2.clear()

        # Add the images to the respective scenes
        self.scene1.addPixmap(pixmap1)
        self.scene2.addPixmap(pixmap2)

        # Fit the views to the images
        self.view1.fitInView(self.scene1.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Reset the zoom scale and pan origin
        self.zoom_scale = 1.0
        self.pan_origin = QPointF()

    def zoom_views(self, event):
        # Get the zoom factor from the wheel event
        zoom_factor = 1.2 ** (event.angleDelta().y() / 120.0)
        print(zoom_factor, self.ratio)

        # Calculate the new zoom scale
        self.zoom_scale *= zoom_factor

        # Zoom view1
        view1_transform = QTransform()
        view1_transform.scale(self.zoom_scale, self.zoom_scale)
        self.view1.setTransform(view1_transform)

        # Zoom view2
        view2_transform = QTransform()
        view2_transform.scale(self.zoom_scale / self.ratio, self.zoom_scale / self.ratio)
        self.view2.setTransform(view2_transform)

    def pan_views(self, event):
        if event.buttons() == Qt.MouseButtons.LeftButton:
            if event.type() == QEvent.Type.MouseButtonPress:
                # Store the initial mouse position for panning
                self.pan_origin = event.pos()
            elif event.type() == QEvent.Type.MouseMove:
                # Calculate the delta between the current and initial mouse position
                delta = event.pos() - self.pan_origin

                # Pan view1
                view1_horizontal_scroll = self.view1.horizontalScrollBar()
                view1_vertical_scroll = self.view1.verticalScrollBar()
                view1_horizontal_scroll.setValue(view1_horizontal_scroll.value() - delta.x())
                view1_vertical_scroll.setValue(view1_vertical_scroll.value() - delta.y())

                # Pan view2
                view2_horizontal_scroll = self.view2.horizontalScrollBar()
                view2_vertical_scroll = self.view2.verticalScrollBar()
                view2_horizontal_scroll.setValue(view2_horizontal_scroll.value() - delta.x())
                view2_vertical_scroll.setValue(view2_vertical_scroll.value() - delta.y())

                # Update the pan origin for the next mouse move
                self.pan_origin = event.pos()
            elif event.type() == event.MouseButtonRelease:
                # Reset the pan origin
                self.pan_origin = QPointF()


class SimpleViewer(QGraphicsView):
    def __init__(self, original_img_path, img_list, parent):
        super(SimpleViewer, self).__init__(parent)
        self.sourceImage = QImage()
        self.destinationImage = QImage()
        self.sourceImage.load(original_img_path)

        self.current_img = 0
        self.img_list = img_list
        self.destinationImage.load(self.img_list[0])

        self.resultSize = self.sourceImage.size()
        self.resultImage = QImage(self.resultSize, QImage.Format_ARGB32_Premultiplied)

        self._photo = QGraphicsPixmapItem()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.r = ''

        self._scene.addItem(self._photo)

    def showEvent(self, event):
        self.recalculate_result()

    def recalculate_result(self):
        self.destinationImage.load(self.img_list[self.current_img])
        painter = QPainter(self.resultImage)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.resultImage.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawImage(0, 0, self.destinationImage)
        painter.setCompositionMode(QPainter.CompositionMode_Screen)
        painter.drawImage(0, 0, self.sourceImage)
        painter.setCompositionMode(QPainter.CompositionMode_DestinationOver)
        painter.fillRect(self.resultImage.rect(), QColor(255, 255, 255))
        painter.end()
        self._photo.setPixmap(QPixmap.fromImage(self.resultImage))

        xmin, xmax, ymin, ymax = process.get_nonzero_coord(self.img_list[self.current_img])
        top = ymin
        left = xmin
        h = ymax - ymin
        w = xmax - xmin

        rect = QRectF(left, top, w, h)
        self.fitInView(rect, Qt.KeepAspectRatio)


def createLineIterator(P1, P2, img):
    """
    Source: https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
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

    return itbuffer


class PhotoViewer(QGraphicsView):
    photoClicked = Signal(QPoint)
    endDrawing_rect = Signal()
    end_point_selection = Signal()
    endDrawing_line_meas = Signal()

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setFrameShape(QFrame.NoFrame)

        self.rect = False
        self.select_point = False
        self.line_meas = False

        self.setMouseTracking(True)
        self.origin = QPoint()

        self._current_rect_item = None
        self._current_line_item = None
        self._current_point = None
        self._current_path = None

        self.crop_coords = []

        self.pen = QPen()
        self.pen.setStyle(Qt.DashDotLine)
        self.pen.setWidth(4)
        self.pen.setColor(QColor(255, 0, 0, a=255))
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setJoinStyle(Qt.RoundJoin)

        self.meas_color = QColor(0, 100, 255, a=255)
        self.pen_yolo = QPen()
        # self.pen.setStyle(Qt.DashDotLine)
        self.pen_yolo.setWidth(2)
        self.pen_yolo.setColor(self.meas_color)
        self.pen_yolo.setCapStyle(Qt.RoundCap)
        self.pen_yolo.setJoinStyle(Qt.RoundJoin)

    def has_photo(self):
        return not self._empty

    def showEvent(self, event):
        self.fitInView()
        super(PhotoViewer, self).showEvent(event)

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        print(rect)
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                print('unity: ', unity)
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                print('view: ', viewrect)
                scenerect = self.transform().mapRect(rect)
                print('scene: ', viewrect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def clean_scene(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsRectItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsTextItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsPolygonItem):
                self._scene.removeItem(item)

    def clean_scene_line(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsLineItem):
                self._scene.removeItem(item)

    def clean_scene_rectangle(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsRectItem):
                self._scene.removeItem(item)

    def clean_scene_poly(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsPolygonItem):
                self._scene.removeItem(item)

    def clean_scene_text(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsTextItem):
                self._scene.removeItem(item)

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def toggleDragMode(self):
        if self.rect or self.select_point:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                self.setDragMode(QGraphicsView.NoDrag)
            elif not self._photo.pixmap().isNull():
                self.setDragMode(QGraphicsView.ScrollHandDrag)

    def add_poly(self, coordinates):
        # Create a QPolygonF from the coordinates
        polygon = QPolygonF()
        for x, y in coordinates:
            polygon.append(QPointF(x, y))

        # Create a QGraphicsPolygonItem and set its polygon
        polygon_item = QGraphicsPolygonItem(polygon)
        fill_color = QColor(0, 255, 255, 100)
        polygon_item.setBrush(fill_color)  # Set fill color

        # Add the QGraphicsPolygonItem to the scene
        self._scene.addItem(polygon_item)

    def add_yolo_box(self, text, x1, y1, x2, y2):
        # add box
        box = QGraphicsRectItem()
        box.setPen(self.pen_yolo)

        r = QRectF(x1, y1, x2 - x1, y2 - y1)
        box.setRect(r)

        # add text
        text_item = QGraphicsTextItem()
        text_item.setPos(x1, y1)
        text_item.setHtml(
            "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text + "</div>")

        # add elements to scene
        self._scene.addItem(box)
        self._scene.addItem(text_item)

    def add_list_poly(self, list_objects):
        for el in list_objects:
            # Create a QPolygonF from the coordinates
            polygon = QPolygonF()
            for x, y in el.coords:
                polygon.append(QPointF(x, y))

            # Create a QGraphicsPolygonItem and set its polygon
            polygon_item = QGraphicsPolygonItem(polygon)
            fill_color = QColor(0, 255, 255, 100)
            polygon_item.setBrush(fill_color)  # Set fill color

            # Add the QGraphicsPolygonItem to the scene
            self._scene.addItem(polygon_item)

    def add_list_infos(self, list_objects, only_name=False):
        for el in list_objects:
            x1, y1, x2, y2, score, class_id = el.yolo_bbox
            text = el.name
            text2 = str(round(el.area, 2)) + 'm² '
            text3 = str(round(el.volume, 2)) + 'm³'

            print(f'adding {text} to viewer')

            # add text 1
            text_item = QGraphicsTextItem()
            text_item.setPos(x1, y1)
            text_item.setHtml(
                "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text + "</div>")

            self._scene.addItem(text_item)

            if not only_name:
                # add text 2 and 3
                text_item2 = QGraphicsTextItem()
                text_item2.setPos(x1, y2)
                text_item2.setHtml(
                    "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text2 + "<br>" + text3 + " </div>")
                self._scene.addItem(text_item2)

    def add_list_boxes(self, list_objects):
        for el in list_objects:
            x1, y1, x2, y2, score, class_id = el.yolo_bbox

            # add box
            box = QGraphicsRectItem()
            box.setPen(self.pen_yolo)

            r = QRectF(x1, y1, x2 - x1, y2 - y1)
            box.setRect(r)

            # add elements to scene
            self._scene.addItem(box)

    def get_coord(self, QGraphicsRect):
        rect = QGraphicsRect.rect()
        coord = [rect.topLeft(), rect.bottomRight()]
        print(coord)

        return coord

    def get_selected_point(self):
        print(self._current_point)
        return self._current_point

    def set_height_data(self, height_data):
        self.height_values = height_data
        print(np.shape(np.array(height_data)))

    # mouse events
    def wheelEvent(self, event):
        print(self._zoom)
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def mousePressEvent(self, event):
        if self.rect:
            self._current_rect_item = QGraphicsRectItem()
            self._current_rect_item.setFlag(QGraphicsItem.ItemIsSelectable)
            self._current_rect_item.setPen(self.pen)
            self._scene.addItem(self._current_rect_item)
            self.origin = self.mapToScene(event.pos())
            r = QRectF(self.origin, self.origin)
            self._current_rect_item.setRect(r)

        elif self.select_point:
            self._current_point = self.mapToScene(event.pos())
            self.get_selected_point()
            self.select_point = False
            self.end_point_selection.emit()

        elif self.line_meas:
            self._current_line_item = QGraphicsLineItem()
            self._current_line_item.setPen(self.pen_yolo)

            self._scene.addItem(self._current_line_item)
            self.origin = self.mapToScene(event.pos())

            self._current_line_item.setLine(QLineF(self.origin, self.origin))

        else:
            if self._photo.isUnderMouse():
                self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rect:
            if self._current_rect_item is not None:
                new_coord = self.mapToScene(event.pos())
                r = QRectF(self.origin, new_coord)
                self._current_rect_item.setRect(r)
        elif self.line_meas:
            if self._current_line_item is not None:
                self.new_coord = self.mapToScene(event.pos())
                self._current_line_item.setLine(QLineF(self.origin, self.new_coord))

        super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect:
            self.rect = False
            self.origin = QPoint()
            if self._current_rect_item is not None:
                self.crop_coords = self.get_coord(self._current_rect_item)
                self.endDrawing_rect.emit()
                print('rectangle ROI added: ' + str(self.crop_coords))
            self._current_rect_item = None
            self.toggleDragMode()

        elif self.line_meas:
            self.line_meas = False

            if self._current_line_item is not None:
                # compute line values
                p1 = np.array([int(self.origin.x()), int(self.origin.y())])
                p2 = np.array([int(self.new_coord.x()), int(self.new_coord.y())])
                print(p1, p2)
                line_values = createLineIterator(p1, p2, self.height_values)

                self.line_values_final = line_values[:, 2]

                # emit signal (end of measure)
                self.endDrawing_line_meas.emit()
                print('Line meas. added')

            self.origin = QPoint()
            self._current_line_item = None
            self.toggleDragMode()

        super(PhotoViewer, self).mouseReleaseEvent(event)


class StandardItem(QStandardItem):
    def __init__(self, txt='', image_path='', font_size=10, set_bold=False, color=QColor(0, 0, 0)):
        super().__init__()

        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)

        if image_path:
            image = QImage(image_path)
            self.setData(image, Qt.DecorationRole)


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 2px dashed #aaa
            }
        ''')


class TestListView(QListWidget):
    dropped = Signal(list)

    def __init__(self, type, parent=None):
        super(TestListView, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(72, 72))
        # self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFixedWidth(350)
        self.setStyleSheet('''
                    QListWidget{
                        border: 2px dashed #aaa
                    }
                ''')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            # self.emit(QtCore.SIGNAL("dropped"), links)
            self.dropped.emit(links)
        else:
            event.ignore()
