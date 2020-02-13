import math
import random
from datetime import datetime

import numpy as np
import PIL.Image
import PIL.ImageQt
from PyQt5.QtGui import QImage, QColor, QPixmap, QPalette
from PyQt5.QtWidgets import QAction, qApp, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QMainWindow, QScrollArea, QSizePolicy, QMenu, QMenuBar

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks

from PyQt5 import QtCore, QtWidgets, QtGui, Qt


class Application(QtWidgets.QWidget):
    def __init__(self, network_pkl=None, truncation_psi=None):
        super().__init__()
        self.zoom_on = False

        self.create_gui()

        # self.resize(1024, 1024)

        if network_pkl is not None:
            self.init_network(network_pkl, truncation_psi)

        self.init_weights_grid()
        self.generate_z_vector()

        # self.init_

    def create_gui(self):
        seed_lbl = QLabel('Seed: ')
        self.seed_text_box = QLineEdit(str(random.randint(0, 9999)))

        self.generate_button = QPushButton(text='Generate image')
        self.generate_button.clicked.connect(self.generate_image)

        self.output_image = QLabel()
        self.output_image.setBackgroundRole(QPalette.Base)
        self.output_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.output_image.setScaledContents(True)

        self.scaleFactor = 0.0

        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.output_image)
        self.scroll_area.setVisible(False)

        self.tools_layout = QHBoxLayout()
        self.tools_layout.addWidget(self.generate_button)
        self.tools_layout.addWidget(seed_lbl)
        self.tools_layout.addWidget(self.seed_text_box)

        # self.second_row = QHBoxLayout()
        # self.second_row.addWidget(self.scroll_area, 1)
        # self.second_row.addLayout(self.weights_grid, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.tools_layout)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

        # main_layout = QGridLayout(self)
        # main_layout.addWidget(self.generate_button, 0, 0, 1, 1)
        # main_layout.addWidget(seed_lbl, 0, 1, 1, 1)
        # main_layout.addWidget(self.seed_text_box, 0, 2, 1, 1)
        # main_layout.addWidget(self.scroll_area, 1, 0, 1, 3)

        # img = PIL.Image.open('test.jpg', 'r')
        # self.set_image(img)

    def init_weights_grid(self):
        self.mu_text_box = QLineEdit('0')
        self.sigma_text_box = QLineEdit('1')
        self.generate_weights_btn = QPushButton(text='Generate weights')
        self.generate_weights_btn.clicked.connect(self.generate_z_vector)

        self.tools_layout.addWidget(self.generate_weights_btn, 1)

        self.tools_layout.addWidget(QLabel('Mu='))
        self.tools_layout.addWidget(self.mu_text_box)

        self.tools_layout.addWidget(QLabel('Sigma='))
        self.tools_layout.addWidget(self.sigma_text_box)

        self.weights_grid_widget = QtWidgets.QDialog(self)
        self.weights_grid_widget.show()
        self.weights_grid = QGridLayout(self.weights_grid_widget)

        self.weights_input_lines = []

        number_of_weights = self.Gs.input_shape[1]
        side_size = math.floor(math.sqrt(number_of_weights))

        for row_n in range(side_size + 1):
            for col_n in range(side_size + 1):
                if len(self.weights_input_lines) >= number_of_weights:
                    break
                box = QLineEdit()
                self.weights_grid.addWidget(box, row_n, col_n)
                self.weights_input_lines.append(box)

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.output_image.resize(self.scaleFactor * self.output_image.pixmap().size())

        self.adjustScrollBar(self.scroll_area.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scroll_area.verticalScrollBar(), factor)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def zoom_in(self):
        self.scaleImage(1.25)

    def zoom_out(self):
        self.scaleImage(0.8)

    def normal_size(self):
        self.output_image.adjustSize()
        self.scaleFactor = 1.0

    def fit_to_window(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scroll_area.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def set_image(self, image: PIL.Image):
        self.image = image
        self.qtimg = PIL.ImageQt.ImageQt(image)
        self.output_image.setPixmap(QPixmap.fromImage(self.qtimg))

        self.scaleFactor = 1.0

        self.scroll_area.setVisible(True)

        self.output_image.adjustSize()

    def init_network(self, network_pkl, truncation_psi):
        self._G, self._D, self.Gs = pretrained_networks.load_networks(network_pkl)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

        self.Gs_kwargs = dnnlib.EasyDict()
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            self.Gs_kwargs.truncation_psi = truncation_psi

    def generate_z_vector(self):
        seed = int(self.seed_text_box.text())
        rnd = np.random.RandomState(seed)
        mu = float(self.mu_text_box.text())
        sigma = float(self.sigma_text_box.text())
        z = rnd.randn(1, *self.Gs.input_shape[1:]) * math.sqrt(sigma) + mu  # [minibatch, component]

        for i, input_line in enumerate(self.weights_input_lines):
            input_line.setText(str(z[0][i]))

    def get_z_vector(self):
        return np.array([[float(line.text()) for line in self.weights_input_lines]])

    def generate_w_vector(self):
        z = self.get_z_vector()

        w = self.Gs.components.mapping.run(z, None)

    def get_w_vector(self):
        pass

    def generate_image(self):
        seed = int(self.seed_text_box.text())
        # seed = int(datetime.now().timestamp())
        rnd = np.random.RandomState(seed)
        # z = rnd.randn(1, *self.Gs.input_shape[1:])  # [minibatch, component]
        z = self.get_z_vector()

        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]

        w = self.get_w_vector()
        images = self
        # images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]

        img = PIL.Image.fromarray(images[0], 'RGB')
        img.save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        self.set_image(img)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        modifiers = event.modifiers()
        angle = event.angleDelta()
        print(f'common angle = {angle}')
        if modifiers == QtCore.Qt.ControlModifier:
            print(f'mods = {modifiers}')
            angle = event.angleDelta()
            if angle.y() > 0:
                self.zoom_in()
            elif angle.y() < 0:
                self.zoom_out()
            event.accept()

    # def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
    #     if event.type() == QtCore.QEvent.Wheel and obj == self.scroll_area and self.zoom_on:
    #         return False
    #
    #     return True


def run_window(network_pkl=None, truncation_psi=None):
    app = QtWidgets.QApplication([])

    widget = Application(network_pkl, truncation_psi)
    widget.show()

    app.exec_()

    print('Finished')


if __name__ == '__main__':
    run_window()
