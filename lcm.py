import sys
import os
import re
import logging
from pathlib import Path

import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

import mplcursors

from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun, get_body
import astropy.units as u
from astropy.visualization import quantity_support
from astroplan import moon_illumination

from PyQt5.QtCore import Qt, QSize, pyqtSlot, QSortFilterProxyModel, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLineEdit, QTextEdit, QGroupBox,
    QComboBox, QSpinBox, QCheckBox, QFileDialog, QDialogButtonBox,
    QMessageBox, QTabWidget, QSizePolicy, QFrame, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QTableView, QHeaderView, QMenuBar, QSystemTrayIcon
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QIcon
)

from datetime import datetime

from simbad_util import create_simbad_link
import importlib

from lcm_app.tess.threads import (
    TESSFutureThread,
    TESSDownloadThread,
    TESSPrefetchThread,
    TESSSectorSearchThread,
    TESSDataFetchThread,
    tess_future_sectors,
)


# If you want to write file path manually, comment out below line
from dropbox_util import find_dropbox_folder

try:
    import dropbox_util

    dropbox_available = True
except ImportError:
    dropbox_available = False

from lcm_app.controller import TESSController
from lcm_app.config import load_download_dir, save_download_dir
from lcm_app.common.ui_utils import show_info
from lcm_app.tess.ui_query import TessQueryMixin
from lcm_app.tess.ui_outlier import OutlierMixin
from lcm_app.tess.ui_detrending import DetrendingMixin
from lcm_app.tess.ui_phase import PhaseMixin


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.data_file_path = None
        self.minima_file_path = None
        self.minima_file_loaded = False
        self.fig0 = None
        self.txt_window = None
        self.star_name = None
        self.sector_numbers = []
        self.tess_window = None
        self.simbad_window = None
        self.data_window = None
        self.tess_window = None
        self.custom_data_window = None
        self._kepler_cls = None
        self._astro_cls = None
        self._custom_cls = None
        self._download_dir = None
        self.info_timer = QTimer(self)

        self.initui()

    def initui(self):
        self.setWindowTitle("lcm")
        self.setWindowIcon(QIcon("images/star.ico"))

        self.data_select_btn = QPushButton()
        self.data_select_btn.setIcon(QIcon("images/lc1.png"))
        self.data_select_btn.setIconSize(QSize(50, 40))
        self.data_select_btn.clicked.connect(self.select_data_file)

        self.minima_select_btn = QPushButton()
        self.minima_select_btn.setIcon(QIcon("images/minima1.png"))
        self.minima_select_btn.setIconSize(QSize(50, 40))
        self.minima_select_btn.clicked.connect(self.select_minima_file)

        self.tess_btn = QPushButton()
        self.tess_btn.setIcon(QIcon("images/tess.png"))
        self.tess_btn.setIconSize(QSize(50, 40))
        self.tess_btn.clicked.connect(self.open_tess_window)

        self.kepler_btn = QPushButton()
        self.kepler_btn.setIcon(QIcon("images/kepler.png"))
        self.kepler_btn.setIconSize(QSize(50, 40))
        self.kepler_btn.clicked.connect(self.open_kepler)

        self.custom_data_btn = QPushButton("LC Import")
        self.custom_data_btn.setIcon(QIcon("images/import_lc.png"))
        self.custom_data_btn.setIconSize(QSize(50, 40))
        self.custom_data_btn.setToolTip("Import arbitrary light curves")
        self.custom_data_btn.clicked.connect(self.open_custom_data_tools)
        self.custom_data_btn.setVisible(False)  # keep code, hide button for now

        self.simbad_btn = QPushButton()
        self.simbad_btn.setIcon(QIcon("images/simbad.png"))
        self.simbad_btn.setIconSize(QSize(50, 40))
        self.simbad_btn.clicked.connect(self.open_simbad_window)

        self.astro_app_btn = QPushButton()
        self.astro_app_btn.setIcon(QIcon("images/icon.ico"))
        self.astro_app_btn.setIconSize(QSize(50, 40))
        self.astro_app_btn.clicked.connect(self.open_astro_app)

        self.warning_label = QLabel()
        self.warning_label.setFixedHeight(18)
        self.warning_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.warning_label.setWordWrap(False)

        # Place all buttons on a single row for quick access
        button_row = QHBoxLayout()
        button_row.addWidget(self.data_select_btn)
        button_row.addWidget(self.minima_select_btn)
        button_row.addWidget(self.custom_data_btn)
        button_row.addWidget(self.tess_btn)
        button_row.addWidget(self.kepler_btn)
        button_row.addWidget(self.simbad_btn)
        button_row.addWidget(self.astro_app_btn)

        self.menu_bar = QMenuBar(self)
        settings_menu = self.menu_bar.addMenu("Settings")
        download_action = settings_menu.addAction("Download Folder…")
        download_action.triggered.connect(self.open_download_settings)
        help_menu = self.menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about_dialog)
        self.menu_bar.setStyleSheet(
            "QMenuBar { background-color: #f2f2f2; border: none; color: #222; }"
            "QMenuBar::item { background: transparent; color: #222; }"
            "QMenuBar::item:selected { background: #d0d0d0; color: #000; }"
            "QMenu { background-color: #f2f2f2; color: #222; }"
            "QMenu::item:selected { background: #d0d0d0; color: #000; }"
        )

        main_layout = QVBoxLayout()
        main_layout.setMenuBar(self.menu_bar)
        main_layout.addLayout(button_row)
        main_layout.addWidget(self.warning_label)

        self.setLayout(main_layout)

        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.screen_width = self.screen_resolution.width()
        self.screen_height = self.screen_resolution.height()

        self.obs = QWidget()
        self.txt_window = QWidget()

        if self.screen_width == 2560 and self.screen_height == 1440:
            self.setGeometry(300, 600, 200, 10)
        elif self.screen_width == 1920 and self.screen_height == 1080:
            self.setGeometry(110, 500, 200, 10)

    def show_info_message(self, text, duration=4000):
        """Show a short inline message under the main buttons (warning_label)."""
        self.warning_label.setText(f"<span style='color: green;'>{text}</span>")
        self.info_timer.stop()
        self.info_timer.setSingleShot(True)
        try:
            self.info_timer.timeout.disconnect()
        except Exception:
            pass
        self.info_timer.timeout.connect(self._hide_info_label)
        self.info_timer.start(duration)

    def _hide_info_label(self):
        self.warning_label.setText("")

    def show_about_dialog(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About lcm")
        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setText(
            "lcm - Scientific software for visualization and analysis of photometric light curves with integrated modules for TESS / Kepler / K2 data processing and observation planning.<br>"
            "Furkan Akar<br>"
            '<a href="mailto:efurkanakar@gmail.com">efurkanakar@gmail.com</a>'
        )
        msg.exec_()

    def open_download_settings(self):
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Download Folder (TESS / Kepler / K2)")
            dlg.setWindowIcon(QIcon("images/star.ico"))
            vbox = QVBoxLayout(dlg)

            use_default = QCheckBox("Use mastDownload default")
            path_edit = QLineEdit()
            path_edit.setReadOnly(True)
            browse_btn = QPushButton("Browse…")

            saved_dir = load_download_dir()
            if saved_dir and saved_dir.exists():
                use_default.setChecked(False)
                path_edit.setText(str(saved_dir))
                path_edit.setEnabled(True)
            else:
                use_default.setChecked(True)
                path_edit.setEnabled(False)

            def toggle(state):
                checked = state == Qt.Checked
                path_edit.setEnabled(not checked)
                if checked:
                    path_edit.clear()

            def browse():
                folder = QFileDialog.getExistingDirectory(self, "Select download folder")
                if folder:
                    path_edit.setText(folder)
                    use_default.setChecked(False)

            use_default.stateChanged.connect(toggle)
            browse_btn.clicked.connect(browse)

            row = QHBoxLayout()
            row.addWidget(use_default)
            vbox.addLayout(row)

            row2 = QHBoxLayout()
            row2.addWidget(path_edit)
            row2.addWidget(browse_btn)
            vbox.addLayout(row2)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            vbox.addWidget(buttons)

            def accept():
                if use_default.isChecked():
                    save_download_dir(None)
                    dlg.accept()
                    return
                path = path_edit.text().strip()
                if not path:
                    QMessageBox.warning(self, "Warning", "Please select a folder or use default.")
                    return
                p = Path(path).expanduser()
                if not p.exists():
                    QMessageBox.warning(self, "Warning", "Selected folder does not exist.")
                    return
                save_download_dir(str(p))
                dlg.accept()

            buttons.accepted.connect(accept)
            buttons.rejected.connect(dlg.reject)

            dlg.resize(420, 100)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while opening settings: {e}")

    def open_kepler(self):
        if self._kepler_cls is None:
            from kepler import KeplerApp  # Lazy import to speed initial launch
            self._kepler_cls = KeplerApp
        self.kepler = self._kepler_cls()
        self.kepler.show()

    def open_custom_data_tools(self):
        if self._custom_cls is None:
            from custom_data_window import CustomDataWindow  # Lazy import to speed initial launch
            self._custom_cls = CustomDataWindow
        self.custom_data_window = self._custom_cls(self.screen_width, self.screen_height)
        self.custom_data_window.show()

    def open_astro_app(self):
        if self._astro_cls is None:
            from astro_app import AstroApp  # Lazy import to speed initial launch
            self._astro_cls = AstroApp
        self.astro_app = self._astro_cls()
        self.astro_app.show()

        if self.star_name:
            self.astro_app.star_input.setText(self.star_name)

    def close_txt_window(self):
        if self.txt_window and self.txt_window.isVisible():
            self.txt_window.close()

    @pyqtSlot()
    def select_data_file(self):
        if dropbox_available:
            dropbox_path = dropbox_util.find_dropbox_folder()
            initial_dir = os.path.join(dropbox_path, "PCEB_project/observations/reduction_outputs")
        else:
            initial_dir = "C:/Users/furka/Documents/data"

        self.data_file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", initial_dir,
                                                             "Data Files (*.dat)")

        if self.data_file_path:
            self.warning_label.setText("Press Enter to save the plot")
            self.plot_data(self.data_file_path)
            self.info(self.data_file_path)

            self.star_name, _ = get_star_name_and_telescope(self.data_file_path)

            if self.tess_window and hasattr(self.tess_window, 'star_name_input'):
                self.tess_window.star_name_input.setText(self.star_name)

            if self.simbad_window and hasattr(self, 'simbad_star_name_input'):
                self.simbad_star_name_input.setText(self.star_name)

    def plot_data(self, file_path):
        if file_path:
            try:
                if self.fig0:
                    self.fig0.clf()
                    plt.close(self.fig0)
                    plt.close()
                    self.close_txt_window()

                data = pd.read_csv(file_path, sep=r'\s+', skiprows=1, names=["time", "flux", "error"])

                time = data["time"].values
                flux = data["flux"].values
                error = data["error"].values

                date = get_observation_date(file_path)
                star_name, telescope = get_star_name_and_telescope(file_path)
                filter_color, filter_name = get_filter(file_path)

                self.fig0 = plt.figure("Reduced Photometric Light Curve Data")
                manager = plt.get_current_fig_manager()
                manager.window.setWindowIcon(QIcon("images/star.ico"))

                if self.screen_width == 2560 and self.screen_height == 1440:
                    self.fig0.canvas.manager.window.setGeometry(600, 150, 1100, 700)
                elif self.screen_width == 1920 and self.screen_height == 1080:
                    self.fig0.canvas.manager.window.setGeometry(430, 30, 970, 540)

                self.ax = self.fig0.add_subplot(111)

                self.ax.errorbar(time, flux, yerr=error, marker=".", markersize=5, capsize=2, elinewidth=1.5, ls="None",
                                 color=filter_color)
                self.ax.set_xlabel('Time [BJD-TDB]')
                self.ax.set_ylabel('Relative Flux')
                self.ax.set_title(f'Date: {date}, Object: {star_name}, Telescope: {telescope}, Filter: {filter_name}')

                self.fig0.subplots_adjust(left=0.086, right=0.975, top=0.925, bottom=0.116)

                self.fig0.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.ax.minorticks_on()
                self.ax.grid(True, which='major', linestyle='-', linewidth=0.1)
                self.fig0.canvas.draw()

            except Exception as e:
                print("Error reading data and plotting:", e)
                self.warning_label.setText("An error occurred while reading data and plotting.")
        else:
            self.warning_label.setText("You should select a .dat file first.")

    def on_key_press(self, event):
        if event.key == 'enter':
            self.save_plot()

    def save_plot(self):
        if not self.minima_file_loaded:
            QMessageBox.warning(self, "Warning", "Minima Data File is not selected.")
            return

        if not self.txt_window:
            QMessageBox.warning(self, "Warning", "Text window is not open. Can't save the plot.")
            return

        if self.fig0:
            if self.minima_file_path:
                output_filename = self.minima_file_path.replace('.txt', '-profile.png')
            else:
                QMessageBox.warning(self, "Warning", "Minima Data File is not selected.")
                return
        else:
            QMessageBox.warning(self, "Warning", "Data File is not selected.")
            return

        self.fig0.savefig(output_filename, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Information", f"Plot saved as {output_filename}.")
        self.warning_label.clear()

    def info(self, file_path):
        global m_file_path
        if file_path:
            data_folder = os.path.dirname(file_path)
            m_file_path = None

            for file_name in os.listdir(data_folder):
                if file_name.endswith(".xls"):
                    m_file_path = os.path.join(data_folder, file_name)
                    break

            if not m_file_path:
                for file_name in os.listdir(data_folder):
                    if file_name.endswith(".tbl"):
                        m_file_path = os.path.join(data_folder, file_name)
                        break

            self.obs.close()

            self.obs = QWidget()

            if self.screen_width == 2560 and self.screen_height == 1440:
                self.obs.setGeometry(600, 900, 190, 400)
            elif self.screen_width == 1920 and self.screen_height == 1080:
                self.obs.setGeometry(430, 650, 200, 380)

            self.obs.setWindowTitle("Info")
            self.obs.setWindowIcon(QIcon("images/star.ico"))

            if m_file_path is not None:
                try:
                    layout = QVBoxLayout()
                    df = pd.read_csv(m_file_path, delimiter="\t")

                    exp_time = int(df["EXPTIME"].mean())
                    start_time = Time(df['JD_UTC'].iloc[0], format='jd', scale='tai').to_datetime()
                    end_time = Time(df['JD_UTC'].iloc[-1], format='jd', scale='tai').to_datetime()

                    start_time += pd.Timedelta(hours=3)
                    end_time += pd.Timedelta(hours=3)

                    rel_flux_SNR_T1 = [round(value) for value in df['rel_flux_SNR_T1'].iloc[:3]]
                    peak_T1 = round(df['Peak_T1'].iloc[:3])
                    sky = [round(value) for value in df['Sky/Pixel_T1'].iloc[:3]]

                    cursor = mplcursors.cursor(self.ax, hover=True, bindings={"toggle_visible": "h"})
                    cursor.connect("add", lambda sel: (sel.annotation.set_text(
                        "{:^12s}: {:^13g}\n{:^10s}: {:^10g}\n{:^10s}: {:^10g}".format(
                            "SNR",
                            round(df['rel_flux_SNR_T1'][sel.index]),
                            "Star Count",
                            round(df['Peak_T1'][sel.index]),
                            "BG Count",
                            round(df['Sky/Pixel_T1'][sel.index])
                        )
                    ),
                                                       sel.annotation.set_bbox({'facecolor': 'white', 'alpha': 0.7})
                    ))

                    self.warning_label.setText("<font color='blue'>Press 'h' for the hover on and off.</font>")

                    lab1 = QLabel("Exposure Time")
                    lab1.setAlignment(Qt.AlignCenter)
                    txt1 = QTextEdit(str(exp_time))
                    txt1.setAlignment(Qt.AlignCenter)
                    txt1.setReadOnly(True)
                    txt1.setFixedHeight(25)

                    lab2 = QLabel("Start Time")
                    lab2.setAlignment(Qt.AlignCenter)
                    txt2 = QTextEdit(start_time.strftime('%H:%M'))
                    txt2.setAlignment(Qt.AlignCenter)
                    txt2.setReadOnly(True)
                    txt2.setFixedHeight(25)

                    lab3 = QLabel("End Time")
                    lab3.setAlignment(Qt.AlignCenter)
                    txt3 = QTextEdit(end_time.strftime('%H:%M'))
                    txt3.setAlignment(Qt.AlignCenter)
                    txt3.setReadOnly(True)
                    txt3.setFixedHeight(25)

                    lab4 = QLabel("F3 SNR Values")
                    lab4.setAlignment(Qt.AlignCenter)
                    txt4 = QTextEdit(", ".join(map(str, rel_flux_SNR_T1)))
                    txt4.setAlignment(Qt.AlignCenter)
                    txt4.setReadOnly(True)
                    txt4.setFixedHeight(25)

                    lab5 = QLabel("F3 Count Values")
                    lab5.setAlignment(Qt.AlignCenter)
                    txt5 = QTextEdit(", ".join(map(str, peak_T1)))
                    txt5.setAlignment(Qt.AlignCenter)
                    txt5.setReadOnly(True)
                    txt5.setFixedHeight(25)

                    lab6 = QLabel("F3 BG Count Values")
                    lab6.setAlignment(Qt.AlignCenter)
                    txt6 = QTextEdit(", ".join(map(str, sky)))
                    txt6.setAlignment(Qt.AlignCenter)
                    txt6.setReadOnly(True)
                    txt6.setFixedHeight(25)

                    lay1 = QVBoxLayout()
                    lay1.addWidget(lab1)
                    lay1.addWidget(txt1)

                    lay2 = QVBoxLayout()
                    lay2.addWidget(lab2)
                    lay2.addWidget(txt2)

                    lay3 = QVBoxLayout()
                    lay3.addWidget(lab3)
                    lay3.addWidget(txt3)

                    lay4 = QVBoxLayout()
                    lay4.addWidget(lab4)
                    lay4.addWidget(txt4)

                    lay5 = QVBoxLayout()
                    lay5.addWidget(lab5)
                    lay5.addWidget(txt5)

                    lay6 = QVBoxLayout()
                    lay6.addWidget(lab6)
                    lay6.addWidget(txt6)

                    hlayout = QVBoxLayout()
                    hlayout.addLayout(lay1)
                    hlayout.addLayout(lay2)
                    hlayout.addLayout(lay3)
                    hlayout.addLayout(lay4)
                    hlayout.addLayout(lay5)
                    hlayout.addLayout(lay6)

                    self.ra_deg = df["RAOBJ2K"][0] * u.hour
                    self.dec_deg = df["DECOBJ2K"][0] * u.deg

                    obj = SkyCoord(ra=self.ra_deg, dec=self.dec_deg)

                    star_name, telescope = get_star_name_and_telescope(file_path)

                    if telescope == "T80" or telescope == "T35":
                        location = EarthLocation(lat=39.84 * u.deg, lon=32.782 * u.deg, height=1256.69 * u.m)
                    elif telescope == "T100":
                        location = EarthLocation(lat=36.825 * u.deg, lon=30.335 * u.deg, height=1256.69 * u.m)

                    utcoffset = 3 * u.hour
                    time = Time(df['JD_UTC'], format='jd', scale='tai') + utcoffset

                    new_time = time[0] - 12 * u.hour
                    midnight_new_time_str = new_time.datetime.strftime('%Y-%m-%d')
                    midnight = Time(midnight_new_time_str + ' 21:00:00')

                    delta_midnight = np.arange(-12 * 60, 13 * 60, 1) * u.minute
                    times = midnight + delta_midnight

                    local_times = times + utcoffset

                    frame = AltAz(obstime=times, location=location)
                    sunaltaz = get_sun(times).transform_to(frame)

                    moon = get_body("moon", times)
                    moonaltaz = moon.transform_to(frame)

                    objaltaz = obj.transform_to(frame)

                    illumination = moon_illumination(Time(df['JD_UTC'].iloc[0], format='jd', scale='tai'))

                    nearest_start_time_index = np.argmin(
                        np.abs((local_times.datetime - start_time).astype('timedelta64[s]')))
                    nearest_end_time_index = np.argmin(
                        np.abs((local_times.datetime - end_time).astype('timedelta64[s]')))

                    start_time_moon_alt = moonaltaz.alt.deg[nearest_start_time_index]
                    end_time_moon_alt = moonaltaz.alt.deg[nearest_end_time_index]

                    start_time_obj_alt = objaltaz.alt.deg[nearest_start_time_index]
                    end_time_obj_alt = objaltaz.alt.deg[nearest_end_time_index]

                    start_alt_difference = np.abs(start_time_obj_alt - start_time_moon_alt)
                    end_alt_difference = np.abs(end_time_obj_alt - end_time_moon_alt)

                    lab7 = QLabel("Init/End Moon-Object Separation")
                    lab7.setAlignment(Qt.AlignCenter)
                    txt7 = QTextEdit(
                        str(int(np.round(start_alt_difference))) + " - " + str(int(np.round(end_alt_difference))))

                    txt7.setAlignment(Qt.AlignCenter)
                    txt7.setReadOnly(True)
                    txt7.setFixedHeight(25)

                    lay7 = QVBoxLayout()
                    lay7.addWidget(lab7)
                    lay7.addWidget(txt7)

                    hlayout.addLayout(lay7)

                    layout.addLayout(hlayout)

                    quantity_support()

                    self.fig2 = plt.figure("Observation Visualization")
                    manager = plt.get_current_fig_manager()
                    manager.window.setWindowIcon(QIcon("images/star.ico"))
                    self.ax = self.fig2.add_subplot(111)

                    if self.screen_width == 2560 and self.screen_height == 1440:
                        self.fig2.canvas.manager.window.setGeometry(800, 900, 900, 400)
                    elif self.screen_width == 1920 and self.screen_height == 1080:
                        self.fig2.canvas.manager.window.setGeometry(660, 650, 740, 380)

                    self.fig2.subplots_adjust(left=0.086, right=0.981, top=0.898, bottom=0.159, hspace=0)
                    self.ax.plot(local_times.datetime, sunaltaz.alt, color="r", label="Sun")
                    self.ax.plot(local_times.datetime, moonaltaz.alt, color=[0.75] * 3, ls="--",
                                 label="Moon ({:.2f}%)".format(illumination * 100))

                    self.ax.scatter(
                        local_times.datetime,
                        objaltaz.alt,
                        c=objaltaz.az.value,
                        label=star_name,
                        lw=0,
                        s=8,
                        cmap="viridis",
                    )

                    self.ax.fill_between(
                        local_times.datetime,
                        0 * u.deg,
                        90 * u.deg,
                        sunaltaz.alt < -0 * u.deg,
                        color="k",
                        alpha=0.05,
                        zorder=0,
                    )
                    self.ax.fill_between(
                        local_times.datetime,
                        0 * u.deg,
                        90 * u.deg,
                        sunaltaz.alt < -6 * u.deg,
                        color="k",
                        alpha=0.15,
                        zorder=0,
                    )

                    self.ax.fill_between(
                        local_times.datetime,
                        0 * u.deg,
                        90 * u.deg,
                        sunaltaz.alt < -12 * u.deg,
                        color="k",
                        alpha=0.25,
                        zorder=0,
                    )

                    self.ax.fill_between(
                        local_times.datetime,
                        0 * u.deg,
                        90 * u.deg,
                        sunaltaz.alt < -18 * u.deg,
                        color="k",
                        alpha=0.35,
                        zorder=0,
                    )
                    self.ax.axvspan(start_time, end_time, color='yellow', alpha=0.3)
                    plt.colorbar(self.ax.collections[0], ax=self.ax, label="Azimuth [deg]")
                    plt.legend(loc="upper left")
                    plt.ylim(0 * u.deg, 90 * u.deg)
                    plt.xlabel("EEST")
                    plt.ylabel("Altitude [deg]")
                    plt.title("Visualization of Observation Duration Highlighted in Yellow")
                    self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

                    self.obs.setLayout(layout)

                    self.obs.show()
                    self.fig2.show()
                    self.fig0.show()

                except Exception as e:
                    print("Error displaying observation info:", e)
                    self.warning_label.setText("An error occurred while displaying observation info.")
        else:
            print("No .xls file found in the folder of the .dat file.")
            QMessageBox.warning(self, "Warning", "No .xls file found in the folder of the .dat file.")

    def select_minima_file(self):
        minima_file_path, _ = QFileDialog.getOpenFileName(self, "Select Minima Data File", "", "Data Files (*.txt)")

        if minima_file_path:
            self.minima_file_loaded = False
            self.minima_file_path = minima_file_path

            self.txt_window = QWidget()
            self.txt_window.setWindowTitle("Minima Data")
            self.txt_window.setWindowIcon(QIcon("images/star.ico"))

            with open(minima_file_path, 'r') as file:
                lines = file.readlines()

                if len(lines) == 0 or any(len(line.strip().split()) != 3 for line in lines):
                    self.warning_label.setText("<font color='red'>The minima file is not in the expected format</font>")
                else:
                    self.minima_file_loaded = True
                    if self.screen_width == 2560 and self.screen_height == 1440:
                        self.txt_window.setGeometry(1800, 550, 600, 200)
                    elif self.screen_width == 1920 and self.screen_height == 1080:
                        self.txt_window.setGeometry(1450, 440, 250, 200)

                    layout = QVBoxLayout()

                    lay1 = QVBoxLayout()
                    label1 = QLabel("Minima Time and Error [BJD-TDB]")
                    label1.setAlignment(Qt.AlignCenter)
                    txt_edit1 = QTextEdit()
                    txt_edit1.setReadOnly(True)
                    txt_edit1.setAlignment(Qt.AlignCenter)
                    txt_edit1.setFixedHeight(40)
                    copy_button1 = QPushButton("Copy")
                    copy_button1.clicked.connect(lambda: self.copy_text(txt_edit1.toPlainText()))
                    lay1.addWidget(label1)
                    lay1.addWidget(txt_edit1)
                    lay1.addWidget(copy_button1)

                    lay2 = QVBoxLayout()
                    label2 = QLabel("Minima Time in EEST")
                    label2.setAlignment(Qt.AlignCenter)
                    txt_edit2 = QTextEdit()
                    txt_edit2.setReadOnly(True)
                    txt_edit2.setAlignment(Qt.AlignCenter)
                    txt_edit2.setFixedHeight(40)
                    copy_button2 = QPushButton("Copy")
                    copy_button2.clicked.connect(lambda: self.copy_text(txt_edit2.toPlainText()))
                    lay2.addWidget(label2)
                    lay2.addWidget(txt_edit2)
                    lay2.addWidget(copy_button2)

                    lay3 = QVBoxLayout()
                    label3 = QLabel("Error in Seconds")
                    label3.setAlignment(Qt.AlignCenter)
                    txt_edit3 = QTextEdit()
                    txt_edit3.setReadOnly(True)
                    txt_edit3.setAlignment(Qt.AlignCenter)
                    txt_edit3.setFixedHeight(40)
                    copy_button3 = QPushButton("Copy")
                    copy_button3.clicked.connect(lambda: self.copy_text(txt_edit3.toPlainText()))
                    lay3.addWidget(label3)
                    lay3.addWidget(txt_edit3)
                    lay3.addWidget(copy_button3)

                    layout.addLayout(lay1)
                    layout.addLayout(lay2)
                    layout.addLayout(lay3)

                    on = get_star_name_and_telescope(minima_file_path)[0]

                    star_coord = None
                    try:
                        star_coord = SkyCoord.from_name(on)
                    except Exception as exc:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            f"Coordinates for {on} could not be resolved; times shown without barycentric correction.\n{exc}"
                        )

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            minima_time = float(parts[0])
                            minima_error = float(parts[2])

                            loc = EarthLocation(0, 0, 0, unit='m')
                            time = Time(minima_time, format='jd', scale='tdb', location=loc)
                            if star_coord is not None:
                                ltt_bary = time.light_travel_time(star_coord)
                                jd_tdb = time.tdb - ltt_bary
                                jd_utc = Time(jd_tdb, format='jd', scale='utc')
                                jd_utc += pd.Timedelta(hours=3)
                                dt = jd_utc.to_datetime()
                            else:
                                dt = time.tdb.to_datetime() + pd.Timedelta(hours=3)

                            txt_edit1.append(f"{minima_time:.6f} +/- {minima_error:.6f}")
                            txt_edit2.append(f"{dt.strftime('%H:%M')}")
                            txt_edit3.append(f"{minima_error * 86400:.2f}")

                    copy_button1 = QPushButton("Copy Minima Time and Error")
                    copy_button1.clicked.connect(lambda: self.copy_text(txt_edit1.toPlainText()))

                    copy_button2 = QPushButton("Copy Minima Time in EEST")
                    copy_button2.clicked.connect(lambda: self.copy_text(txt_edit2.toPlainText()))

                    copy_button3 = QPushButton("Copy Error in Seconds")
                    copy_button3.clicked.connect(lambda: self.copy_text(txt_edit3.toPlainText()))

                    self.txt_window.setLayout(layout)
                    self.txt_window.show()

            if not self.fig0:
                self.warning_label.setText("Data File is not selected.")
            else:
                if self.data_file_path:
                    minima_data_list = []

                    for line in lines:
                        minima_data = line.strip().split()
                        if len(minima_data) != 3:
                            self.warning_label.setText(
                                "<font color='red'>The minima file is not in the expected format</font>")
                        else:
                            minima_time = float(minima_data[0])
                            minima_type = int(minima_data[1])
                            minima_error = float(minima_data[2])
                            upper_limit = minima_time + minima_error
                            lower_limit = minima_time - minima_error
                            minima_data_list.append((minima_time, minima_type, minima_error, lower_limit, upper_limit))

                    for minima_time, minima_type, _, lower_limit, upper_limit in minima_data_list:
                        if self.fig0:
                            ax = self.fig0.gca()
                            if minima_type == 1:
                                ax.axvline(x=minima_time, color='black', linestyle='-', label='Min I')
                            elif minima_type == 2:
                                ax.axvline(x=minima_time, color='red', linestyle='-', label='Min II')

                        ax.axvline(x=lower_limit, color='grey', linestyle='--', lw=1)
                        ax.axvline(x=upper_limit, color='grey', linestyle='--', lw=1)

                        ax.set_xlabel('Time (BJD(TDB))')
                        ax.set_ylabel('Relative Flux')
                        ax.legend(loc="best")
                        self.warning_label.setText("<font color='green'>Press Enter to save the plot.</font>")
                        self.fig0.canvas.draw()
                        self.fig0.show()

                if not self.txt_window:
                    self.minima_file_loaded = False
        else:
            self.warning_label.setText("You should select a .txt file first.")

    def bjd_tdb_to_time_string(self, x, pos):
        jd = x
        try:
            if not hasattr(self, "ra_deg") or not hasattr(self, "dec_deg"):
                raise AttributeError("RA/Dec not available for barycentric correction.")

            star_coord = SkyCoord(ra=self.ra_deg, dec=self.dec_deg)
            loc = EarthLocation(0, 0, 0, unit='m')
            time = Time(jd, format='jd', scale='tdb', location=loc)
            ltt_bary = time.light_travel_time(star_coord)
            bjd_tdb = time.tdb + ltt_bary
            bjd_tdb += pd.Timedelta(hours=3)
            dt = bjd_tdb.to_datetime()
            return dt.strftime('%H:%M')
        except Exception:
            try:
                dt = Time(jd, format='jd', scale='tdb').to_datetime() + pd.Timedelta(hours=3)
                return dt.strftime('%H:%M')
            except Exception:
                return f"{jd:.5f}"

    def copy_text(self, text):
        QApplication.clipboard().setText(text)

    def open_tess_window(self):
        try:
            if self.tess_window is None or not self.tess_window.isVisible():
                star_name = self.star_name if self.star_name else ""
                self.tess_window = TESSWindow(
                    self.screen_width,
                    self.screen_height,
                    star_name=star_name,
                    sector_numbers=self.sector_numbers,
                    main_window=self,
                )
                self.tess_window.show()
            else:
                self.tess_window.raise_()
                self.tess_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while opening TESS window: {e}")

    def open_simbad_window(self):
        try:
            self.simbad_window = QWidget()
            self.simbad_window.setWindowTitle("SIMBAD Query")
            self.simbad_window.setWindowIcon(QIcon("images/star.ico"))

            self.simbad_star_name_input = QLineEdit()
            self.simbad_star_name_input.setPlaceholderText("Object Name")

            self.simbad_search_btn = QPushButton("Open SIMBAD Link")
            self.simbad_search_btn.clicked.connect(
                lambda: self.open_simbad_link(self.simbad_star_name_input.text().strip()))

            layout = QVBoxLayout()
            layout.addWidget(self.simbad_star_name_input)
            layout.addWidget(self.simbad_search_btn)

            self.simbad_window.setLayout(layout)
            self.simbad_window.show()

            if self.star_name:
                self.simbad_star_name_input.setText(self.star_name)

            if self.screen_width == 2560 and self.screen_height == 1440:
                self.simbad_window.setGeometry(1800, 550, 250, 70)
            elif self.screen_width == 1920 and self.screen_height == 1080:
                self.simbad_window.setGeometry(610, 500, 250, 70)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def open_simbad_link(self, star_name):
        if star_name:
            simbad_link = create_simbad_link(star_name)
            webbrowser.open(simbad_link)
        else:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")


def get_observation_date(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        date = datetime.strptime(date_str, '%Y%m%d')
        date_n = date.strftime('%d %B %Y')
        return date_n
    return None


def get_star_name_and_telescope(filename):
    global star_name
    star_name = None

    if os.path.isfile(filename):
        filename = os.path.basename(filename)

    match = re.search(r'_(.*?)_(T35|T80|T100|ATA50|IST60)', filename)
    if match:
        star = match.group(1)
        telescope = match.group(2)
    else:
        star = filename
        telescope = None

    if '_' in star:
        star_name = star.replace('_', ' ')
    else:
        if len(star) > 3 and star[-3:] in constellation_abbr:
            star_name = star[:-3] + " " + star[-3:]
        elif star.startswith("GAIA"):
            parts = re.split(r'(\d+)', star, 1)
            star_name = f"{parts[0]} {parts[1][:3]} {parts[1][3:]}{parts[2]}"
        elif star.startswith("FBS"):
            parts = re.split(r'(\d+)', star, 1)
            if parts[2] == "":
                star_name = f"{parts[0]} {parts[1][:-3]}+{parts[1][-3:]}{parts[2]}"
            else:
                star_name = f"{parts[0]} {parts[1][:-3]}{parts[1][-3:]}{parts[2]}"
        elif star.startswith("KPD"):
            parts = re.split(r'(\d+)', star, 1)
            if parts[2] == "":
                star_name = f"{parts[0]} {parts[1][:-4]}+{parts[1][-4:]}{parts[2]}"
            else:
                star_name = f"{parts[0]} {parts[1][:-4]}{parts[1][-4:]}{parts[2]}"
        elif star.startswith("HS"):
            parts = re.split(r'(\d+)', star, 1)
            star_name = f"{parts[0]} {parts[1]}{parts[2]}"
        elif star.startswith("ATO"):
            parts = re.split(r'(\d+)', star, 1)
            star_name = f"{parts[0][0:3]} {parts[0][-1]}{parts[1]}{parts[2]}"
        elif star.startswith("GALEX"):
            # GALEX'ten sonra bir boşluk ekle
            star_name = f"GALEX {star[5:]}"
        elif star.startswith("ZTF"):
            # ZTF'ten sonra boşluk ekle ve J ile koordinatları koru
            star_name = f"ZTF {star[3:]}"
        elif star.startswith(("TIC", "KIC", "CHSS", "LAMOST")):
            parts = re.split(r'(\d+)', star, 1)
            star_name = f"{parts[0]} {parts[1]}{parts[2]}"
        else:
            star_name = star


    return star_name, telescope


constellation_abbr = [
    "And", "Ant", "Aps", "Aql", "Aqr", "Ara", "Ari", "Aur", "Boo", "Cae", "Cam", "Cap", "Car", "Cas", "Cen", "Cep",
    "Cet", "Cha", "Cir", "CMa", "CMi", "Cnc", "Col", "Com", "CrA", "CrB", "Crt", "Cru", "Crv", "CVn", "Cyg", "Del",
    "Dor", "Dra", "Equ", "Eri", "For", "Gem", "Gru", "Her", "Hor", "Hya", "Hyi", "Ind", "Lac", "Leo", "Lep", "Lib",
    "LMi", "Lup", "Lyn", "Lyr", "Men", "Mic", "Mon", "Mus", "Nor", "Oct", "Oph", "Ori", "Pav", "Peg", "Per", "Phe",
    "Pic", "PsA", "Psc", "Pup", "Pyx", "Ret", "Scl", "Sco", "Sct", "Ser", "Sex", "Sge", "Sgr", "Tau", "Tel", "TrA",
    "Tri", "Tuc", "UMa", "UMi", "Vel", "Vir", "Vol", "Vul"
]


def get_filter(filename):
    if os.path.isfile(filename):
        filename = os.path.basename(filename)

    match = re.search(r'_(.*?)_(LC|lc)', filename)
    if match:
        prefix = match.group(1)
    else:
        # Gracefully handle unexpected filenames
        return "dodgerblue", "Unknown"

    filter_color = None
    filter_name = None

    if prefix.endswith('V'):
        filter_color = "green"
        filter_name = "V"
    elif prefix.endswith('g'):
        filter_color = "green"
        filter_name = "g"
    elif prefix.endswith('R'):
        filter_color = "red"
        filter_name = "R"
    elif prefix.endswith('r'):
        filter_color = "red"
        filter_name = "r"
    elif prefix.endswith('B'):
        filter_color = "blue"
        filter_name = "B"
    elif prefix.endswith('I'):
        filter_color = "saddlebrown"
        filter_name = "I"
    elif prefix.endswith('i'):
        filter_color = "brown"
        filter_name = "i"
    elif prefix.endswith('u'):
        filter_color = "purple"
        filter_name = "u"
    elif prefix.endswith('z'):
        filter_color = "gray"
        filter_name = "z"
    elif prefix.endswith('C'):
        filter_color = "black"
        filter_name = "Clear"

    if filter_color is None or filter_name is None:
        return "dodgerblue", "Unknown"

    return filter_color, filter_name



def cadence_label(exptimes) -> str:
    """Light-curve products of one sector as shown to the user."""
    if exptimes is None:
        return "?"
    return ", ".join({120: "2-min", 20: "20-s"}[e] for e in exptimes) or "FFI only"


def html_tess_sector_message(sectors_raw, mapping: dict[int, tuple] | None, cadences: dict) -> str:
    """
    Generate an HTML snippet displaying TESS sectors with mid-point, camera, CCD and edge warning.
    - sectors_raw: iterable of sector numbers or iterables that contain sector numbers.
    - mapping: dict mapping sector number to (mid-point datetime, camera, CCD, near CCD edge).
    """
    # Ensure mapping is a dict
    mapping = mapping or {}

    # Flatten and parse sector values
    flat = []
    for item in sectors_raw:
        if isinstance(item, (list, tuple, set)):
            flat.extend(int(x) for x in item)
        else:
            flat.append(int(item))
    sectors = sorted(set(flat))

    # No data case
    if not sectors:
        return (
            "<div style='font-family:Helvetica, Arial, sans-serif;"
            " text-align:center; color:#d32f2f;'>"
            "<strong>No TESS data found.</strong>"
            "</div>"
        )

    # Build table rows with centered cells
    rows = []
    for idx, sec in enumerate(sectors):
        bg = "#fafafa" if idx % 2 else "white"
        if sec in mapping:
            mid_dt, cam, ccd, near_edge = mapping[sec]
            cells = [cadence_label(cadences.get(sec)), mid_dt.strftime("%Y-%m-%d"), cam, ccd,
                     "<span style='color:#e65100;'>near edge</span>" if near_edge else ""]
        else:
            cells = [cadence_label(cadences.get(sec)), "—", "—", "—", ""]
        rows.append(
            f"<tr style='background:{bg};'>"
            f"<td style='padding:6px 12px; text-align:center;'>S{sec}</td>"
            + "".join(f"<td style='padding:6px 12px; text-align:center;'>{c}</td>" for c in cells)
            + "</tr>"
        )

    # Final HTML
    return (
        "<div style='font-family:Helvetica, Arial, sans-serif;"
        " text-align:center; line-height:1.4; color:#333;'>"
        "<div style='font-size:20px; font-weight:600; margin-bottom:4px; color:#2e7d32;'>"
        "TESS data found!""</div>"
        f"<div style='font-size:14px; margin-bottom:12px;'>"
        f"Total sectors: <strong>{len(sectors)}</strong>""</div>"
        "<table style='border-collapse:collapse; margin:12px auto;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:8px 16px; border-bottom:1px solid #ccc; text-align:center;'>Sector</th>"
        + "".join(
            f"<th style='padding:8px 16px; border-bottom:1px solid #ccc; text-align:center;'>{h}</th>"
            for h in ["Data", "Mid-point", "Camera", "CCD", "CCD edge"]
        ) +
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows) +
        "</tbody>"
        "</table>"
        "</div>"
    )

_SECTOR_MID_CACHE = None

def sector_midpoints():
    global _SECTOR_MID_CACHE
    if _SECTOR_MID_CACHE is None:
        url = "https://tess.mit.edu/wp-content/uploads/pointing/sector_times.csv"
        df = pd.read_csv(url, comment="#")
        df["mid"] = (pd.to_datetime(df["TSTART"], utc=True) +
                     (pd.to_datetime(df["TSTOP"], utc=True) -
                      pd.to_datetime(df["TSTART"], utc=True))/2)
        _SECTOR_MID_CACHE = dict(zip(df["SECTOR"].astype(int), df["mid"]))
    return _SECTOR_MID_CACHE

class TESSWindow(TessQueryMixin, OutlierMixin, DetrendingMixin, PhaseMixin, QWidget):
    def __init__(self, screen_width, screen_height, star_name=None, sector_numbers=None, main_window=None):
        super().__init__()
        self.ctrl = TESSController()
        self.lc = self.ctrl.state  # convenience alias for property accessors
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.star_name = star_name
        self.sector_numbers = sector_numbers
        self.sector_cadences = {}
        self.main_window = main_window
        self.canvas = None
        self.canvas_outliers = None
        self.toolbar = None
        self.selected_start_time = None
        self.selected_end_time = None
        self.canvas_detrending = None
        self.automated_segments = []
        self.outlier_segments = []
        self.detrend_segments = []

        self.selector = None
        self.region_patch = None
        self._fixed_xlim = None
        self._fixed_ylim = None

        self.manual_patches = []
        self.detrending_selector = None
        self.detrending_patches = []

        self.outlier_patches = []
        self.segment_mode = "automate"

        self.toast_label = None
        self.toast_timer = None

        self.init_ui()
        self._download_dir = load_download_dir()

        # Inline status updates (tray balloon disabled per preference)
        self.tray_icon = None

    def _build_download_path_controls(self):
        layout = QHBoxLayout()

        self.default_download_checkbox = QCheckBox("Use mastDownload default")
        self.default_download_checkbox.stateChanged.connect(self._toggle_download_path_mode)

        self.download_path_edit = QLineEdit()
        self.download_path_edit.setReadOnly(True)
        self.download_path_edit.setPlaceholderText("No custom folder selected")

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._select_download_dir)

        layout.addWidget(self.default_download_checkbox)
        layout.addWidget(self.download_path_edit)
        layout.addWidget(browse_btn)

        # Initialize state from config
        saved_dir = load_download_dir()
        if saved_dir and saved_dir.exists():
            self._download_dir = saved_dir
            self.default_download_checkbox.setChecked(False)
            self.download_path_edit.setText(str(saved_dir))
            self.download_path_edit.setEnabled(True)
        else:
            self._download_dir = None
            self.default_download_checkbox.setChecked(True)
            self.download_path_edit.clear()
            self.download_path_edit.setEnabled(False)

        return layout

    def _toggle_download_path_mode(self, state):
        use_default = state == Qt.Checked
        if use_default:
            self._download_dir = None
            self.download_path_edit.clear()
            self.download_path_edit.setEnabled(False)
            try:
                save_download_dir(None)  # remove user override
            except Exception:
                pass
        else:
            self.download_path_edit.setEnabled(True)
            if not self.download_path_edit.text():
                self._select_download_dir()

    def _select_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select download folder")
        if folder:
            path = Path(folder)
            if not path.exists():
                QMessageBox.warning(self, "Invalid Folder", "Selected folder does not exist.")
                return
            self._download_dir = path
            self.download_path_edit.setText(str(path))
            self.default_download_checkbox.setChecked(False)
            try:
                save_download_dir(str(path))
            except Exception as exc:
                QMessageBox.warning(self, "Warning", f"Could not save preference: {exc}")

    # --- Light-curve state accessors (keeps state centralized) ---
    @property
    def time(self):
        return self.lc.current_time

    @time.setter
    def time(self, value):
        self.lc.current_time = value

    @property
    def flux(self):
        return self.lc.current_flux

    @flux.setter
    def flux(self, value):
        self.lc.current_flux = value

    @property
    def flux_err(self):
        return self.lc.current_flux_err

    @flux_err.setter
    def flux_err(self, value):
        self.lc.current_flux_err = value

    @property
    def original_time(self):
        return self.lc.original_time

    @original_time.setter
    def original_time(self, value):
        self.lc.original_time = value

    @property
    def original_flux(self):
        return self.lc.original_flux

    @original_flux.setter
    def original_flux(self, value):
        self.lc.original_flux = value

    @property
    def original_flux_err(self):
        return self.lc.original_flux_err

    @original_flux_err.setter
    def original_flux_err(self, value):
        self.lc.original_flux_err = value

    @property
    def cleaned_time(self):
        return self.lc.cleaned_time

    @cleaned_time.setter
    def cleaned_time(self, value):
        self.lc.cleaned_time = value

    @property
    def cleaned_flux(self):
        return self.lc.cleaned_flux

    @cleaned_flux.setter
    def cleaned_flux(self, value):
        self.lc.cleaned_flux = value

    @property
    def cleaned_flux_err(self):
        return self.lc.cleaned_flux_err

    @cleaned_flux_err.setter
    def cleaned_flux_err(self, value):
        self.lc.cleaned_flux_err = value

    @property
    def pre_detrend_time(self):
        return self.lc.pre_detrend_time

    @pre_detrend_time.setter
    def pre_detrend_time(self, value):
        self.lc.pre_detrend_time = value

    @property
    def pre_detrend_flux(self):
        return self.lc.pre_detrend_flux

    @pre_detrend_flux.setter
    def pre_detrend_flux(self, value):
        self.lc.pre_detrend_flux = value

    @property
    def pre_detrend_flux_err(self):
        return self.lc.pre_detrend_flux_err

    @pre_detrend_flux_err.setter
    def pre_detrend_flux_err(self, value):
        self.lc.pre_detrend_flux_err = value

    def init_ui(self):
        try:
            self.setWindowTitle("TESS Query")
            self.setWindowIcon(QIcon("images/star.ico"))

            tabs = QTabWidget()

            # ---- Query Tab ----
            query_tab = self._init_query_tab()

            # ---- Outlier Tab ----
            outlier_tab = self._init_outlier_tab()

            # ---- Detrending Tab ----
            detrending_tab = self._init_detrending_tab()

            # ---- Phase Fold Tab ----
            phase_tab = self._init_phase_tab()


            tabs.addTab(query_tab, "Query")
            tabs.addTab(outlier_tab, "Outlier")
            tabs.addTab(detrending_tab, "Detrending")
            tabs.addTab(phase_tab, "Phase Fold")
            tabs.setContentsMargins(2, 2, 2, 2)

            main_layout = QVBoxLayout()
            self.status_label = QLabel()
            self.status_label.setStyleSheet("color: green;")
            self.status_label.hide()
            main_layout.setContentsMargins(6, 6, 6, 6)
            main_layout.addWidget(self.status_label)
            main_layout.addWidget(tabs)
            self.setLayout(main_layout)

            if self.star_name:
                self.star_name_input.setText(self.star_name)
            if self.sector_numbers:
                self.sector_combo.addItems(map(str, self.sector_numbers))
                self.sector_combo.setCurrentIndex(len(self.sector_numbers) - 1)
            self.sector_combo.currentTextChanged.connect(self._update_exptimes)

            self.setGeometry(300, 200, 1000, 800)  # Adjusted size


        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def _on_sector_search_finished(self, msg, sector_numbers, mapping, cadences):
        star = self.star_name_input.text().strip() or self.star_name
        if not sector_numbers:
            msg = (
                f"<div style='font-family:Helvetica, Arial, sans-serif; "
                f"color:#d32f2f; font-weight:bold; font-size:14px;'>"
                f"Not found!</div>"
                f"<div style='font-family:Helvetica, Arial, sans-serif; "
                f"color:#333; margin-top:4px;'>"
                f"{msg}</div>"
            )
            QMessageBox(self, windowTitle="Not Found", textFormat=Qt.RichText, text=msg).exec_()
            return
        html = html_tess_sector_message(sector_numbers, mapping, cadences)

        QMessageBox(
            self,
            windowTitle="TESS Search Result",
            textFormat=Qt.RichText,
            text=html
        ).exec_()

        if sector_numbers:
            self._remember_star_name(star)
            self.sector_numbers = sector_numbers
            self.sector_cadences = cadences
            self.sector_combo.clear()
            self.sector_combo.addItems(f"{s} (FFI)" if cadences.get(s) == [] else str(s) for s in sector_numbers)
            with_lc = [s for s in sector_numbers if cadences.get(s)]
            self.sector_combo.setCurrentIndex(sector_numbers.index((with_lc or sector_numbers)[-1]))
            if with_lc:
                self.prefetch_thread = TESSPrefetchThread(star, with_lc[-1], cadences[with_lc[-1]][0])
                self.prefetch_thread.start()

    def _update_exptimes(self, sector_text):
        """Offer only the exposure times that have a light curve in the selected sector."""
        sector = sector_text.split(" ")[0]
        exptimes = self.sector_cadences.get(int(sector)) if sector.isdigit() else None
        self.exptime_combo.clear()
        self.exptime_combo.addItems([str(e) for e in exptimes or (120, 20)])

    def _on_sector_search_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_future_sectors(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return

        self.future_btn.setEnabled(False)
        self.future_btn.setText("Loading…")
        self.future_btn.setStyleSheet("color:red;")

        self.future_thread = TESSFutureThread(star_name)
        self.future_thread.resultReady.connect(self._on_future_ok)
        self.future_thread.error.connect(lambda m: QMessageBox.critical(self, "Error", m))

        # 3) On finish, restore
        def restore():
            self.future_btn.setEnabled(True)
            self.future_btn.setText("Upcoming Sectors")
            self.future_btn.setStyleSheet("")

        self.future_thread.finished.connect(restore)

        self.future_thread.start()

    def _on_future_ok(self, rows: list[tuple[int, str]]):
        if not rows:
            self._info("No upcoming TESS observations scheduled for this target.")
            return

        dlg = QDialog(self);
        dlg.setWindowTitle("Upcoming TESS Sectors")
        dlg.setWindowIcon(QIcon("images/star.ico"))
        outer = QVBoxLayout(dlg)

        model = QStandardItemModel(len(rows), 2)
        model.setHorizontalHeaderLabels(["Sector", "Sector Midpoint"])
        for r, (sec, mid) in enumerate(rows):
            model.setItem(r, 0, QStandardItem(str(sec)))
            model.setItem(r, 1, QStandardItem(mid))
        table = QTableView();
        table.setModel(model)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableView.NoEditTriggers)

        outer.addWidget(table)
        dlg.resize(300, 200)
        dlg.exec_()

    def search_tess_sectors(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return

        # UI’yi “meşgul” göster
        self.tess_search_btn.setEnabled(False)
        self.tess_search_btn.setText("Searching…")
        self.tess_search_btn.setStyleSheet("color:red;")

        # Thread’i başlat
        self.sector_thread = TESSSectorSearchThread(star_name)
        self.sector_thread.resultReady.connect(self._on_sector_search_finished)
        self.sector_thread.error.connect(self._on_sector_search_error)
        self.sector_thread.finished.connect(
            lambda: (self.tess_search_btn.setEnabled(True),
                     self.tess_search_btn.setText("Search Sectors"),
                     self.tess_search_btn.setStyleSheet("color:black;"))
        )
        self.sector_thread.start()

    def update_query_plot(self):
        """Update the query plot based on the checkbox state."""
        self.plot_data_check(self.figure_query, self.canvas, self.query_error_checkbox.isChecked())

    def plot_data_check(self, figure, canvas, display_errors):
        """Plot data on the given figure and canvas with optional error bars."""
        figure.clear()
        ax = figure.add_subplot(111)

        # veri varsa çiz
        if self.time is not None and self.flux is not None:
            if display_errors and self.flux_err is not None:
                ax.errorbar(
                    self.time, self.flux, yerr=self.flux_err,
                    fmt=".", markersize=5, capsize=1, elinewidth=1.5,
                    ls="none", color="dodgerblue"
                )
            else:
                ax.plot(self.time, self.flux, ".", markersize=5, color="dodgerblue")

        ax.set_xlabel("Time – 2457000 ($day$)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")

        flux_label = getattr(self, "flux_type", "Light Curve").replace("_", " ")
        sector = self.sector_combo.currentText().split(" ")[0]
        ax.set_title(f"{flux_label} Light Curve – Sector {sector}")
        canvas.draw()

    def save_query_data(self):
        """Save the query data to a file in the chosen file format."""
        try:
            if self.time is None or self.flux is None or self.flux_err is None:
                QMessageBox.warning(self, "Warning", "No query data available to save.")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Save Query Data Options")

            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format to save:"))

            file_format_combo = QComboBox()
            file_format_combo.addItems(["XLSX", "TXT", "CSV"])
            layout.addWidget(file_format_combo)

            add_2457000_checkbox = QCheckBox("Add 2457000 to Time")
            add_2457000_checkbox.setChecked(True)  # Checked by default
            layout.addWidget(add_2457000_checkbox)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            file_format = file_format_combo.currentText().lower()  # Convert to lowercase for consistency

            time_data = self.time + 2457000 if add_2457000_checkbox.isChecked() else self.time
            data = pd.DataFrame({
                "Time": time_data,
                "Flux": self.flux,
                "Error": self.flux_err
            })

            file_name, _ = QFileDialog.getSaveFileName(
                self, f"Save Query Data as {file_format.upper()}", "", f"{file_format.upper()} Files (*.{file_format})"
            )
            if not file_name:
                return

            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"

            try:
                if file_format == "csv":
                    data.to_csv(file_name, index=False, float_format="%.10f")
                elif file_format == "xlsx":
                    with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                        data.to_excel(writer, sheet_name="Query Data", index=False, float_format="%.10f")
                elif file_format == "txt":
                    np.savetxt(
                        file_name,
                        data.values,
                        fmt="%.10f",
                        header="\t".join(data.columns),
                        delimiter="\t",
                        comments=""
                    )

                self._info(f"Query data saved successfully to {file_name}.")

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"An error occurred while saving query data as {file_format.upper()}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")


    def detach_query_plot(self):
        """Detach the query plot and display it in a separate window."""
        if self.time is None or self.flux is None or self.flux_err is None:
            QMessageBox.warning(self, "Warning", "No data available for plotting.")
            return

        fig, ax = plt.subplots()

        ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt='.', color='dodgerblue', markersize=5, capsize=1,
                    elinewidth=1.5, ls='none')
        ax.set_ylabel("Flux ($electrons^{-1}$)")
        ax.set_xlabel("Time - 2457000 ($day$)")
        ax.set_title("Detached Light Curve - Query Data")

        fig.tight_layout()
        fig.show()

    def download_and_plot_data(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        sector_txt = self.sector_combo.currentText().split(" ")[0]
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return
        if not sector_txt.isdigit():
            QMessageBox.warning(self, "Warning", "Sector number must be an integer.")
            return

        sector = int(sector_txt)
        exptime = int(self.exptime_combo.currentText())

        # flux tipini underscore’dan boşluğa çevirip saklıyoruz
        flux_type = self.flux_type_combo.currentText().replace(" ", "_")
        self.flux_type = flux_type

        apply_filter = self.quality_nan_checkbox.isChecked()

        # persist chosen download dir for worker threads
        if self._download_dir is not None:
            save_download_dir(str(self._download_dir))

        # butonu “meşgul” hâle getir
        self.download_data_btn.setEnabled(False)
        self.download_data_btn.setText("Downloading and plotting…")
        self.download_data_btn.setStyleSheet("color:red;")

        # indirme thread’ini başlat
        self.tess_thread = TESSDownloadThread(
            star_name, sector, exptime, flux_type, apply_filter
        )
        self.tess_thread.resultReady.connect(self._on_tess_finished)
        self.tess_thread.error.connect(self._on_tess_error)
        # bittiğinde butonu eski haline getir
        self.tess_thread.finished.connect(
            lambda: (
                self.download_data_btn.setEnabled(True),
                self.download_data_btn.setText("Download and Plot Data"),
                self.download_data_btn.setStyleSheet("color:black;")
            )
        )
        self.tess_thread.start()
    def _on_tess_finished(self, time, flux, flux_err, quality):
        self._reset_tess_state()
        self.ctrl.load_new_data(time, flux, flux_err)

        self.update_query_plot()
        self.plot_data_in_outliers()
        self.plot_data_in_detrending()

    def _on_tess_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def _info(self, text, duration=4000):
        """Show message on main window if available; otherwise in-app toast/label."""
        if getattr(self, "main_window", None):
            try:
                self.main_window.show_info_message(text, duration)
                return
            except Exception:
                pass
        shown = self._show_toast(text, duration)
        if not shown:
            self.status_label.show()
            show_info(self.status_label, text, duration_ms=duration)

    def _show_toast(self, text, duration_ms):
        try:
            if self.toast_label is None:
                self.toast_label = QLabel(self)
                self.toast_label.setStyleSheet(
                    "background-color: rgba(50, 50, 50, 200);"
                    "color: white;"
                    "padding: 8px 12px;"
                    "border-radius: 6px;"
                    "font-size: 12px;"
                )
                self.toast_label.setAttribute(Qt.WA_TranslucentBackground, True)
                self.toast_label.setWindowFlag(Qt.FramelessWindowHint, True)
                self.toast_label.setWindowFlag(Qt.Tool, True)
                self.toast_label.setWordWrap(True)
                self.toast_timer = QTimer(self)
                self.toast_timer.setSingleShot(True)
                self.toast_timer.timeout.connect(lambda: self.toast_label.hide())

            self.toast_label.setText(text)
            self.toast_label.adjustSize()

            margin = 12
            x = (self.width() - self.toast_label.width()) // 2
            y = self.height() - self.toast_label.height() - margin
            self.toast_label.move(max(x, 0), max(y, 0))
            self.toast_label.show()
            if self.toast_timer:
                self.toast_timer.start(duration_ms)
            return True
        except Exception:
            return False

    def _reset_tess_state(self):
        """Clear selections, patches, and cached data before loading a new dataset."""
        # Reset data variants
        self.ctrl.reset_state()

        # Clear segment bookkeeping
        self.outlier_segments.clear()
        self.detrend_segments.clear()
        self.automated_segments.clear()

        # Clear Outlier overlays
        self.clear_outlier_overlays()

        # Clear detrending overlays
        for attr in ("detrending_patches",):
            patches = getattr(self, attr, [])
            for p in list(patches):
                try:
                    p.remove()
                except ValueError:
                    pass
            patches.clear()

        if self.detrending_selector is not None:
            try:
                self.detrending_selector.set_active(False)
                self.detrending_selector.disconnect_events()
            except Exception:
                pass
            self.detrending_selector = None

    def display_tess_sector_data(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.");
            return

        self.display_data_btn.setEnabled(False)
        self.display_data_btn.setText("Querying…");
        self.display_data_btn.setStyleSheet("color:red;")

        self.data_thread = TESSDataFetchThread(star_name)
        self.data_thread.resultReady.connect(self._show_sector_window)
        self.data_thread.error.connect(lambda m: QMessageBox.critical(self, "Error", m))
        self.data_thread.finished.connect(lambda: (
            self.display_data_btn.setEnabled(True),
            self.display_data_btn.setText("Query More"),
            self.display_data_btn.setStyleSheet("color:black;")
        ))
        self.data_thread.start()

    def _show_sector_window(self,
                            lc_data: list[tuple[int, int]],
                            ffi_data: list[tuple[int, str, int]]):
        """Show Timeseries & FFI sector info side‑by‑side in a tidy dialog."""

        # ── Title / basic dialog -------------------------------------------------
        star_name = self.star_name_input.text().strip() or self.star_name or "—"
        dlg = QDialog(self)
        dlg.setWindowTitle("TESS Sector Data")
        dlg.setWindowIcon(QIcon("images/star.ico"))
        outer = QVBoxLayout(dlg)

        title = QLabel(
            f"<b>{star_name}</b> &nbsp;—&nbsp; <b><font color='red'>{len({row[0] for row in lc_data})}</font></b> sector(s) found"
        )
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        # ── Shared quick‑filter --------------------------------------------------
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Quick filter… (sector no. / exposure)")
        outer.addWidget(filter_edit)

        # ── Helper to build a styled, filterable table --------------------------
        def make_table(data, headers, tint):
            model = QStandardItemModel(len(data), len(headers))
            model.setHorizontalHeaderLabels(headers)

            for r, row in enumerate(data):
                for c, value in enumerate(row):
                    item = QStandardItem(str(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    if c == 0:  # sector sütunu renklensin
                        item.setBackground(QColor(tint).lighter(185))
                    model.setItem(r, c, item)

            # ------------- DÜZELTME -------------
            proxy = QSortFilterProxyModel()  # parent yok
            proxy.setSourceModel(model)  # modeli burada bağla
            # ------------------------------------

            table = QTableView()
            table.setModel(proxy)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setSelectionBehavior(QTableView.SelectRows)
            table.setAlternatingRowColors(True)  # hoş bir zebra efekti
            table.setSortingEnabled(True)
            table.sortByColumn(0, Qt.AscendingOrder)
            table.setMinimumHeight(180)
            return table, proxy
        # Build both tables
        ts_table, ts_proxy = make_table(lc_data, ["Sector", "Exp. (s)"], "#BBDEFB")
        ffi_table, ffi_proxy = make_table([(s, expt) for s, _, expt in ffi_data], ["Sector", "Exp. (s)"], "#C8E6C9")

        # ── GroupBoxes to give each table a header ------------------------------
        ts_box = QGroupBox("Timeseries Data")
        ts_box.setLayout(QVBoxLayout())
        ts_box.layout().addWidget(ts_table)

        ffi_box = QGroupBox("FFI Data")
        ffi_box.setLayout(QVBoxLayout())
        ffi_box.layout().addWidget(ffi_table)

        # ── Place side‑by‑side ---------------------------------------------------
        hbox = QHBoxLayout()
        hbox.addWidget(ts_box)

        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(vline)

        hbox.addWidget(ffi_box)
        outer.addLayout(hbox)

        # ── Wire‑up filter field -------------------------------------------------
        filter_edit.textChanged.connect(ts_proxy.setFilterFixedString)
        filter_edit.textChanged.connect(ffi_proxy.setFilterFixedString)



        dlg.resize(700, 500)  # wider because tables sit horizontally
        dlg.exec_()


LOG_FILE = Path.home() / ".lcm" / "lcm.log"


def report_crash(kind, error, trace):
    """Writes an unhandled error to the log file and shows it, since a shortcut start has no console."""
    sys.__excepthook__(kind, error, trace)
    LOG_FILE.parent.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOG_FILE, format="%(asctime)s %(message)s")
    logging.error("Unhandled error", exc_info=(kind, error, trace))
    QMessageBox.critical(None, "LCM error", f"{error}\n\nDetails were saved to {LOG_FILE}")


def main():
    os.chdir(Path(__file__).resolve().parent)  # icons and data files are opened by relative path
    sys.excepthook = report_crash
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
