import os
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog, QLabel,
    QLineEdit, QComboBox, QPushButton, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from kepler import KeplerOutlierTab, KeplerDetrendingTab, KeplerPhaseFoldTab


class CustomDataWindow(QWidget):
    """Standalone window for importing arbitrary light curves and reusing Kepler tools."""

    def __init__(self, screen_width, screen_height):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.df = None
        self.time = None
        self.flux = None
        self.flux_err = None
        self._is_mag = False
        self._flux_label = "Flux / Brightness"

        self.shared_time = None
        self.shared_flux = None
        self.shared_flux_err = None

        self.init_ui()
        self.adjust_geometry()

    def init_ui(self):
        self.setWindowTitle("Imported Light Curve")
        self.setWindowIcon(QIcon("images/star.ico"))

        self.tabs = QTabWidget()

        # --- Import Tab ---
        import_tab = QWidget()
        import_layout = QVBoxLayout()
        import_layout.setContentsMargins(10, 10, 10, 6)
        import_layout.setSpacing(6)

        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("Select CSV/TXT/Excel file…")
        path_layout.addWidget(self.file_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        path_layout.addWidget(browse_btn)
        import_layout.addLayout(path_layout)

        combo_layout = QHBoxLayout()
        combo_layout.setSpacing(8)
        self.time_combo = QComboBox()
        self.flux_combo = QComboBox()
        self.error_combo = QComboBox()
        self._setup_combo(self.time_combo, "Time column")
        self._setup_combo(self.flux_combo, "Flux/Brightness column")
        self._setup_combo(self.error_combo, "Flux error column")
        self.error_combo.addItem("Use unity errors")
        combo_layout.addWidget(self.time_combo)
        combo_layout.addWidget(self.flux_combo)
        combo_layout.addWidget(self.error_combo)
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Auto detect", "Flux", "Magnitude"])
        combo_layout.addWidget(self.data_type_combo)
        import_layout.addLayout(combo_layout)

        load_btn = QPushButton("Load Selected Columns")
        load_btn.clicked.connect(self.apply_column_selection)
        import_layout.addWidget(load_btn)

        self.status_label = QLabel("Select file, choose columns, then load.")
        self.status_label.setStyleSheet("color:#2c3e50; font-size:11px;")
        import_layout.addWidget(self.status_label)

        self.import_figure = Figure()
        self.import_canvas = FigureCanvas(self.import_figure)
        self.import_toolbar = NavigationToolbar(self.import_canvas, self)
        self.import_error_checkbox = QCheckBox("Display with Errors")
        self.import_error_checkbox.setChecked(True)
        self.import_error_checkbox.stateChanged.connect(self.plot_loaded_data)

        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(4)
        plot_layout.addWidget(self.import_canvas)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.import_toolbar)
        toolbar_layout.addWidget(self.import_error_checkbox)
        toolbar_layout.setAlignment(Qt.AlignLeft)
        plot_layout.addLayout(toolbar_layout)
        import_layout.addLayout(plot_layout)

        import_tab.setLayout(import_layout)
        self.tabs.addTab(import_tab, "Import")

        # --- Analysis Tabs (reuse Kepler tools) ---
        self.outlier_tab = KeplerOutlierTab()
        self.detrending_tab = KeplerDetrendingTab()
        self.phase_fold_tab = KeplerPhaseFoldTab()

        self.tabs.addTab(self.outlier_tab, "Outlier")
        self.tabs.addTab(self.detrending_tab, "Detrending")
        self.tabs.addTab(self.phase_fold_tab, "Phase Fold")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def adjust_geometry(self):
        if self.screen_width == 2560 and self.screen_height == 1440:
            self.setGeometry(400, 300, 1000, 800)
        elif self.screen_width == 1920 and self.screen_height == 1080:
            self.setGeometry(300, 200, 1000, 800)
        else:
            self.setGeometry(250, 150, 1000, 800)
        self.setMinimumSize(900, 700)

    def _setup_combo(self, combo: QComboBox, placeholder: str):
        combo.clear()
        combo.addItem(placeholder)
        combo.model().item(0).setEnabled(False)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Light Curve File",
            "",
            "Data Files (*.csv *.txt *.dat *.tsv *.xls *.xlsx);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_dataframe(file_path)

    def load_dataframe(self, file_path: str):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, sep=None, engine="python")

            if df.empty:
                raise ValueError("Loaded file is empty.")

            self.df = df
            self._setup_combo(self.time_combo, "Time column")
            self._setup_combo(self.flux_combo, "Flux/Brightness column")
            self._setup_combo(self.error_combo, "Flux error column")
            self.error_combo.addItem("Use unity errors")
            for col in df.columns:
                col_name = str(col)
                self.time_combo.addItem(col_name)
                self.flux_combo.addItem(col_name)
                self.error_combo.addItem(col_name)

            self.status_label.setText("Select the columns and click 'Load Columns' to plot the data.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to read file: {e}")

    def _coerce_numeric(self, series):
        return pd.to_numeric(series, errors="coerce").to_numpy()

    def apply_column_selection(self):
        try:
            if self.df is None:
                QMessageBox.warning(self, "Warning", "Please load a file first.")
                return

            time_col = self.time_combo.currentText()
            flux_col = self.flux_combo.currentText()
            err_col = self.error_combo.currentText()

            if "column" in time_col.lower() or "column" in flux_col.lower():
                QMessageBox.warning(self, "Warning", "Please choose valid columns for time and flux.")
                return

            time_arr = self._coerce_numeric(self.df[time_col])
            flux_arr = self._coerce_numeric(self.df[flux_col])

            if err_col == "Use unity errors":
                err_arr = np.ones_like(flux_arr, dtype=float)
            else:
                err_arr = self._coerce_numeric(self.df[err_col])

            mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & np.isfinite(err_arr)
            time_arr = time_arr[mask]
            flux_arr = flux_arr[mask]
            err_arr = err_arr[mask]

            if time_arr.size == 0:
                QMessageBox.warning(self, "Warning", "No valid rows after filtering NaN/inf values.")
                return

            if not np.all(err_arr > 0):
                finite_err = err_arr[np.isfinite(err_arr) & (err_arr > 0)]
                fallback = np.median(finite_err) if finite_err.size else 1.0
                err_arr = np.where(err_arr > 0, err_arr, fallback)

            flux_name = flux_col.lower()
            col_is_mag = "mag" in flux_name
            col_is_flux = "flux" in flux_name
            choice = self.data_type_combo.currentText()

            if col_is_mag:
                self._is_mag = True
            elif col_is_flux:
                self._is_mag = False
            else:
                self._is_mag = (choice == "Magnitude")

            self.time = time_arr
            self.flux = flux_arr
            self.flux_err = err_arr
            self._flux_label = "Magnitude" if self._is_mag else "Flux / Brightness"

            self.updateSharedData(self.time, self.flux, self.flux_err)
            self.status_label.setText(f"Loaded {self.time.size} points.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load selected columns: {e}")

    def plot_loaded_data(self):
        ax = self.import_figure.gca()
        ax.clear()
        if self.time is not None and self.flux is not None and len(self.time) > 0:
            if self.import_error_checkbox.isChecked() and self.flux_err is not None:
                ax.errorbar(
                    self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                    markersize=5, capsize=1, elinewidth=1.5, ls="none"
                )
            else:
                ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)
            ax.set_xlabel("Time")
            ax.set_ylabel(self._flux_label)
            ax.set_title("Imported Light Curve")
            if self._is_mag:
                ax.invert_yaxis()
        self.import_figure.tight_layout()
        self.import_canvas.draw()

    def updateSharedData(self, time, flux, flux_err):
        self.shared_time = time
        self.shared_flux = flux
        self.shared_flux_err = flux_err

        if hasattr(self.outlier_tab, "_invert_y"):
            self.outlier_tab._invert_y = self._is_mag
        if hasattr(self.detrending_tab, "_invert_y"):
            self.detrending_tab._invert_y = self._is_mag
        if hasattr(self.phase_fold_tab, "_invert_y"):
            self.phase_fold_tab._invert_y = self._is_mag
        # Propagate label preference so manual/auto trend plots stay consistent
        try:
            self.outlier_tab._flux_label = self._flux_label
            self.detrending_tab._flux_label = self._flux_label
            self.phase_fold_tab._flux_label = self._flux_label
        except Exception:
            pass

        self.outlier_tab.setData(time, flux, flux_err)
        self.detrending_tab.setData(time, flux, flux_err)
        self.phase_fold_tab.setData(time, flux, flux_err)
        self._apply_axis_style(self.outlier_tab.figure_outliers, self.outlier_tab.canvas_outliers)
        self._apply_axis_style(self.detrending_tab.figure_detrending, self.detrending_tab.canvas_detrending)
        self._apply_axis_style(self.phase_fold_tab.figure_phase, self.phase_fold_tab.canvas_phase)
        self.plot_loaded_data()

    def _apply_axis_style(self, figure, canvas):
        """Set y-label and inversion based on magnitude/flux selection."""
        try:
            ax = figure.gca()
            ax.set_ylabel(self._flux_label)
            ymin, ymax = ax.get_ylim()
            if self._is_mag:
                if ymin < ymax:
                    ax.set_ylim(ymax, ymin)
            else:
                if ymin > ymax:
                    ax.set_ylim(ymax, ymin)
            canvas.draw_idle()
        except Exception:
            pass
