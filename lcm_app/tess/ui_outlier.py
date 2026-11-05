import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QComboBox,
    QCheckBox, QSpinBox, QLabel, QSizePolicy, QDialog, QDialogButtonBox, QFileDialog,
    QMessageBox
)
import numpy as np
import pandas as pd
from scipy.special import erfcinv


class OutlierMixin:
    """
    Outlier tab UI builder.
    Expects inheriting class to provide:
      - screen_width, screen_height
      - handler methods: enable_region_selection, remove_selected_interval,
        apply_outlier_detection, remove_outliers, apply_normalization,
        plot_original_data, detach_plot, save_cleaned_data,
        plot_data_check, time/flux/flux_err state, _info helper,
        selector/outlier_segments/outlier_patches, canvas_outliers,
        figure_outliers, plot_data_in_detrending, _fixed_xlim/_fixed_ylim
    """

    def _init_outlier_tab(self) -> QWidget:
        outlier_tab = QWidget()
        outlier_layout = QVBoxLayout()

        # Manual Detection Group
        manual_detection_group = QGroupBox("Manual Detection")
        manual_layout = QVBoxLayout()

        self.select_region_button = QPushButton("Select Region")
        self.select_region_button.clicked.connect(self.enable_region_selection)
        manual_layout.addWidget(self.select_region_button)

        self.remove_selected_button = QPushButton("Remove Region")
        self.remove_selected_button.clicked.connect(self.remove_selected_interval)
        manual_layout.addWidget(self.remove_selected_button)

        manual_layout.addStretch()
        manual_detection_group.setLayout(manual_layout)
        manual_detection_group.setFixedHeight(100)

        # Automated Detection Group
        automated_detection_group = QGroupBox("Automated Detection")
        automated_layout = QVBoxLayout()

        detection_method_layout = QHBoxLayout()
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["Sigma Clipping", "Boxplot", "Chauvenet"])
        self.outlier_method_combo.currentIndexChanged.connect(self.toggle_sigma_input)
        self.outlier_method_combo.setToolTip("Select the method for detecting outliers")
        detection_method_layout.addWidget(self.outlier_method_combo)

        self.sigma_value_spinbox = QSpinBox()
        self.sigma_value_spinbox.setRange(1, 10)
        self.sigma_value_spinbox.setValue(3)
        self.sigma_value_spinbox.setFixedWidth(50)
        self.sigma_value_spinbox.setToolTip("Select the sigma value for clipping outliers")
        detection_method_layout.addWidget(self.sigma_value_spinbox)

        automated_layout.addLayout(detection_method_layout)

        apply_remove_layout = QHBoxLayout()
        self.apply_detection_button = QPushButton("Apply Detection")
        self.apply_detection_button.clicked.connect(self.apply_outlier_detection)
        apply_remove_layout.addWidget(self.apply_detection_button)

        self.remove_outliers_button = QPushButton("Remove Outliers")
        self.remove_outliers_button.clicked.connect(self.remove_outliers)
        apply_remove_layout.addWidget(self.remove_outliers_button)

        automated_layout.addLayout(apply_remove_layout)
        automated_layout.addStretch()
        automated_detection_group.setLayout(automated_layout)
        automated_detection_group.setFixedHeight(100)

        # Normalization Group
        normalization_group = QGroupBox("Normalization")
        normalization_layout = QVBoxLayout()

        self.apply_normalization_button = QPushButton("Apply Normalization")
        self.apply_normalization_button.clicked.connect(self.apply_normalization)
        normalization_layout.addWidget(self.apply_normalization_button)

        normalization_group.setLayout(normalization_layout)
        normalization_group.setFixedHeight(100)

        top_layout = QHBoxLayout()
        top_layout.addWidget(manual_detection_group)
        top_layout.addWidget(automated_detection_group)
        top_layout.addWidget(normalization_group)
        outlier_layout.addLayout(top_layout)

        restore_original_data_button = QPushButton("Restore Original Data")
        restore_original_data_button.clicked.connect(self.plot_original_data)
        outlier_layout.addWidget(restore_original_data_button)

        outlier_light_curve_group = QGroupBox("Light Curve")
        outlier_light_curve_layout = QVBoxLayout()

        self.figure_outliers = Figure()
        self.canvas_outliers = FigureCanvas(self.figure_outliers)
        self.toolbar_outliers = NavigationToolbar(self.canvas_outliers, self)
        self.outlier_error_checkbox = QCheckBox("Display with Errors")
        self.outlier_error_checkbox.setChecked(True)
        self.outlier_error_checkbox.stateChanged.connect(self.update_outlier_plot)

        toolbar_layout_outliers = QHBoxLayout()
        toolbar_layout_outliers.addWidget(self.toolbar_outliers)
        toolbar_layout_outliers.addWidget(self.outlier_error_checkbox)
        toolbar_layout_outliers.setAlignment(Qt.AlignLeft)

        outlier_light_curve_layout.addWidget(self.canvas_outliers)
        outlier_light_curve_layout.addLayout(toolbar_layout_outliers)
        outlier_light_curve_group.setLayout(outlier_light_curve_layout)

        outlier_layout.addWidget(outlier_light_curve_group)

        bottom_button_layout = QHBoxLayout()
        self.detach_plot_button = QPushButton("Detached")
        self.detach_plot_button.clicked.connect(self.detach_plot)
        self.detach_plot_button.setToolTip("Click to detach the plot for viewing separately.")
        bottom_button_layout.addWidget(self.detach_plot_button)

        self.save_cleaned_data_button = QPushButton("Save Cleaned Data")
        self.save_cleaned_data_button.clicked.connect(self.save_cleaned_data)
        self.save_cleaned_data_button.setToolTip("Click to save the cleaned data to a file.")
        bottom_button_layout.addWidget(self.save_cleaned_data_button)

        outlier_layout.addLayout(bottom_button_layout)
        outlier_tab.setLayout(outlier_layout)

        return outlier_tab

    def update_outlier_plot(self):
        """Update the outlier plot based on the checkbox state."""
        self._forget_manual_regions()
        self.plot_data_check(self.figure_outliers, self.canvas_outliers, self.outlier_error_checkbox.isChecked())

    def save_cleaned_data(self):
        try:
            if self.time is None or self.flux is None or self.flux_err is None:
                QMessageBox.warning(self, "Warning", "No cleaned data available to save.")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Save Options")

            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format to save:"))

            self.file_format_combo = QComboBox()
            self.file_format_combo.addItems(["XLSX", "TXT", "CSV"])
            layout.addWidget(self.file_format_combo)

            self.add_2457000_checkbox = QCheckBox("Add 2457000 to Time")
            self.add_2457000_checkbox.setChecked(True)
            layout.addWidget(self.add_2457000_checkbox)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            file_format = self.file_format_combo.currentText().lower()

            time_data = self.time + 2457000 if self.add_2457000_checkbox.isChecked() else self.time
            data = pd.DataFrame({
                "Time": time_data,
                "Flux": self.flux,
                "Error": self.flux_err
            })

            file_name, _ = QFileDialog.getSaveFileName(
                self, f"Save Data as {file_format.upper()}", "", f"{file_format.upper()} Files (*.{file_format})"
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
                        data.to_excel(writer, sheet_name="Cleaned Data", index=False, float_format="%.10f")
                elif file_format == "txt":
                    np.savetxt(
                        file_name,
                        data.values,
                        fmt="%.10f",
                        header="\t".join(data.columns),
                        delimiter="\t",
                        comments=""
                    )

                self._info(f"Cleaned data saved successfully to {file_name}.")

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"An error occurred while saving data as {file_format.upper()}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def detach_plot(self):
        try:
            if self.time is None or self.flux is None or self.flux_err is None:
                QMessageBox.warning(self, "Warning", "No data available for plotting.")
                return

            fig1, ax = plt.subplots()
            ax.errorbar(
                self.time,
                self.flux,
                yerr=self.flux_err,
                fmt=".",
                color="dodgerblue",
                markersize=5,
                capsize=1,
                elinewidth=1.5,
                ls="none",
            )
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Detached Light Curve")
            fig1.tight_layout()
            fig1.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while detaching plot: {e}")

    def enable_region_selection(self):
        """Activate rectangle selection on the Outlier plot."""
        if getattr(self, "selector", None) is not None:
            self.selector.set_active(False)
            self.selector = None

        ax = self.figure_outliers.gca()
        self._fixed_xlim = ax.get_xlim()
        self._fixed_ylim = ax.get_ylim()

        self.selector = RectangleSelector(
            ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        self._info("Drag to select the region you want to remove.")

    def on_select(self, eclick, erelease):
        """Handle rectangle selection in Outlier plot."""
        self.selected_start_time = eclick.xdata
        self.selected_end_time = erelease.xdata
        self.selected_start_flux = eclick.ydata
        self.selected_end_flux = erelease.ydata

        ax = self.figure_outliers.gca()

        if getattr(self, "region_patch", None) is not None:
            try:
                self.region_patch.remove()
            except ValueError:
                pass
            self.region_patch = None

        x1, x2 = sorted([self.selected_start_time, self.selected_end_time])
        y1, y2 = sorted([self.selected_start_flux, self.selected_end_flux])

        self.outlier_segments.append((x1, x2, y1, y2))

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="yellow", edgecolor="none", alpha=0.3, zorder=10)
        ax.add_patch(rect)
        self.outlier_patches.append(rect)

        if getattr(self, "_fixed_xlim", None) and getattr(self, "_fixed_ylim", None):
            ax.set_xlim(self._fixed_xlim)
            ax.set_ylim(self._fixed_ylim)

        self.canvas_outliers.draw_idle()

    def remove_selected_interval(self):
        """Remove data inside manually selected regions."""
        try:
            if not self.outlier_segments:
                QMessageBox.warning(self, "Warning", "No manual segment stored.")
                return

            mask = np.ones_like(self.time, dtype=bool)
            for x1, x2, y1, y2 in self.outlier_segments:
                mask &= ~((self.time >= x1) & (self.time <= x2) & (self.flux >= y1) & (self.flux <= y2))

            self.time = self.time[mask]
            self.flux = self.flux[mask]
            self.flux_err = self.flux_err[mask]

            for patch in getattr(self, "outlier_patches", []):
                try:
                    patch.remove()
                except ValueError:
                    pass
            self.outlier_patches.clear()
            self.outlier_segments.clear()

            if getattr(self, "selector", None) is not None:
                try:
                    for artist in getattr(self.selector, "artists", []):
                        artist.set_visible(False)
                    if hasattr(self.selector, "_selection_artist"):
                        self.selector._selection_artist.set_visible(False)

                    self.selector.set_active(False)
                    self.selector.disconnect_events()
                except Exception:
                    pass
                self.selector = None

            self.plot_data_in_outliers()
            self.plot_data_in_detrending()
            self.canvas_outliers.draw_idle()

            self._info("Selected region(s) have been removed.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def get_selected_range_from_plot(self):
        """Attach a selector and return the currently selected start/end times."""

        def onselect(eclick, erelease):
            self.selected_start_time = eclick.xdata
            self.selected_end_time = erelease.xdata

            ax = self.figure_outliers.gca()
            ax.axvspan(self.selected_start_time, self.selected_end_time, color="yellow", alpha=0.3)
            self.canvas_outliers.draw()

        ax = self.figure_outliers.gca()

        self.selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            interactive=True,
            button=[1],
            minspanx=0.1,
            minspany=0.1,
            spancoords="data",
        )

        self.canvas_outliers.draw()

        return getattr(self, "selected_start_time", None), getattr(self, "selected_end_time", None)

    def remove_selected_data(self, time, flux, flux_err, selected_range):
        """Remove data within the selected time range for time, flux, and flux_err."""
        start_time, end_time = selected_range

        if start_time > end_time:
            start_time, end_time = end_time, start_time

        mask = (time < start_time) | (time > end_time)

        time_filtered = time[mask]
        flux_filtered = flux[mask]
        flux_err_filtered = flux_err[mask]

        return time_filtered, flux_filtered, flux_err_filtered

    def remove_selected_time_flux_data(self, time, flux, flux_err, selected_range_time, selected_range_flux):
        """Remove data within the selected time and flux range and return the filtered data."""
        start_time, end_time = selected_range_time
        start_flux, end_flux = selected_range_flux

        if start_time > end_time:
            start_time, end_time = end_time, start_time
        if start_flux > end_flux:
            start_flux, end_flux = end_flux, start_flux

        mask = ((time < start_time) | (time > end_time)) | ((flux < start_flux) | (flux > end_flux))

        time_filtered = time[mask]
        flux_filtered = flux[mask]
        flux_err_filtered = flux_err[mask]

        return time_filtered, flux_filtered, flux_err_filtered

    def update_plot(self, time, flux):
        """Utility to redraw a simple time/flux plot (kept for compatibility)."""
        if getattr(self, "canvas", None) is not None:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        ax = self.figure.add_subplot(111)
        ax.plot(time, flux, ".", color="blue", markersize=5)
        ax.set_ylabel("Flux")
        ax.set_xlabel("Time")
        self.canvas.draw()

    def _forget_manual_regions(self):
        """Drop manual selections; a redraw erases their boxes, so they must not stay active unseen."""
        self.outlier_segments.clear()
        self.outlier_patches.clear()

    def plot_data_in_outliers(self):
        """Plot data in the Outliers tab based on the current state of self.time, self.flux, and self.flux_err."""
        self._forget_manual_regions()
        ax = self.figure_outliers.gca()
        ax.clear()

        if self.outlier_error_checkbox.isChecked() and self.flux_err is not None:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                        markersize=5, capsize=1, elinewidth=1.5, ls="none")
        else:
            ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)

        ax.set_ylabel("Flux ($electrons^{-1}$)")
        ax.set_xlabel("Time - 2457000 ($day$)")
        ax.set_title("Light Curve without Outliers")

        self.canvas_outliers.draw()

    # -------------------- Outlier detection helpers --------------------
    def apply_outlier_detection(self):
        method = self.outlier_method_combo.currentText()
        if method == "Sigma Clipping":
            sigma_val = self.sigma_value_spinbox.value()
            self.apply_sigma_clipping(sigma_val)
        elif method == "Boxplot":
            self.apply_boxplot_outliers()
        elif method == "Chauvenet":
            self.apply_chauvenet_outliers()

    def remove_outliers(self):
        method = self.outlier_method_combo.currentText()
        if method == "Sigma Clipping":
            self.remove_sigma_clipped_data()
        elif method == "Boxplot":
            self.remove_boxplot_outliers()
        elif method == "Chauvenet":
            self.remove_chauvenet_outliers()

    def apply_sigma_clipping(self, sigma_value):
        """Apply sigma clipping using the provided sigma value."""
        try:
            if not hasattr(self, "current_flux") or self.current_flux is None:
                self.current_flux = self.original_flux.copy()
                self.current_time = self.original_time.copy()
                self.current_flux_err = self.original_flux_err.copy()

            overall_mean = np.mean(self.current_flux)
            overall_std = np.std(self.current_flux)

            lower_bound = overall_mean - sigma_value * overall_std
            upper_bound = overall_mean + sigma_value * overall_std

            if hasattr(self, "lower_bound_line") and hasattr(self, "upper_bound_line"):
                self.lower_bound_line.remove()
                self.upper_bound_line.remove()

            ax = self.figure_outliers.gca()
            self.lower_bound_line = ax.axhline(lower_bound, color="red", linestyle="--", label=f"{sigma_value} sigma lower bound")
            self.upper_bound_line = ax.axhline(upper_bound, color="green", linestyle="--", label=f"{sigma_value} sigma upper bound")

            self.canvas_outliers.draw()

            self._info(f"Applied {sigma_value}-sigma clipping with mean={overall_mean:.2f} and std={overall_std:.2f}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during sigma clipping: {e}")

    def remove_sigma_clipped_data(self):
        """Remove data outside the sigma clipping bounds and refresh plots."""
        try:
            if hasattr(self, "lower_bound_line") and hasattr(self, "upper_bound_line"):
                lower_bound = self.lower_bound_line.get_ydata()[0]
                upper_bound = self.upper_bound_line.get_ydata()[0]

                mask = (self.current_flux >= lower_bound) & (self.current_flux <= upper_bound)

                if np.any(~mask):
                    self.current_time = self.current_time[mask]
                    self.current_flux = self.current_flux[mask]
                    self.current_flux_err = self.current_flux_err[mask]

                    self.time = self.current_time
                    self.flux = self.current_flux
                    self.flux_err = self.current_flux_err

                    self.plot_data_in_outliers()
                    self.plot_data_in_detrending()

                    self._info("Data outside of the sigma clipping bounds has been removed and graph updated.")
                else:
                    self._info("Sigma clipping removed no data points.")
            else:
                QMessageBox.warning(self, "Warning", "No sigma clipping has been applied yet.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while removing sigma clipped data: {e}")

    def apply_boxplot_outliers(self):
        try:
            if not hasattr(self, "current_flux") or self.current_flux is None:
                self.current_flux = self.original_flux.copy()
                self.current_time = self.original_time.copy()
                self.current_flux_err = self.original_flux_err.copy()

            q1 = np.percentile(self.current_flux, 25)
            q3 = np.percentile(self.current_flux, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if hasattr(self, "boxplot_lower_line") and hasattr(self, "boxplot_upper_line"):
                self.boxplot_lower_line.remove()
                self.boxplot_upper_line.remove()

            ax = self.figure_outliers.gca()
            self.boxplot_lower_line = ax.axhline(lower_bound, color="red", linestyle="--", label="Boxplot lower bound")
            self.boxplot_upper_line = ax.axhline(upper_bound, color="green", linestyle="--", label="Boxplot upper bound")
            self.canvas_outliers.draw()

            self._info(f"Boxplot thresholds applied. Lower: {lower_bound:.3f}, Upper: {upper_bound:.3f}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying Boxplot outliers: {e}")

    def remove_boxplot_outliers(self):
        try:
            if hasattr(self, "boxplot_lower_line") and hasattr(self, "boxplot_upper_line"):
                lower_bound = self.boxplot_lower_line.get_ydata()[0]
                upper_bound = self.boxplot_upper_line.get_ydata()[0]

                mask = (self.current_flux >= lower_bound) & (self.current_flux <= upper_bound)

                if np.any(~mask):
                    self.current_time = self.current_time[mask]
                    self.current_flux = self.current_flux[mask]
                    self.current_flux_err = self.current_flux_err[mask]

                    self.time = self.current_time
                    self.flux = self.current_flux
                    self.flux_err = self.current_flux_err

                    self.plot_data_in_outliers()
                    self.plot_data_in_detrending()

                    self._info("Data outside of Boxplot bounds has been removed.")
                else:
                    self._info("No data points removed (all within Boxplot bounds).")
            else:
                QMessageBox.warning(self, "Warning", "Boxplot thresholds have not been applied yet.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while removing Boxplot outliers: {e}")

    def apply_chauvenet_outliers(self):
        try:
            if not hasattr(self, "current_flux") or self.current_flux is None:
                self.current_flux = self.original_flux.copy()
                self.current_time = self.original_time.copy()
                self.current_flux_err = self.original_flux_err.copy()

            mean_flux = np.mean(self.current_flux)
            std_flux = np.std(self.current_flux)
            n = len(self.current_flux)

            probability = 1.0 / (2 * n)
            critical_value = np.sqrt(2) * erfcinv(2 * probability)

            lower_bound = mean_flux - critical_value * std_flux
            upper_bound = mean_flux + critical_value * std_flux

            if hasattr(self, "chauvenet_lower_line") and hasattr(self, "chauvenet_upper_line"):
                self.chauvenet_lower_line.remove()
                self.chauvenet_upper_line.remove()

            ax = self.figure_outliers.gca()
            self.chauvenet_lower_line = ax.axhline(lower_bound, color="red", linestyle="--", label="Chauvenet lower bound")
            self.chauvenet_upper_line = ax.axhline(upper_bound, color="green", linestyle="--", label="Chauvenet upper bound")
            self.canvas_outliers.draw()

            self._info(f"Chauvenet limit applied. Lower: {lower_bound:.3f}, Upper: {upper_bound:.3f}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying Chauvenet outliers: {e}")

    def remove_chauvenet_outliers(self):
        try:
            if hasattr(self, "chauvenet_lower_line") and hasattr(self, "chauvenet_upper_line"):
                lower_bound = self.chauvenet_lower_line.get_ydata()[0]
                upper_bound = self.chauvenet_upper_line.get_ydata()[0]

                mask = (self.current_flux >= lower_bound) & (self.current_flux <= upper_bound)

                if np.any(~mask):
                    self.current_time = self.current_time[mask]
                    self.current_flux = self.current_flux[mask]
                    self.current_flux_err = self.current_flux_err[mask]

                    self.time = self.current_time
                    self.flux = self.current_flux
                    self.flux_err = self.current_flux_err

                    self.plot_data_in_outliers()
                    self.plot_data_in_detrending()

                    self._info("Data outside of Chauvenet bounds has been removed.")
                else:
                    self._info("No data points removed (all within Chauvenet bounds).")
            else:
                QMessageBox.warning(self, "Warning", "Chauvenet thresholds have not been applied yet.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while removing Chauvenet outliers: {e}")

    def apply_normalization(self):
        """Apply normalization to the current data based on the median value and update the plots."""
        try:
            flux_median = np.median(self.flux)

            normalized_flux = self.flux / flux_median
            normalized_flux_err = self.flux_err / flux_median

            self.time = self.time
            self.flux = normalized_flux
            self.flux_err = normalized_flux_err

            self.update_outlier_data(self.time, self.flux, self.flux_err)

            self.plot_data_in_detrending()

            self._info("Data has been normalized based on median successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during normalization: {e}")

    def plot_original_data(self):
        """Redraw the original data and update the plots."""
        try:
            self.time = self.original_time.copy()
            self.flux = self.original_flux.copy()
            self.flux_err = self.original_flux_err.copy()

            ax = self.figure_outliers.gca()
            ax.clear()
            ax.errorbar(
                self.time,
                self.flux,
                yerr=self.flux_err,
                fmt=".",
                color="dodgerblue",
                markersize=5,
                capsize=1,
                elinewidth=1.5,
                ls="none",
            )
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Original Light Curve from Query Tab")
            self.canvas_outliers.draw()

            self.plot_data_in_detrending()

            self._info("Original data restored in Outlier and Detrending sections.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while restoring the original data: {e}")

    def update_outlier_data(self, time, flux, flux_err):
        """Update cleaned data state and refresh the Outlier/Detrending plots."""
        self.ctrl.set_cleaned_data(time, flux, flux_err)

        ax = self.figure_outliers.gca()
        ax.clear()
        ax.errorbar(
            self.cleaned_time,
            self.cleaned_flux,
            yerr=self.cleaned_flux_err if self.outlier_error_checkbox.isChecked() else None,
            fmt=".",
            color="dodgerblue",
            markersize=5,
            capsize=1,
            elinewidth=1.5,
            ls="none",
        )
        ax.set_ylabel("Normalized Flux ($electrons^{-1}$)")
        ax.set_xlabel("Time - 2457000 ($day$)")
        ax.set_title("Normalized Light Curve")
        self.canvas_outliers.draw()

        self.plot_data_in_detrending()

    def toggle_sigma_input(self):
        """Show sigma input only when Sigma Clipping is selected."""
        method = self.outlier_method_combo.currentText()
        self.sigma_value_spinbox.setVisible(method == "Sigma Clipping")

    def clear_outlier_overlays(self):
        """Reset outlier selections, patches, and threshold lines."""
        for attr in ("outlier_patches", "manual_patches"):
            patches = getattr(self, attr, [])
            for p in list(patches):
                try:
                    p.remove()
                except ValueError:
                    pass
            patches.clear()

        for attr in (
            "lower_bound_line",
            "upper_bound_line",
            "boxplot_lower_line",
            "boxplot_upper_line",
            "chauvenet_lower_line",
            "chauvenet_upper_line",
        ):
            line = getattr(self, attr, None)
            if line is not None:
                try:
                    line.remove()
                except ValueError:
                    pass
                setattr(self, attr, None)

        if getattr(self, "selector", None) is not None:
            try:
                self.selector.set_active(False)
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        if getattr(self, "region_patch", None) is not None:
            try:
                self.region_patch.remove()
            except ValueError:
                pass
            self.region_patch = None

        self._fixed_xlim = None
        self._fixed_ylim = None

    def restore_data_detrending(self):
        """Restore data in the Detrending tab to match the latest cleaned data from the Outlier tab."""
        if self.cleaned_time is not None and self.cleaned_flux is not None and self.cleaned_flux_err is not None:
            self.time = self.cleaned_time
            self.flux = self.cleaned_flux
            self.flux_err = self.cleaned_flux_err
            data_type = "Cleaned"
        else:
            self.time = self.original_time
            self.flux = self.original_flux
            self.flux_err = self.original_flux_err
            data_type = "Original"

        self.plot_data_in_detrending()
        self._info(f"Detrending data has been restored to {data_type} data.")
