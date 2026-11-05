import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QSizePolicy,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
)


class DetrendingMixin:
    """
    Detrending tab UI and logic.
    Expects host to provide:
      - state: time/flux/flux_err, cleaned data, pre_detrend caches, _info helper, plot_data_check
      - attrs: detrend_segments, automated_segments, detrending_selector, detrending_patches, _fixed_xlim/_fixed_ylim
      - figures/canvases: figure_detrending, canvas_detrending
    """

    def _init_detrending_tab(self) -> QWidget:
        detrending_tab = QWidget()
        detrending_layout = QVBoxLayout()

        # Manual Detection Group
        manual_detrending_group = QGroupBox("Manual Detection")
        manual_detrending_layout = QVBoxLayout()

        self.select_detrending_region_button = QPushButton("Select Region for Manual Detrending")
        self.select_detrending_region_button.clicked.connect(self.enable_detrending_region_selection)

        manual_buttons_layout = QHBoxLayout()
        self.show_manual_trend_button = QPushButton("Show Trend")
        self.show_manual_trend_button.clicked.connect(self.show_manual_trend)
        self.remove_manual_trend_button = QPushButton("Remove Trend")
        self.remove_manual_trend_button.clicked.connect(self.remove_manual_trend)

        manual_buttons_layout.addWidget(self.show_manual_trend_button)
        manual_buttons_layout.addWidget(self.remove_manual_trend_button)

        manual_detrending_layout.addWidget(self.select_detrending_region_button)
        manual_detrending_layout.addLayout(manual_buttons_layout)

        manual_detrending_group.setLayout(manual_detrending_layout)
        manual_detrending_group.setFixedHeight(100)

        # Polynomial Fit Group
        polynomial_fit_group = QGroupBox("Polynomial Fit")
        polynomial_fit_layout = QVBoxLayout()

        self.degree_combo = QComboBox()
        self.degree_combo.addItems(
            [
                "1st Degree Polynomial",
                "2nd Degree Polynomial",
                "3rd Degree Polynomial",
                "4th Degree Polynomial",
            ]
        )
        self.degree_combo.setCurrentIndex(1)

        polynomial_fit_layout.addWidget(self.degree_combo)
        polynomial_fit_group.setLayout(polynomial_fit_layout)
        polynomial_fit_group.setFixedHeight(100)

        # Automated Detection Group
        automated_detection_group = QGroupBox("Automated Detection")
        automated_detection_layout = QVBoxLayout()
        threshold_detection_layout = QHBoxLayout()

        self.detect_gaps_button = QPushButton("Detect Gaps for Automated Detrending")
        self.detect_gaps_button.clicked.connect(self.detect_gaps)

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0, 10)
        self.threshold_input.setValue(1.0)
        self.threshold_input.setSingleStep(0.1)
        self.threshold_input.setDecimals(2)
        self.threshold_input.setLocale(QLocale(QLocale.English))

        threshold_detection_layout.addWidget(self.detect_gaps_button)
        threshold_detection_layout.addWidget(self.threshold_input)

        automated_buttons_layout = QHBoxLayout()
        self.show_automated_trend_button = QPushButton("Show Trend")
        self.show_automated_trend_button.clicked.connect(self.show_automated_trend)
        self.remove_automated_trend_button = QPushButton("Remove Trend")
        self.remove_automated_trend_button.clicked.connect(self.remove_automated_trend)

        automated_buttons_layout.addWidget(self.show_automated_trend_button)
        automated_buttons_layout.addWidget(self.remove_automated_trend_button)

        automated_detection_layout.addLayout(threshold_detection_layout)
        automated_detection_layout.addLayout(automated_buttons_layout)

        automated_detection_group.setLayout(automated_detection_layout)
        automated_detection_group.setFixedHeight(100)

        detrending_controls_layout = QHBoxLayout()
        detrending_controls_layout.addWidget(manual_detrending_group)
        detrending_controls_layout.addWidget(polynomial_fit_group)
        detrending_controls_layout.addWidget(automated_detection_group)
        detrending_layout.addLayout(detrending_controls_layout)

        self.restore_button = QPushButton("Restore Data")
        self.restore_button.clicked.connect(self.restore_data)
        detrending_layout.addWidget(self.restore_button)

        detrending_light_curve_group = QGroupBox("Light Curve")
        detrending_light_curve_layout = QVBoxLayout()

        self.figure_detrending = Figure()
        self.canvas_detrending = FigureCanvas(self.figure_detrending)
        self.toolbar_detrending = NavigationToolbar(self.canvas_detrending, self)
        self.detrending_error_checkbox = QCheckBox("Display with Errors")
        self.detrending_error_checkbox.setChecked(True)
        self.detrending_error_checkbox.stateChanged.connect(self.update_detrending_plot)

        toolbar_layout_detrending = QHBoxLayout()
        toolbar_layout_detrending.addWidget(self.toolbar_detrending)
        toolbar_layout_detrending.addWidget(self.detrending_error_checkbox)
        toolbar_layout_detrending.setAlignment(Qt.AlignLeft)

        detrending_light_curve_layout.addWidget(self.canvas_detrending)
        detrending_light_curve_layout.addLayout(toolbar_layout_detrending)
        detrending_light_curve_group.setLayout(detrending_light_curve_layout)
        detrending_layout.addWidget(detrending_light_curve_group)

        detrending_button_layout = QHBoxLayout()
        self.detach_detrending_plot_button = QPushButton("Detached")
        self.detach_detrending_plot_button.clicked.connect(self.detach_detrending_plot)
        detrending_button_layout.addWidget(self.detach_detrending_plot_button)

        self.save_detrending_data_button = QPushButton("Save")
        self.save_detrending_data_button.clicked.connect(self.save_detrending_data)
        detrending_button_layout.addWidget(self.save_detrending_data_button)

        detrending_layout.addLayout(detrending_button_layout)

        detrending_tab.setLayout(detrending_layout)
        return detrending_tab

    # -------------------- Plot / update helpers --------------------
    def update_detrending_plot(self):
        """Update the detrending plot based on the checkbox state."""
        self.plot_data_check(self.figure_detrending, self.canvas_detrending, self.detrending_error_checkbox.isChecked())

    def plot_data_in_detrending(self):
        """Plot data in the Detrending tab based on the current state."""
        ax = self.figure_detrending.gca()
        ax.clear()

        if self.detrending_error_checkbox.isChecked() and self.flux_err is not None:
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
        else:
            ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)

        ax.set_ylabel("Flux ($electrons^{-1}$)")
        ax.set_xlabel("Time - 2457000 ($day$)")
        ax.set_title("Detrending of Light Curve")
        self.canvas_detrending.draw()

    # -------------------- Manual detrending --------------------
    def enable_detrending_region_selection(self):
        """Enable rectangle selection for manual detrending."""
        ax = self.figure_detrending.gca()
        if self.detrending_selector is None:
            self.detrending_selector = RectangleSelector(
                ax,
                self.on_select_detrending,
                useblit=True,
                button=[1],
                minspanx=0.1,
                minspany=0.1,
                spancoords="data",
                interactive=True,
            )
        else:
            self.detrending_selector.set_active(True)

        self._fixed_xlim = ax.get_xlim()
        self._fixed_ylim = ax.get_ylim()
        self._info("Drag to select regions for manual detrending.")

    def on_select_detrending_region(self, eclick, erelease):
        """Callback when a region is selected for detrending based on time/flux values."""
        start_time = eclick.xdata
        end_time = erelease.xdata
        if start_time is not None and end_time is not None:
            self.detrend_segments.append((min(start_time, end_time), max(start_time, end_time)))
            self._info(f"Manual segment added from {start_time:.2f} to {end_time:.2f}.")
        else:
            QMessageBox.warning(self, "Warning", "No valid region selected.")

    def on_select_detrending(self, eclick, erelease):
        try:
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])

            self.detrend_segments.append((x1, x2))

            ax = self.figure_detrending.gca()
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                facecolor="yellow",
                edgecolor="none",
                alpha=0.30,
                zorder=10,
            )
            ax.add_patch(rect)
            self.detrending_patches.append(rect)

            ax.set_xlim(self._fixed_xlim)
            ax.set_ylim(self._fixed_ylim)

            self.canvas_detrending.draw_idle()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"on_select_detrending failed: {e}")

    def show_manual_trend(self):
        try:
            degree = int(self.degree_combo.currentText().split()[0][0])

            if not self.detrend_segments:
                QMessageBox.warning(self, "Warning", "No manual regions selected for detrending.")
                return

            ax = self.figure_detrending.gca()
            ax.clear()
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)

            for start_time, end_time in self.detrend_segments:
                mask = (self.time >= start_time) & (self.time <= end_time)
                if not mask.any():
                    continue

                p = np.poly1d(np.polyfit(self.time[mask], self.flux[mask], w=1 / self.flux_err[mask], deg=degree))
                ax.plot(self.time[mask], p(self.time[mask]), "--", color="red", zorder=10)

            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Detrending of Light Curve")

            if hasattr(self, "detrending_selector") and self.detrending_selector:
                self.detrending_selector.set_active(False)
                self.detrending_selector.disconnect_events()
                self.detrending_selector = None

            if hasattr(self, "detrending_patch") and self.detrending_patch:
                try:
                    self.detrending_patch.remove()
                except ValueError:
                    pass
                self.detrending_patch = None

            self.canvas_detrending.draw_idle()
            self._info(f"Manual detrending displayed with degree {degree}. Selection cleared.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while showing manual trend: {e}")

    def remove_manual_trend(self):
        """Remove polynomial fit (detrend) for manually selected segments."""
        try:
            if self.pre_detrend_time is None:
                self.pre_detrend_time = self.time.copy()
                self.pre_detrend_flux = self.flux.copy()
                self.pre_detrend_flux_err = self.flux_err.copy()

            if not self.detrend_segments:
                QMessageBox.warning(self, "Warning", "No regions selected for manual detrending.")
                return

            degree = int(self.degree_combo.currentText().split()[0][0])

            updated_flux = self.flux.copy()
            updated_flux_err = self.flux_err.copy()

            for start_time, end_time in self.detrend_segments:
                mask = (self.time >= start_time) & (self.time <= end_time)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]

                if len(time_segment) == 0:
                    continue

                pdc_coef = np.polyfit(time_segment, flux_segment, w=1 / flux_err_segment, deg=degree)
                pdc_parabola = np.poly1d(pdc_coef)
                pdc_par_val = pdc_parabola(time_segment)

                updated_flux[mask] = flux_segment / pdc_par_val
                updated_flux_err[mask] = flux_err_segment / pdc_par_val

            self.flux = updated_flux
            self.flux_err = updated_flux_err

            median_flux = np.median(self.flux)
            self.flux /= median_flux
            self.flux_err /= median_flux

            self.detrend_segments.clear()

            ax = self.figure_detrending.gca()
            ax.clear()
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)

            ax.set_ylabel("Normalized Flux")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Detrended and Normalized Light Curve - Manual")
            self.canvas_detrending.draw()

            self._info("Manual detrending applied and normalized.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while removing manual detrending: {e}")

    # -------------------- Automated detrending --------------------
    def detect_gaps(self):
        """Detect gaps and set segment mode to automate."""
        try:
            self.segment_mode = "automate"
            self.segment_data()

            if not self.automated_segments:
                QMessageBox.warning(self, "Warning", "No segments detected.")
                return

            ax = self.figure_detrending.gca()
            ax.clear()
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5, ls="none")

            ax.set_ylabel("Normalized Flux")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Light Curve with Segments")

            for i, (start_time, end_time) in enumerate(self.automated_segments):
                ax.axvspan(start_time, end_time, color="gray", alpha=0.3, label="Detected Segment" if i == 0 else "")

            ax.legend(loc="best")
            self.canvas_detrending.draw()

            self._info(f"{len(self.automated_segments)} segment(s) detected based on threshold {self.threshold_input.value()}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while detecting gaps: {e}")

    def segment_data(self):
        """Segment data based on threshold gaps."""
        time_diffs = np.diff(self.time)
        threshold = self.threshold_input.value()
        gap_indices = np.where(time_diffs > threshold)[0]

        self.automated_segments.clear()
        start_idx = 0
        for idx in gap_indices:
            end_idx = idx + 1
            self.automated_segments.append((self.time[start_idx], self.time[end_idx - 1]))
            start_idx = end_idx

        if start_idx < len(self.time) - 1:
            self.automated_segments.append((self.time[start_idx], self.time[-1]))

    def show_automated_trend(self):
        try:
            degree = int(self.degree_combo.currentText().split()[0][0])

            if not self.automated_segments:
                QMessageBox.warning(self, "Warning", "No automated regions selected for detrending.")
                return

            ax = self.figure_detrending.gca()
            ax.clear()
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5, ls="none")

            for start_time, end_time in self.automated_segments:
                mask = (self.time >= start_time) & (self.time <= end_time)
                if not mask.any():
                    continue

                p = np.poly1d(np.polyfit(self.time[mask], self.flux[mask], w=1 / self.flux_err[mask], deg=degree))
                ax.plot(self.time[mask], p(self.time[mask]), "--", color="red", zorder=10)

            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Automated Detrending of Light Curve")
            self.canvas_detrending.draw()

            self._info(f"Automated detrending displayed with degree {degree}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while showing automated trend: {e}")

    def remove_automated_trend(self):
        """Remove polynomial fit (detrend) for automatically detected segments."""
        try:
            if self.pre_detrend_time is None:
                self.pre_detrend_time = self.time.copy()
                self.pre_detrend_flux = self.flux.copy()
                self.pre_detrend_flux_err = self.flux_err.copy()

            if not self.automated_segments:
                QMessageBox.warning(self, "Warning", "No segments detected for automated detrending.")
                return

            degree = int(self.degree_combo.currentText().split()[0][0])

            updated_flux = self.flux.copy()
            updated_flux_err = self.flux_err.copy()

            for start_time, end_time in self.automated_segments:
                mask = (self.time >= start_time) & (self.time <= end_time)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]

                if len(time_segment) == 0:
                    continue

                pdc_coef = np.polyfit(time_segment, flux_segment, w=1 / flux_err_segment, deg=degree)
                pdc_parabola = np.poly1d(pdc_coef)
                pdc_par_val = pdc_parabola(time_segment)

                updated_flux[mask] = flux_segment / pdc_par_val
                updated_flux_err[mask] = flux_err_segment / pdc_par_val

            self.flux = updated_flux
            self.flux_err = updated_flux_err

            median_flux = np.median(self.flux)
            self.flux /= median_flux
            self.flux_err /= median_flux

            self.automated_segments.clear()

            ax = self.figure_detrending.gca()
            ax.clear()
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)

            ax.set_ylabel("Normalized Flux")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Detrended and Normalized Light Curve")
            self.canvas_detrending.draw()

            self._info("Automated detrending applied and normalized.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while removing automated detrending: {e}")

    # -------------------- Restore / detach / save --------------------
    def restore_data(self):
        """Restore data in the Detrending tab to match the current data in the Outlier tab."""
        try:
            if self.pre_detrend_flux is not None:
                self.time = self.pre_detrend_time.copy()
                self.flux = self.pre_detrend_flux.copy()
                self.flux_err = self.pre_detrend_flux_err.copy()
                self.ctrl.clear_pre_detrend()
            else:
                state = self.ctrl.restore_cleaned_or_original()
                if state == "Empty":
                    QMessageBox.warning(self, "Warning", "No data available to restore in detrending tab.")
                    return

            self.plot_data_in_detrending()
            self._info("Detrending data restored to match Outlier tab.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while restoring data: {e}")

    def detach_detrending_plot(self):
        """Detach the detrended plot and display it in a separate window."""
        try:
            if self.cleaned_time is not None and len(self.cleaned_time) > 0:
                time = self.cleaned_time
                flux = self.cleaned_flux
                flux_err = self.cleaned_flux_err
            elif self.time is not None and len(self.time) > 0:
                time = self.time
                flux = self.flux
                flux_err = self.flux_err
            else:
                QMessageBox.warning(self, "Warning", "No data available for plotting.")
                return

            fig, ax = plt.subplots()

            ax.errorbar(time, flux, yerr=flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5, ls="none")
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Time - 2457000 ($day$)")
            ax.set_title("Detrended Light Curve - Detached View")
            fig.tight_layout()
            fig.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while detaching the detrending plot: {e}")

    def save_detrending_data(self):
        """Save only the detrended data in the chosen file format."""
        try:
            file_format_combo = QComboBox()
            file_format_combo.addItems(["XLSX", "CSV", "TXT"])

            dialog = QDialog(self)
            dialog.setWindowTitle("Save Detrended Data Options")

            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format to save:"))
            layout.addWidget(file_format_combo)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            file_format = file_format_combo.currentText().lower()

            file_name, _ = QFileDialog.getSaveFileName(
                self,
                f"Save Detrended Data as {file_format.upper()}",
                "",
                f"{file_format.upper()} Files (*.{file_format})",
            )
            if not file_name:
                return

            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"

            detrended_data = pd.DataFrame({"Time": self.time, "Flux": self.flux, "Error": self.flux_err})

            try:
                if file_format == "xlsx":
                    with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                        detrended_data.to_excel(writer, sheet_name="Detrended Data", index=False, float_format="%.10f")
                elif file_format == "csv":
                    detrended_data.to_csv(file_name, index=False, float_format="%.10f")
                elif file_format == "txt":
                    np.savetxt(
                        file_name,
                        detrended_data.values,
                        fmt="%.10f",
                        header="\t".join(detrended_data.columns),
                        delimiter="\t",
                        comments="",
                    )

                self._info(f"Detrended data saved as {file_format.upper()}.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving detrended data as {file_format.upper()}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while saving detrended data: {e}")
