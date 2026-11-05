import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QDialogButtonBox,
    QMessageBox,
    QDialog,
    QLabel,
    QSizePolicy,
)

from tess_util import phase_fold_manually, eclipse_ephemeris, sinusoid_ephemeris
from lcm_app.tess.threads import PhaseFoldThread

FOLD_METHODS = {
    "BLS + KvW (eclipsing)": eclipse_ephemeris,
    "Lomb-Scargle (sinusoidal)": lambda time, flux: (*sinusoid_ephemeris(time, flux), None),
}


class PhaseMixin:
    """
    Phase tab UI and logic.
    Expects host to provide:
      - state: time/flux/flux_err, _info helper, plot_data_check
      - attrs: phase_time/flux/flux_err (created lazily)
    """

    def _init_phase_tab(self) -> QWidget:
        phase_tab = QWidget()
        phase_layout = QVBoxLayout()

        # Manual Phase Folding Group
        manual_group = QGroupBox("Manual Phase Folding")
        manual_layout = QVBoxLayout()

        manual_inputs_layout = QHBoxLayout()
        self.t0_input = QLineEdit()
        self.t0_input.setPlaceholderText("e.g., 2451350.024259")
        self.period_input = QLineEdit()
        self.period_input.setPlaceholderText("e.g., 0.2375")
        manual_inputs_layout.addWidget(QLabel("Minima Time:"))
        manual_inputs_layout.addWidget(self.t0_input)
        manual_inputs_layout.addWidget(QLabel("Period Value:"))
        manual_inputs_layout.addWidget(self.period_input)

        manual_button_layout = QHBoxLayout()
        manual_phase_button = QPushButton("Phase Fold Manually")
        manual_phase_button.clicked.connect(self.manual_phase_folding)
        manual_button_layout.addWidget(manual_phase_button)

        manual_layout.addLayout(manual_inputs_layout)
        manual_layout.addLayout(manual_button_layout)
        manual_group.setLayout(manual_layout)

        # Automate Phase Folding Group
        automate_group = QGroupBox("Automate Phase Folding")
        automate_layout = QVBoxLayout()

        automate_inputs_layout = QHBoxLayout()
        self.best_t0_display = QLineEdit("<Not calculated>")
        self.best_t0_display.setReadOnly(True)
        self.best_t0_display.setStyleSheet("background-color: #f0f0f0;")
        self.best_period_display = QLineEdit("<Not calculated>")
        self.best_period_display.setReadOnly(True)
        self.best_period_display.setStyleSheet("background-color: #f0f0f0;")
        automate_inputs_layout.addWidget(QLabel("Adjusted Minima Time:"))
        automate_inputs_layout.addWidget(self.best_t0_display)
        automate_inputs_layout.addWidget(QLabel("Best Period:"))
        automate_inputs_layout.addWidget(self.best_period_display)

        automate_button_layout = QHBoxLayout()
        self.period_dropdown = QComboBox()
        self.period_dropdown.addItems(FOLD_METHODS)
        self.automate_button = QPushButton("Automate Phase Folding")
        self.automate_button.clicked.connect(self.automate_phase_folding)
        automate_button_layout.addWidget(self.period_dropdown)
        automate_button_layout.addWidget(self.automate_button)

        automate_layout.addLayout(automate_inputs_layout)
        automate_layout.addLayout(automate_button_layout)
        automate_group.setLayout(automate_layout)

        phase_controls_layout = QHBoxLayout()
        manual_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        automate_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        phase_controls_layout.addWidget(manual_group)
        phase_controls_layout.addWidget(automate_group)
        phase_layout.addLayout(phase_controls_layout)

        # Light Curve Group
        phase_light_curve_group = QGroupBox("Light Curve")
        phase_light_curve_layout = QVBoxLayout()

        self.figure_phase = Figure()
        self.canvas_phase = FigureCanvas(self.figure_phase)
        self.toolbar_phase = NavigationToolbar(self.canvas_phase, self)
        self.phase_error_checkbox = QCheckBox("Display with Errors")
        self.phase_error_checkbox.setChecked(True)
        self.phase_error_checkbox.stateChanged.connect(self.update_phase_plot)

        toolbar_layout_phase = QHBoxLayout()
        toolbar_layout_phase.addWidget(self.toolbar_phase)
        toolbar_layout_phase.addWidget(self.phase_error_checkbox)
        toolbar_layout_phase.setAlignment(Qt.AlignLeft)

        phase_light_curve_layout.addWidget(self.canvas_phase)
        phase_light_curve_layout.addLayout(toolbar_layout_phase)
        phase_light_curve_group.setLayout(phase_light_curve_layout)
        phase_layout.addWidget(phase_light_curve_group)

        phase_button_layout = QHBoxLayout()
        self.detach_phase_plot_button = QPushButton("Detached")
        self.detach_phase_plot_button.clicked.connect(self.detach_phase_plot)
        phase_button_layout.addWidget(self.detach_phase_plot_button)

        self.save_phase_data_button = QPushButton("Save")
        self.save_phase_data_button.clicked.connect(self.save_phase_data)
        phase_button_layout.addWidget(self.save_phase_data_button)

        phase_layout.addLayout(phase_button_layout)

        phase_tab.setLayout(phase_layout)
        return phase_tab

    # -------------------- Plot helpers --------------------
    def plot_data_in_phase_folding(self, detrended_data=None):
        """Plot data in the Phase Folding tab, including errors if provided."""
        try:
            ax = self.figure_phase.gca()
            ax.clear()

            if detrended_data:
                time_segments, flux_segments, flux_err_segments = detrended_data
                for time_segment, flux_segment, flux_err_segment in zip(time_segments, flux_segments, flux_err_segments):
                    ax.errorbar(
                        time_segment,
                        flux_segment,
                        yerr=flux_err_segment,
                        fmt=".",
                        color="dodgerblue",
                        markersize=5,
                        capsize=1,
                        elinewidth=1.5,
                        label="Detrended and Normalized Data",
                    )
            else:
                ax.errorbar(
                    self.time,
                    self.flux,
                    yerr=self.flux_err,
                    fmt=".",
                    color="dodgerblue",
                    markersize=5,
                    capsize=1,
                    elinewidth=1.5,
                    label="Original Data",
                )

            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_xlabel("Phase")
            ax.set_title("Phase Folded Light Curve")
            ax.legend(loc="best")
            self.canvas_phase.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while plotting in Phase Folding tab: {e}")

    def update_phase_plot(self):
        """Update the phase plot based on the checkbox state."""
        self.plot_data_check(self.figure_phase, self.canvas_phase, self.phase_error_checkbox.isChecked())

    # -------------------- Phase folding --------------------
    def manual_phase_folding(self):
        try:
            t0 = float(self.t0_input.text().strip())
            period = float(self.period_input.text().strip())

            if not t0 or not period:
                raise ValueError("Both t0 and period must be provided.")

            if self.time.size == 0 or self.flux.size == 0 or self.flux_err.size == 0:
                raise ValueError("Loaded data is empty. Please ensure the TESS data is loaded properly.")

            phase, flux, flux_err = phase_fold_manually(
                time=self.time, flux=self.flux, flux_err=self.flux_err, t0=t0, period=period, canvas=self.canvas_phase
            )

            self.phase_time = phase
            self.phase_flux = flux
            self.phase_flux_err = flux_err

            self._info("Manual phase folding completed successfully.")

        except ValueError as ve:
            QMessageBox.warning(self, "Input Error", f"Invalid input for t0 or period: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while manually phase folding: {e}")

    def automate_phase_folding(self):
        if self.time is None or self.time.size == 0:
            QMessageBox.warning(self, "Warning", "No data available for phase folding.")
            return

        self.automate_button.setEnabled(False)
        self.automate_button.setText("Searching…")
        self.fold_thread = PhaseFoldThread(FOLD_METHODS[self.period_dropdown.currentText()], self.time, self.flux)
        self.fold_thread.resultReady.connect(self._on_automate_fold_done)
        self.fold_thread.error.connect(
            lambda m: QMessageBox.critical(self, "Error", f"An error occurred during automatic phase folding: {m}"))
        self.fold_thread.finished.connect(
            lambda: (self.automate_button.setEnabled(True), self.automate_button.setText("Automate Phase Folding")))
        self.fold_thread.start()

    def _on_automate_fold_done(self, result):
        t0, period, n_minima = result
        self.best_t0_display.setText(f"{t0:.6f}")
        self.best_period_display.setText(f"{period:.8f}")
        self.phase_time, self.phase_flux, self.phase_flux_err = phase_fold_manually(
            time=self.time, flux=self.flux, flux_err=self.flux_err, t0=t0, period=period, canvas=self.canvas_phase
        )
        if n_minima == 0:
            QMessageBox.warning(self, "Warning", "No complete eclipses found; showing the BLS estimate only. "
                                                 "Try Lomb-Scargle if the star is not eclipsing.")
        elif n_minima is None:
            self._info("Phase folded with the Lomb-Scargle period.")
        else:
            self._info(f"Phase folded with an ephemeris from {n_minima} KvW minima.")

    # -------------------- Detach / save --------------------
    def detach_phase_plot(self):
        """Detach the phase-folded plot and display it in a separate window."""
        try:
            if not hasattr(self, "phase_time") or self.phase_time is None:
                QMessageBox.warning(self, "Warning", "No data available for plotting.")
                return

            fig, ax = plt.subplots()

            ax.errorbar(
                self.phase_time,
                self.phase_flux,
                yerr=self.phase_flux_err,
                fmt=".",
                color="dodgerblue",
                markersize=5,
                capsize=1,
                elinewidth=1.5,
                ls="none",
            )
            ax.set_ylabel("Normalized Flux")
            ax.set_xlabel("Phase")
            ax.set_title("Detached Light Curve - Phase-Folded Data")

            ax.set_xlim(0.2, 1.2)

            fig.tight_layout()
            fig.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while detaching the phase plot: {e}")

    def save_phase_data(self):
        """Save the phase-folded data in the chosen file format."""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Save Phase-Folded Data Options")

            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format to save:"))

            file_format_combo = QComboBox()
            file_format_combo.addItems(["XLSX", "CSV", "TXT"])
            layout.addWidget(file_format_combo)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            file_format = file_format_combo.currentText().lower()

            file_name, _ = QFileDialog.getSaveFileName(
                self, f"Save Phase-Folded Data as {file_format.upper()}", "", f"{file_format.upper()} Files (*.{file_format})"
            )
            if not file_name:
                return

            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"

            if hasattr(self, "phase_time") and self.phase_time is not None:
                phase_data = pd.DataFrame({"Phase": self.phase_time, "Flux": self.phase_flux, "Error": self.phase_flux_err})
            else:
                QMessageBox.warning(self, "No Data Available", "Phase-folded data is not available to save.")
                return

            try:
                if file_format == "xlsx":
                    with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                        phase_data.to_excel(writer, sheet_name="Phase-Folded Data", index=False, float_format="%.10f")
                elif file_format == "csv":
                    phase_data.to_csv(file_name, index=False, float_format="%.10f")
                elif file_format == "txt":
                    np.savetxt(
                        file_name,
                        phase_data.values,
                        fmt="%.10f",
                        header="\t".join(phase_data.columns),
                        delimiter="\t",
                        comments="",
                    )

                self._info(f"Phase-folded data saved as {file_format.upper()}.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving phase-folded data as {file_format.upper()}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")
