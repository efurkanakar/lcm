from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QEvent, QSortFilterProxyModel, QSettings, QStringListModel, QTimer
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit, QLabel,
    QPushButton, QComboBox, QCheckBox, QFileDialog, QDialogButtonBox,
    QMessageBox, QDialog, QTableView, QHeaderView, QSizePolicy
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from lcm_app.controller import TESSController
from lcm_app.config import load_download_dir, save_download_dir
from lcm_app.common.ui_utils import show_info
from lcm_app.tess.threads import (
    TESSFutureThread,
    TESSDownloadThread,
    TESSSectorSearchThread,
    TESSDataFetchThread,
    tess_future_sectors
)
from tess_util import (
    search_tess_sector, get_tess_sector_data,
    phase_fold_manually
)

STAR_HISTORY_LIMIT = 50


class TessQueryMixin(QWidget):
    """
    Mixin that builds and manages the Query tab of the TESS window.
    Expects inheriting class to provide:
      - screen_width, screen_height
      - star_name, sector_numbers
      - _info(text)
      - ctrl (TESSController) and lc state properties
    """

    def _init_query_tab(self):
        query_tab = QWidget()
        self.query_layout = QVBoxLayout(query_tab)

        self.star_name_input = QLineEdit()
        self.star_name_input.setPlaceholderText("Enter star name")
        self.star_history = QStringListModel(QSettings("lcm", "lcm").value("star_history", [], type=list))
        self.star_name_input.installEventFilter(self)
        self.star_ghost = QLabel(self.star_name_input)
        self.star_ghost.setTextFormat(Qt.RichText)
        self.star_ghost.setStyleSheet("color: gray; background: transparent;")
        self.star_ghost.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._suggestion = None
        self._tab_matches = None
        self.star_name_input.textChanged.connect(self._update_star_ghost)
        self.star_name_input.textEdited.connect(lambda: setattr(self, "_tab_matches", None))

        self.tess_search_btn = QPushButton("Search Sectors")
        self.tess_search_btn.clicked.connect(self.search_tess_sectors)

        self.display_data_btn = QPushButton("Query More")
        self.display_data_btn.clicked.connect(self.display_tess_sector_data)

        self.future_btn = QPushButton("Upcoming Sectors")
        self.future_btn.clicked.connect(self.show_future_sectors)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.tess_search_btn)
        button_layout.addWidget(self.display_data_btn)
        button_layout.addWidget(self.future_btn)

        self.query_layout.addWidget(self.star_name_input)
        # Download path controls intentionally omitted in Query tab
        self.query_layout.addLayout(button_layout)

        dropdown_layout = QHBoxLayout()
        self.sector_combo = QComboBox()
        self.sector_combo.setEditable(True)
        self.sector_combo.lineEdit().setPlaceholderText("Enter sector number")

        self.exptime_combo = QComboBox()
        self.exptime_combo.addItems(["120", "20"])

        self.flux_type_combo = QComboBox()
        self.flux_type_combo.addItems(["SAP FLUX", "PDCSAP FLUX"])

        dropdown_layout.addWidget(self.sector_combo)
        dropdown_layout.addWidget(self.exptime_combo)
        dropdown_layout.addWidget(self.flux_type_combo)

        self.query_layout.addLayout(dropdown_layout)

        download_layout = QHBoxLayout()
        self.download_data_btn = QPushButton("Download and Plot Data")
        self.download_data_btn.clicked.connect(self.download_and_plot_data)
        self.quality_nan_checkbox = QCheckBox("Filter Data")
        self.quality_nan_checkbox.setToolTip("Filter out data where QUALITY == 0 and drop NaN values.")
        self.quality_nan_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.quality_nan_checkbox.setChecked(True)
        download_layout.addWidget(self.download_data_btn)
        download_layout.addWidget(self.quality_nan_checkbox)
        self.query_layout.addLayout(download_layout)

        light_curve_group = QGroupBox("Light Curve")
        light_curve_layout = QVBoxLayout()
        self.figure_query = Figure()
        self.canvas = FigureCanvas(self.figure_query)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.query_error_checkbox = QCheckBox("Display with Errors")
        self.query_error_checkbox.setChecked(True)
        self.query_error_checkbox.stateChanged.connect(self.update_query_plot)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addWidget(self.query_error_checkbox)
        toolbar_layout.setAlignment(Qt.AlignLeft)
        light_curve_layout.addWidget(self.canvas)
        light_curve_layout.addLayout(toolbar_layout)
        light_curve_group.setLayout(light_curve_layout)
        self.query_layout.addWidget(light_curve_group)

        query_button_layout = QHBoxLayout()
        self.detach_query_plot_button = QPushButton("Detached")
        self.detach_query_plot_button.clicked.connect(self.detach_query_plot)
        query_button_layout.addWidget(self.detach_query_plot_button)
        self.save_query_data_button = QPushButton("Save Data")
        self.save_query_data_button.clicked.connect(self.save_query_data)
        query_button_layout.addWidget(self.save_query_data_button)
        self.query_layout.addLayout(query_button_layout)

        return query_tab

    def _names_containing(self, text):
        """Saved star names containing the text; names starting with it come first, then most recent first."""
        text = text.strip().lower()
        names = [n for n in self.star_history.stringList() if text and text in n.lower()]
        return sorted(names, key=lambda n: not n.lower().startswith(text))

    def _update_star_ghost(self, text):
        """Show the best saved name in gray after the typed text, with the matching part in bold."""
        matches = self._names_containing(text)
        self._suggestion = matches[0] if matches and matches[0].lower() != text.strip().lower() else None
        if self._suggestion is None:
            self.star_ghost.clear()
            return
        name, typed = self._suggestion, text.strip()
        if name.lower().startswith(text.lower()):
            self.star_ghost.setText(escape(name[len(text):]).replace(" ", "&nbsp;"))
        else:
            i = name.lower().index(typed.lower())
            parts = escape(name[:i]), escape(name[i:i + len(typed)]), escape(name[i + len(typed):])
            self.star_ghost.setText("&nbsp;&nbsp;{}<b>{}</b>{}".format(*parts).replace(" ", "&nbsp;"))
        QTimer.singleShot(0, self._place_star_ghost)

    def _place_star_ghost(self):
        """Put the gray suggestion right after the typed text inside the star name box."""
        box, ghost = self.star_name_input, self.star_ghost
        ghost.adjustSize()
        ghost.move(box.cursorRect().center().x() + 1, (box.height() - ghost.height()) // 2)
        ghost.setVisible(box.cursorPosition() == len(box.text()))

    def eventFilter(self, obj, event):
        """Star name box: Right/End/Tab take the gray suggestion, Esc hides it, Up fills the last star."""
        if obj is not self.star_name_input:
            return super().eventFilter(obj, event)
        if event.type() in (QEvent.Resize, QEvent.KeyRelease, QEvent.MouseButtonRelease) and self._suggestion:
            self._place_star_ghost()
        if event.type() != QEvent.KeyPress:
            return super().eventFilter(obj, event)

        key, at_end = event.key(), obj.cursorPosition() == len(obj.text())

        if key == Qt.Key_Up and not obj.text() and self.star_history.rowCount():
            obj.setText(self.star_history.stringList()[0])
            return True

        if key in (Qt.Key_Right, Qt.Key_End) and self._suggestion and at_end:
            obj.setText(self._suggestion)
            return True

        if key == Qt.Key_Escape and self._suggestion:
            self._suggestion = None
            self.star_ghost.clear()
            return True

        if key == Qt.Key_Tab:
            if self._tab_matches is None:
                self._tab_matches = self._names_containing(obj.text())
                self._tab_index = -1
            if self._tab_matches:
                self._tab_index = (self._tab_index + 1) % len(self._tab_matches)
                obj.setText(self._tab_matches[self._tab_index])
                return True

        return super().eventFilter(obj, event)

    def _remember_star_name(self, name):
        """Move a successfully searched star name to the top of the saved history."""
        names = [n for n in self.star_history.stringList() if n.lower() != name.lower()]
        names = [name] + names[:STAR_HISTORY_LIMIT - 1]
        self.star_history.setStringList(names)
        QSettings("lcm", "lcm").setValue("star_history", names)

    def update_query_plot(self):
        self.plot_data_check(self.figure_query, self.canvas, self.query_error_checkbox.isChecked())

    def search_tess_sectors(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return

        self.tess_search_btn.setEnabled(False)
        self.tess_search_btn.setText("Searching…")
        self.tess_search_btn.setStyleSheet("color:red;")

        self.sector_thread = TESSSectorSearchThread(star_name)
        self.sector_thread.resultReady.connect(self._on_sector_search_finished)
        self.sector_thread.error.connect(self._on_sector_search_error)
        self.sector_thread.finished.connect(
            lambda: (self.tess_search_btn.setEnabled(True),
                     self.tess_search_btn.setText("Search Sectors"),
                     self.tess_search_btn.setStyleSheet("color:black;"))
        )
        self.sector_thread.start()

    def _on_sector_search_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def _on_sector_search_finished(self, msg, sector_numbers):
        star = self.star_name_input.text().strip() or self.star_name
        try:
            mapping = self.tess_all_sectors(star)
            html = self.html_tess_sector_message(sector_numbers, mapping)
        except ValueError:
            msg = (
                f"<div style='font-family:Helvetica, Arial, sans-serif; "
                f"color:#d32f2f; font-weight:bold; font-size:14px;'>"
                f"Not found!</div>"
                f"<div style='font-family:Helvetica, Arial, sans-serif; "
                f"color:#333; margin-top:4px;'>"
                f"No TESS sectors could be found for {star}.</div>"
            )
            QMessageBox(self, windowTitle="Not Found", textFormat=Qt.RichText, text=msg).exec_()
            return
        except Exception:
            html = self.html_tess_sector_message(sector_numbers)

        QMessageBox(
            self,
            windowTitle="TESS Search Result",
            textFormat=Qt.RichText,
            text=html
        ).exec_()

        if sector_numbers:
            self.sector_numbers = sector_numbers
            self.sector_combo.clear()
            self.sector_combo.addItems(map(str, sector_numbers))
            self.sector_combo.setCurrentIndex(len(sector_numbers) - 1)

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

        def restore():
            self.future_btn.setEnabled(True)
            self.future_btn.setText("Upcoming Sectors")
            self.future_btn.setStyleSheet("")

        self.future_thread.finished.connect(restore)
        self.future_thread.start()

    def _on_future_ok(self, rows):
        if not rows:
            self._info("No upcoming TESS observations scheduled for this target.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Upcoming TESS Sectors")
        dlg.setWindowIcon(QIcon("images/star.ico"))
        outer = QVBoxLayout(dlg)

        model = QStandardItemModel(len(rows), 2)
        model.setHorizontalHeaderLabels(["Sector", "Sector Midpoint"])
        for r, (sec, mid) in enumerate(rows):
            model.setItem(r, 0, QStandardItem(str(sec)))
            model.setItem(r, 1, QStandardItem(mid))
        table = QTableView()
        table.setModel(model)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableView.NoEditTriggers)

        outer.addWidget(table)
        dlg.resize(300, 200)
        dlg.exec_()

    def download_and_plot_data(self):
        star_name = self.star_name_input.text().strip() or self.star_name
        sector_txt = self.sector_combo.currentText()
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return
        if not sector_txt.isdigit():
            QMessageBox.warning(self, "Warning", "Sector number must be an integer.")
            return

        sector = int(sector_txt)
        exptime = int(self.exptime_combo.currentText())
        flux_type = self.flux_type_combo.currentText().replace(" ", "_")
        self.flux_type = flux_type
        apply_filter = self.quality_nan_checkbox.isChecked()

        if self._download_dir is not None:
            save_download_dir(str(self._download_dir))

        self.download_data_btn.setEnabled(False)
        self.download_data_btn.setText("Downloading and plotting…")
        self.download_data_btn.setStyleSheet("color:red;")

        self.tess_thread = TESSDownloadThread(
            star_name, sector, exptime, flux_type, apply_filter
        )
        self.tess_thread.resultReady.connect(self._on_tess_finished)
        self.tess_thread.error.connect(self._on_tess_error)
        self.tess_thread.finished.connect(
            lambda: (
                self.download_data_btn.setEnabled(True),
                self.download_data_btn.setText("Download and Plot Data"),
                self.download_data_btn.setStyleSheet("color:black;")
            )
        )
        self.tess_thread.start()

    def save_query_data(self):
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
        add_2457000_checkbox.setChecked(True)
        layout.addWidget(add_2457000_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        file_format = file_format_combo.currentText().lower()
        time_data = self.time + 2457000 if add_2457000_checkbox.isChecked() else self.time
        data = pd.DataFrame({"Time": time_data, "Flux": self.flux, "Error": self.flux_err})

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
            QMessageBox.critical(self, "Error", f"An error occurred while saving query data as {file_format.upper()}: {e}")

    def detach_query_plot(self):
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

    # Placeholder to keep interface compatible; phase/outlier/detrend will be added via other mixins.
    def _on_tess_finished(self, time, flux, flux_err, quality):
        pass

    def _on_tess_error(self, message):
        QMessageBox.critical(self, "Error", message)
