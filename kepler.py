import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit, QLabel, QTabWidget, QSizePolicy, QApplication,
    QPushButton, QComboBox, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


from astropy.io import fits
from astroquery.mast import Observations
from lcm_app.config import load_download_dir

from scipy.special import erfcinv
from sklearn.mixture import GaussianMixture
from tess_util import phase_fold_manually
from astropy.timeseries import LombScargle, BoxLeastSquares
from scipy.optimize import minimize_scalar, curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline


from itertools import zip_longest


def html_campaign_table(short_c, long_c):
    rows = []
    for s, l in zip_longest(sorted(short_c), sorted(long_c), fillvalue="&nbsp;"):
        rows.append(
            "<tr>"
            f"<td style='padding:2px 12px;text-align:center;'>{s}</td>"
            f"<td style='padding:2px 12px;text-align:center;'>{l}</td>"
            "</tr>"
        )

    return (
        "<div style='text-align:center;font-family:monospace;line-height:1.3;'>"
        "<div style='color:#098a00;font-weight:bold;margin:0 0 6px 0;'>"
        "K2 data found for this star."
        "</div>"
        "<div style='font-weight:bold;margin-bottom:6px;'>Available&nbsp;Campaigns</div>"

        "<table style='margin:6px auto 0;border-collapse:collapse;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:2px 12px;border-right:1px solid #999;"
        "border-bottom:1px solid #999;text-align:center;'>Short</th>"
        "<th style='padding:2px 12px;border-bottom:1px solid #999;text-align:center;'>Long</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def html_table(short_q, long_q):
    rows = []
    for s, l in zip_longest(sorted(short_q), sorted(long_q), fillvalue="&nbsp;"):
        rows.append(
            "<tr>"
            f"<td style='padding:2px 12px;text-align:center;'>{s}</td>"
            f"<td style='padding:2px 12px;text-align:center;'>{l}</td>"
            "</tr>"
        )

    return (
        "<div style='text-align:center;font-family:monospace;line-height:1.3;'>"
        "<div style='color:#098a00;font-weight:bold;margin:0 0 6px 0;'>"
        "Kepler data found!"
        "</div>"
        "<div style='font-weight:bold;margin-bottom:6px;'>Available&nbsp;Quarters</div>"

        "<table style='margin:6px auto 0;border-collapse:collapse;'>"
        "<thead>"
        "<tr>"

        "<th style='padding:2px 12px;border-right:1px solid #999;"
        "border-bottom:1px solid #999;text-align:center;'>Short</th>"

        "<th style='padding:2px 12px;border-bottom:1px solid #999;text-align:center;'>Long</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )

def html_notfound(survey, star):
    return (
        "<div style='text-align:center;font-family:monospace;line-height:1.3;'>"
        f"<div style='color:#FF0000;'>No {survey} data found for <span>{star}</span>.</div>"
        "</div>"
    )



number_re = re.compile(r'\d+')
campaign_re = re.compile(r"C(\d+)")


def _add_placeholder(combo: QComboBox, text: str):
    """Add a grey, disabled first item that acts like a placeholder."""
    combo.addItem(text)
    combo.setCurrentIndex(0)
    combo.model().item(0).setEnabled(False)


class KeplerSearchThread(QThread):
    resultReady = pyqtSignal(object)

    def __init__(self, star_name):
        super().__init__()
        self.star_name = star_name

    def run(self):
        try:
            result = Observations.query_criteria(
                obs_collection="Kepler",
                objectname=self.star_name,
                dataproduct_type="timeseries",
                radius=0
            )
        except Exception as e:
            result = e
        self.resultReady.emit(result)

class K2SearchThread(QThread):
    resultReady = pyqtSignal(object)

    def __init__(self, star_name):
        super().__init__()
        self.star_name = star_name

    def run(self):
        try:
            result = Observations.query_criteria(
                obs_collection="K2",
                objectname=self.star_name,
                dataproduct_type="timeseries",
                radius=0
            )
        except Exception as e:
            result = e
        self.resultReady.emit(result)


class KeplerDownloadThread(QThread):
    resultReady = pyqtSignal(object, object, object, object)  # t, f, e, extra

    def __init__(self, star, qres, cadence, quarter, flux_type, apply_quality_filter, parent=None):
        super().__init__(parent)
        self.star = star
        self.qres = qres
        self.cad = cadence       # "SLC" | "LLC"
        self.quarter = quarter
        self.ftype = flux_type   # e.g. "SAP_FLUX"
        self.qfilt = apply_quality_filter

    def run(self):
        try:
            # 1) Sorgu
            if self.qres is None:
                self.qres = Observations.query_criteria(
                    obs_collection="Kepler",
                    objectname=self.star,
                    dataproduct_type="timeseries",
                    radius=0
                )
                if len(self.qres) == 0:
                    raise RuntimeError(f"No Kepler data found for {self.star!r}")

            # 2) Ürünleri al, cadence filtresi
            products = Observations.get_product_list(self.qres)
            filtered = Observations.filter_products(
                products, productSubGroupDescription=self.cad
            )

            # Quarter eşleştirmesi
            mprods = []
            for p in filtered:
                desc = p["description"]
                # description sonundaki sayıya bakıyoruz
                digits = "".join(filter(str.isdigit, desc.split("_")[-1]))
                if digits and int(digits) == self.quarter:
                    mprods.append(p)

            if not mprods:
                raise RuntimeError(f"Quarter {self.quarter} – cadence {self.cad} yok.")

            # 3) İndirme + okuma
            all_t, all_f, all_e = [], [], []
            for p in mprods:
                dl_dir = load_download_dir()
                kwargs = {"mrp_only": True}
                if dl_dir is not None:
                    kwargs["download_dir"] = str(dl_dir)
                dl_tbl = Observations.download_products(p, **kwargs)
                path = dl_tbl["Local Path"][0]
                with fits.open(path) as hdul:
                    t = hdul[1].data["TIME"]
                    f = hdul[1].data[self.ftype]
                    e = hdul[1].data[f"{self.ftype}_ERR"]
                    q = hdul[1].data["SAP_QUALITY"]

                    if self.qfilt:
                        mask = (q == 0) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
                    else:
                        mask = np.isfinite(t) & np.isfinite(f)

                    all_t.append(t[mask])
                    all_f.append(f[mask])
                    all_e.append(e[mask])

            # 4) Birleştir & sırala
            t = np.concatenate(all_t)
            f = np.concatenate(all_f)
            e = np.concatenate(all_e)
            idx = np.argsort(t)

            # 5) Sonucu emit et
            self.resultReady.emit(t[idx], f[idx], e[idx], self.qres)

        except Exception as exc:
            self.resultReady.emit(None, None, None, exc)

class KeplerQueryTab(QWidget):
    def __init__(self, screen_width, screen_height, star_name=None, parent=None):
        super().__init__(parent)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.star_name = star_name

        self.query_result = None
        self.selected_cadence = None
        self.time = None
        self.flux = None
        self.flux_err = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.init_ui()
    def init_ui(self):
        """Arayüzü kurar – Download butonu durumunu dinamikleştirir."""
        main_layout = QVBoxLayout(self)

        # 1) Üst arama ve seçim kutuları
        top_control = QGroupBox("Data Query Controls")
        top_control_layout = QVBoxLayout(top_control)

        # 1a) Arama satırı
        search_row = QHBoxLayout()
        self.star_name_input = QLineEdit(placeholderText="Object Name")
        if self.star_name:
            self.star_name_input.setText(self.star_name)
        search_row.addWidget(self.star_name_input)

        self.kepler_search_btn = QPushButton("Search Kepler Data", minimumWidth=300)
        self.kepler_search_btn.clicked.connect(self.check_kepler_data)
        search_row.addWidget(self.kepler_search_btn)

        self.quality_nan_checkbox = QCheckBox("Filter Data", checked=True)
        search_row.addWidget(self.quality_nan_checkbox)
        top_control_layout.addLayout(search_row)

        # 1b) Cadence / Quarter / Flux-type satırı
        combo_row = QHBoxLayout()

        self.cadence_combo = QComboBox()
        self.cadence_combo.setEditable(True)
        self.cadence_combo.lineEdit().setPlaceholderText("Cadence (SLC = short, LLC = long)")
        combo_row.addWidget(self.cadence_combo)

        self.quarter_combo = QComboBox()
        self.quarter_combo.setEditable(True)
        self.quarter_combo.lineEdit().setPlaceholderText("Quarter (e.g. 5)")
        combo_row.addWidget(self.quarter_combo)

        self.flux_type_combo = QComboBox()
        self.flux_type_combo.addItems(["SAP FLUX", "PDCSAP FLUX"])
        combo_row.addWidget(self.flux_type_combo)

        top_control_layout.addLayout(combo_row)
        top_control.setFixedHeight(100)
        main_layout.addWidget(top_control)

        # 2) Download düğmesi
        self.download_and_plot_btn = QPushButton("Download and Plot Data")
        self.download_and_plot_btn.setEnabled(False)        # başlangıçta kilitli
        self.download_and_plot_btn.clicked.connect(self.download_and_plot)
        main_layout.addWidget(self.download_and_plot_btn)
        # ------------------------------------------------------------------------
        # 3) Işık eğrisi alanı  +  Display-with-Errors kutucuğu
        # ------------------------------------------------------------------------
        lc_group = QGroupBox("Light Curve")
        lc_layout = QVBoxLayout(lc_group)

        lc_layout.addWidget(self.canvas)

        # alt satır: Matplotlib toolbar + checkbox
        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.toolbar)

        self.error_checkbox = QCheckBox("Display with Errors", checked=True)
        self.error_checkbox.stateChanged.connect(self.update_query_plot)
        toolbar_row.addWidget(self.error_checkbox)
        toolbar_row.setAlignment(Qt.AlignLeft)

        lc_layout.addLayout(toolbar_row)

        lc_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(lc_group, stretch=1)

        # ------------------------------------------------------------------------
        # 4) Alt butonlar
        # ------------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        self.detach_query_plot_button = QPushButton("Detached")
        self.detach_query_plot_button.clicked.connect(self.detach_query_plot)
        bottom_row.addWidget(self.detach_query_plot_button)

        self.save_query_data_button = QPushButton("Save Data")
        self.save_query_data_button.clicked.connect(self.save_query_data)
        bottom_row.addWidget(self.save_query_data_button)

        # ── Sinyalleri buton kontrolüne bağla ────────────────────────────────
        self.star_name_input.textChanged.connect(self._update_download_button_state)
        self.cadence_combo.lineEdit().textChanged.connect(self._update_download_button_state)
        self.quarter_combo.lineEdit().textChanged.connect(self._update_download_button_state)
        # combobox seçimi değiştiğinde de tetiklensin
        self.cadence_combo.currentIndexChanged.connect(self._update_download_button_state)
        self.quarter_combo.currentIndexChanged.connect(self._update_download_button_state)

        # ilk kontrol
        self._update_download_button_state()


        main_layout.addLayout(bottom_row)

    def _update_download_button_state(self):
        """Girdi kutularına bakarak Download düğmesini aç/kapat."""
        star_ok     = bool(self.star_name_input.text().strip())
        cadence_ok  = bool(self.cadence_combo.currentText().strip())
        quarter_ok  = bool(self.quarter_combo.currentText().strip())
        self.download_and_plot_btn.setEnabled(star_ok and cadence_ok and quarter_ok)

    def update_query_plot(self):
        """Display-with-Errors kutusu değiştiğinde grafiği yeniden çizer."""
        ax = self.figure.gca()
        ax.clear()

        if self.time is None:  # henüz veri yok
            self.canvas.draw()
            return

        if self.error_checkbox.isChecked() and self.flux_err is not None:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err,
                        fmt=".", markersize=5, capsize=1, elinewidth=1.5,
                        color="dodgerblue")
        else:
            ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)

        ax.set_xlabel("Time – 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")

        # ➜ Daha önce kaydedilen başlık varsa kullan, yoksa genel bir başlık yaz
        ax.set_title(getattr(self, "_plot_title", "Light Curve"))

        self.figure.tight_layout()
        self.canvas.draw()

    def check_kepler_data(self):
        star_name = self.star_name_input.text().strip()
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a object name.")
            return
        self.kepler_search_btn.setText("Searching...")
        self.kepler_search_btn.setEnabled(False)
        self.kepler_search_btn.setStyleSheet("QPushButton:disabled { color: red; }")
        QApplication.processEvents()
        self.search_thread = KeplerSearchThread(star_name)
        self.search_thread.resultReady.connect(self.on_search_finished)
        self.search_thread.start()

    def on_search_finished(self, result):
        # --- arama düğmesini eski hâline getir ---------------------------------
        self.kepler_search_btn.setEnabled(True)
        self.kepler_search_btn.setText("Search Kepler Data")

        star_txt = self.star_name_input.text().strip() or "this star"

        # --- hata / hiç veri yok durumları --------------------------------------
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error", f"An error occurred: {result}")
            return
        if len(result) == 0:
            QMessageBox(self, windowTitle="Search Result", textFormat=Qt.RichText,
                        text=html_notfound("Kepler", star_txt)).exec_()
            return

        # ------------------------------------------------------------------------
        # 1) Ürünleri al, kısa-uzun cadence kümelerini hazırla
        # ------------------------------------------------------------------------
        self.query_result = result
        products = Observations.get_product_list(result)
        filtered_slc = Observations.filter_products(products, productSubGroupDescription="SLC")
        filtered_llc = Observations.filter_products(products, productSubGroupDescription="LLC")

        # Quarter kümelerini çıkar
        self._slc_quarters = {int(''.join(filter(str.isdigit, p["description"].split("_")[-1])))
                              for p in filtered_slc
                              if ''.join(filter(str.isdigit, p["description"].split("_")[-1]))}
        self._llc_quarters = {int(''.join(filter(str.isdigit, p["description"].split("_")[-1])))
                              for p in filtered_llc
                              if ''.join(filter(str.isdigit, p["description"].split("_")[-1]))}

        # ------------------------------------------------------------------------
        # 2) Bilgilendirme kutusu
        # ------------------------------------------------------------------------
        QMessageBox(self, windowTitle="Search Result", textFormat=Qt.RichText,
                    text=html_table(self._slc_quarters, self._llc_quarters)).exec_()

        # ------------------------------------------------------------------------
        # 3) Cadence combobox’ını sinyal göndermeden doldur
        # ------------------------------------------------------------------------
        # 3) Cadence combobox’ını sinyal göndermeden doldur
        options = []
        if filtered_slc:
            options.append("Short Cadence")
        if filtered_llc:
            options.append("Long Cadence")

        self.cadence_combo.blockSignals(True)
        self.cadence_combo.clear()

        if options:  # en az bir cadence bulundu
            self.cadence_combo.addItems(options)
            self.cadence_combo.setCurrentIndex(0)
            self.cadence_combo.setEnabled(True)  # <-- EKLENDİ
            self.quarter_combo.setEnabled(True)
        else:  # hiç cadence yok
            _add_placeholder(self.cadence_combo, "─ Cadence ─")
            self.cadence_combo.setEnabled(False)  # <-- EKLENDİ
            self.quarter_combo.clear()
            _add_placeholder(self.quarter_combo, "─ Quarter ─")
            self.quarter_combo.setEnabled(False)
            self.download_and_plot_btn.setEnabled(False)

        self.cadence_combo.blockSignals(False)

        # ------------------------------------------------------------------------
        # 4) Quarter listesi hemen dolsun
        # ------------------------------------------------------------------------
        self.update_quarter_list()

    def update_quarter_list(self):
        if not self._slc_quarters and not self._llc_quarters:
            return  # arama henüz yapılmamışsa

        cadence_option = self.cadence_combo.currentText()
        self.selected_cadence = "SLC" if cadence_option == "Short Cadence" else "LLC"

        # Ağ çağrısı yerine önceden hazırlanmış kümeyi kullan
        quarters = self._slc_quarters if self.selected_cadence == "SLC" else self._llc_quarters

        if quarters:
            self.flux_type_combo.clear()
            self.flux_type_combo.addItems(["SAP FLUX", "PDCSAP FLUX"])
            self.flux_type_combo.setCurrentIndex(0)
            self.flux_type_combo.setEnabled(True)
            self.download_and_plot_btn.setEnabled(True)

            self.quarter_combo.clear()
            for q in sorted(quarters):
                self.quarter_combo.addItem(str(q))
            self.download_and_plot_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Warning", f"No {cadence_option} data found.")
            self.download_and_plot_btn.setEnabled(False)

        self._update_download_button_state()

    def download_and_plot(self):
        """Kullanıcı 'Download and Plot Data' düğmesine bastığında çalışır."""
        # ── 1) Girdiler
        star = self.star_name_input.text().strip()
        cad = self.cadence_combo.currentText().strip().lower()
        qtxt = self.quarter_combo.currentText().strip()

        if not star:
            QMessageBox.warning(self, "Warning", "Object name is empty.");
            return
        if not cad or not qtxt:
            QMessageBox.warning(self, "Warning",
                                "Cadence ve Quarter kutularını doldurun.");
            return

        self.selected_cadence = "SLC" if ("short" in cad or "slc" in cad) else "LLC"
        try:
            quarter = int(qtxt)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Quarter tam sayı olmalı.");
            return

        # ── 2) Butonu & görünümü anında güncelle
        self.download_and_plot_btn.setEnabled(False)
        self.download_and_plot_btn.setText("Downloading and Plotting…")
        self.download_and_plot_btn.setStyleSheet(
            "QPushButton {color:red;} QPushButton:disabled{color:red;}")
        QApplication.processEvents()  # repaint

        # ── 3) Arka-plan iş parçasını başlat
        flux_type = self.flux_type_combo.currentText().replace(" ", "_")
        self._dlThread = KeplerDownloadThread(
            star, self.query_result,
            self.selected_cadence, quarter,
            flux_type,
            self.quality_nan_checkbox.isChecked()
        )
        self._dlThread.resultReady.connect(self._on_download_finished,
                                           Qt.QueuedConnection)
        self._dlThread.start()

    def _on_download_finished(self, t, f, e, extra):
        """İndirme w/işleme tamamlandığında GUI güncellemesi."""
        # Butonu eski hâline getir
        self.download_and_plot_btn.setEnabled(True)
        self.download_and_plot_btn.setText("Download and Plot Data")
        self.download_and_plot_btn.setStyleSheet("QPushButton {color:black;}")

        # Hata mı geldi?
        if any(x is None for x in (t, f, e)):
            QMessageBox.critical(self, "Error", f"{extra}")
            return

        # Başarılı – verileri sakla & çiz
        self.time, self.flux, self.flux_err = t, f, e
        self.query_result = extra  # query_result cache’i (olay yeni sorguysa)

        flux_type = self.flux_type_combo.currentText().replace(" ", "_")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.errorbar(self.time, self.flux, yerr=self.flux_err,
                    fmt=".", markersize=5, capsize=1,
                    elinewidth=1.5, color="dodgerblue")
        ax.set_xlabel("Time − 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")
        title = (f"{flux_type.replace('_', ' ')} Light Curve  -  "
                 f"Q{self.quarter_combo.currentText()}  -  {self.selected_cadence}")
        ax.set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()

        # Ortak veri havuzuna gönder
        main_win = self.window()
        if hasattr(main_win, "updateSharedData"):
            main_win.updateSharedData(self.time, self.flux, self.flux_err)
    def detach_query_plot(self):
        if self.time is None or self.flux is None or self.flux_err is None:
            QMessageBox.warning(self, "Warning", "No data available for plotting.")
            return
        fig, ax = plt.subplots()
        ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                    markersize=5, capsize=1, elinewidth=1.5)
        ax.set_xlabel("Time - 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")
        ax.set_title("Detached Light Curve - Query Data")
        fig.tight_layout()
        fig.show()

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

            add_2457000_checkbox = QCheckBox("Add 2454833 to Time")
            add_2457000_checkbox.setChecked(True)
            layout.addWidget(add_2457000_checkbox)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec_() != QDialog.Accepted:
                return

            file_format = file_format_combo.currentText().lower()

            time_data = self.time + 2454833 if add_2457000_checkbox.isChecked() else self.time
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

                QMessageBox.information(self, "Data Saved", f"Query data saved successfully to {file_name}.")

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"An error occurred while saving query data as {file_format.upper()}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

class KeplerOutlierTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.time = None
        self.flux = None
        self.flux_err = None

        self.original_time = None
        self.original_flux = None
        self.original_flux_err = None

        self.lower_bound_line = None
        self.upper_bound_line = None
        self.boxplot_lower_line = None
        self.boxplot_upper_line = None
        self.chauvenet_lower_line = None
        self.chauvenet_upper_line = None

        self.selector = None
        self.selected_start_time = None
        self.selected_end_time = None
        self.selected_start_flux = None
        self.selected_end_flux = None

        self.selector = None
        self.region_patch = None
        self._fixed_xlim = None
        self._fixed_ylim = None

        self.figure_outliers = Figure()
        self.canvas_outliers = FigureCanvas(self.figure_outliers)
        self.toolbar_outliers = NavigationToolbar(self.canvas_outliers, self)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        manual_group = QGroupBox("Manual Detection")
        manual_layout = QVBoxLayout()
        self.select_region_button = QPushButton("Select Region")
        self.select_region_button.clicked.connect(self.enable_region_selection)
        manual_layout.addWidget(self.select_region_button)
        self.remove_selected_button = QPushButton("Remove Region")
        self.remove_selected_button.clicked.connect(self.remove_selected_interval)
        manual_layout.addWidget(self.remove_selected_button)
        manual_layout.addStretch()
        manual_group.setLayout(manual_layout)
        manual_group.setFixedHeight(100)

        automated_group = QGroupBox("Automated Detection")
        automated_layout = QVBoxLayout()
        detection_method_layout = QHBoxLayout()
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["Sigma Clipping", "Boxplot", "Chauvenet"])
        detection_method_layout.addWidget(self.outlier_method_combo)
        self.sigma_value_spinbox = QSpinBox()
        self.sigma_value_spinbox.setRange(1, 10)
        self.sigma_value_spinbox.setValue(3)
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
        automated_group.setLayout(automated_layout)
        automated_group.setFixedHeight(100)

        normalization_group = QGroupBox("Normalization")
        normalization_layout = QVBoxLayout()
        self.apply_normalization_button = QPushButton("Apply Normalization")
        self.apply_normalization_button.clicked.connect(self.apply_normalization)
        normalization_layout.addWidget(self.apply_normalization_button)
        normalization_group.setLayout(normalization_layout)
        normalization_group.setFixedHeight(100)
        top_layout.addWidget(manual_group)
        top_layout.addWidget(automated_group)
        top_layout.addWidget(normalization_group)
        layout.addLayout(top_layout)

        self.restore_original_data_button = QPushButton("Restore Original Data")
        self.restore_original_data_button.clicked.connect(self.restore_original_data)
        layout.addWidget(self.restore_original_data_button)

        outlier_group = QGroupBox("Light Curve")
        outlier_layout = QVBoxLayout()
        outlier_layout.addWidget(self.canvas_outliers)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.toolbar_outliers)
        self.outlier_error_checkbox = QCheckBox("Display with Errors")
        self.outlier_error_checkbox.setChecked(True)
        self.outlier_error_checkbox.stateChanged.connect(self.plot_data_in_outliers)
        toolbar_layout.addWidget(self.outlier_error_checkbox)
        toolbar_layout.setAlignment(Qt.AlignLeft)
        outlier_layout.addLayout(toolbar_layout)
        outlier_group.setLayout(outlier_layout)
        layout.addWidget(outlier_group)

        bottom_layout = QHBoxLayout()
        self.detach_plot_button = QPushButton("Detached")
        self.detach_plot_button.clicked.connect(self.detach_plot)
        bottom_layout.addWidget(self.detach_plot_button)
        self.save_cleaned_data_button = QPushButton("Save Cleaned Data")
        self.save_cleaned_data_button.clicked.connect(self.save_cleaned_data)
        bottom_layout.addWidget(self.save_cleaned_data_button)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)


    def _lock_y_format(self, ax, fmt="%.2f"):
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))


    def showEvent(self, event):
        super().showEvent(event)

        try:
            if self.figure_outliers is not None and self.figure_outliers.axes:
                self.figure_outliers.tight_layout()
                self.canvas_outliers.draw_idle()
        except Exception:
            pass

    def setData(self, time, flux, flux_err):
        self.time = time.copy()
        self.flux = flux.copy()
        self.flux_err = flux_err.copy()
        if self.original_time is None:
            self.original_time = self.time.copy()
            self.original_flux = self.flux.copy()
            self.original_flux_err = self.flux_err.copy()
        self.plot_data_in_outliers()

    def plot_data_in_outliers(self):
        ax = self.figure_outliers.gca()
        ax.clear()
        if self.time is not None and self.flux is not None:
            if self.outlier_error_checkbox.isChecked() and self.flux_err is not None:
                ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                            markersize=5, capsize=1, elinewidth=1.5, ls="none")
            else:
                ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)
            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_title("Outlier Tab - Light Curve")
        self.figure_outliers.tight_layout()
        self._lock_y_format(ax, "%.2f")  # << eklenen satır
        self.canvas_outliers.draw()

    def on_outlier_data_changed(self):
        main_win = self.window()
        if hasattr(main_win, 'updateSharedData'):
            main_win.updateSharedData(self.time, self.flux, self.flux_err)

    def enable_region_selection(self):
        if self.selector:
            self.selector.set_active(False)
            self.selector = None

        ax = self.figure_outliers.gca()

        self._fixed_xlim = ax.get_xlim()
        self._fixed_ylim = ax.get_ylim()

        self.selector = RectangleSelector(
            ax, self.on_select, useblit=True, interactive=True,
            button=[1], minspanx=0.1, minspany=0.1, spancoords='data'
        )
        QMessageBox.information(self, "Region Selection",
                                "Drag to select the region you want to remove.")

    def on_select(self, eclick, erelease):
        self.selected_start_time = eclick.xdata
        self.selected_end_time = erelease.xdata
        self.selected_start_flux = eclick.ydata
        self.selected_end_flux = erelease.ydata

        ax = self.figure_outliers.gca()

        if self.region_patch is not None:
            try:
                self.region_patch.remove()
            except Exception:
                pass
            self.region_patch = None

        x1, x2 = sorted([self.selected_start_time, self.selected_end_time])
        y1, y2 = sorted([self.selected_start_flux, self.selected_end_flux])

        self.region_patch = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor='yellow', edgecolor='none', alpha=0.3, zorder=10
        )
        ax.add_patch(self.region_patch)

        if self._fixed_xlim is not None and self._fixed_ylim is not None:
            ax.set_xlim(self._fixed_xlim)
            ax.set_ylim(self._fixed_ylim)

        self.canvas_outliers.draw_idle()

    def remove_selected_interval(self):
        try:
            if (self.selected_start_time is None or self.selected_end_time is None or
                    self.selected_start_flux is None or self.selected_end_flux is None):
                QMessageBox.warning(self, "Warning", "No valid region selected.")
                return

            ax = self.figure_outliers.gca()
            t1, t2 = sorted([self.selected_start_time, self.selected_end_time])
            f1, f2 = sorted([self.selected_start_flux, self.selected_end_flux])

            mask = ~((self.time >= t1) & (self.time <= t2) &
                     (self.flux >= f1) & (self.flux <= f2))

            self.time = self.time[mask]
            self.flux = self.flux[mask]
            if self.flux_err is not None:
                self.flux_err = self.flux_err[mask]

            self.selected_start_time = self.selected_end_time = None
            self.selected_start_flux = self.selected_end_flux = None

            if self.region_patch is not None:
                try:
                    self.region_patch.remove()
                except Exception:
                    pass
                self.region_patch = None

            if self.selector is not None:
                try:
                    self.selector.set_active(False)
                    if hasattr(self.selector, "set_visible"):
                        self.selector.set_visible(False)
                except Exception:
                    pass
                self.selector = None

            self.plot_data_in_outliers()

            ax = self.figure_outliers.gca()
            if self.time is not None and self.time.size > 0:
                xmin, xmax = np.nanmin(self.time), np.nanmax(self.time)
                if np.isfinite(xmin) and np.isfinite(xmax):
                    xpad = max((xmax - xmin) * 0.02, 1e-6)
                    ax.set_xlim(xmin - xpad, xmax + xpad)
            if self.flux is not None and self.flux.size > 0:
                ymin, ymax = np.nanmin(self.flux), np.nanmax(self.flux)
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ypad = max((ymax - ymin) * 0.05, 1e-6)
                    ax.set_ylim(ymin - ypad, ymax + ypad)
            self.canvas_outliers.draw_idle()

            self.on_outlier_data_changed()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

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
        ax = self.figure_outliers.gca()
        ax.set_title("Sigma Clipping")
        mean_flux = np.mean(self.flux)
        std_flux = np.std(self.flux)
        lower_bound = mean_flux - sigma_value * std_flux
        upper_bound = mean_flux + sigma_value * std_flux
        if self.lower_bound_line:
            self.lower_bound_line.remove()
        if self.upper_bound_line:
            self.upper_bound_line.remove()
        self.lower_bound_line = ax.axhline(lower_bound, color='red', linestyle='--')
        self.upper_bound_line = ax.axhline(upper_bound, color='green', linestyle='--')
        self.canvas_outliers.draw()
        QMessageBox.information(self, "Sigma Clipping",
                                f"Applied {sigma_value}-sigma bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")

    def remove_sigma_clipped_data(self):
        if self.lower_bound_line is None or self.upper_bound_line is None:
            QMessageBox.warning(self, "Warning", "No sigma clipping applied.")
            return
        lower_bound = self.lower_bound_line.get_ydata()[0]
        upper_bound = self.upper_bound_line.get_ydata()[0]
        mask = (self.flux >= lower_bound) & (self.flux <= upper_bound)
        self.time = self.time[mask]
        self.flux = self.flux[mask]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[mask]
        self.plot_data_in_outliers()

    def apply_boxplot_outliers(self):
        ax = self.figure_outliers.gca()
        ax.set_title("Boxplot Detection")
        q1 = np.percentile(self.flux, 25)
        q3 = np.percentile(self.flux, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if self.boxplot_lower_line:
            self.boxplot_lower_line.remove()
        if self.boxplot_upper_line:
            self.boxplot_upper_line.remove()
        self.boxplot_lower_line = ax.axhline(lower_bound, color='red', linestyle='--')
        self.boxplot_upper_line = ax.axhline(upper_bound, color='green', linestyle='--')
        self.canvas_outliers.draw()
        QMessageBox.information(self, "Boxplot Detection",
                                f"Boxplot thresholds: [{lower_bound:.3f}, {upper_bound:.3f}]")

    def remove_boxplot_outliers(self):
        if self.boxplot_lower_line is None or self.boxplot_upper_line is None:
            QMessageBox.warning(self, "Warning", "No boxplot detection applied.")
            return
        lower_bound = self.boxplot_lower_line.get_ydata()[0]
        upper_bound = self.boxplot_upper_line.get_ydata()[0]
        mask = (self.flux >= lower_bound) & (self.flux <= upper_bound)
        self.time = self.time[mask]
        self.flux = self.flux[mask]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[mask]
        self.plot_data_in_outliers()

    def apply_chauvenet_outliers(self):
        ax = self.figure_outliers.gca()
        ax.set_title("Chauvenet Detection")
        mean_flux = np.mean(self.flux)
        std_flux = np.std(self.flux)
        n = len(self.flux)
        probability = 1.0 / (2 * n)
        critical_value = np.sqrt(2) * erfcinv(2 * probability)
        lower_bound = mean_flux - critical_value * std_flux
        upper_bound = mean_flux + critical_value * std_flux
        if self.chauvenet_lower_line:
            self.chauvenet_lower_line.remove()
        if self.chauvenet_upper_line:
            self.chauvenet_upper_line.remove()
        self.chauvenet_lower_line = ax.axhline(lower_bound, color='red', linestyle='--')
        self.chauvenet_upper_line = ax.axhline(upper_bound, color='green', linestyle='--')
        self.canvas_outliers.draw()
        QMessageBox.information(self, "Chauvenet Detection",
                                f"Chauvenet thresholds: [{lower_bound:.3f}, {upper_bound:.3f}]")

    def remove_chauvenet_outliers(self):
        if self.chauvenet_lower_line is None or self.chauvenet_upper_line is None:
            QMessageBox.warning(self, "Warning", "No Chauvenet detection applied.")
            return
        lower_bound = self.chauvenet_lower_line.get_ydata()[0]
        upper_bound = self.chauvenet_upper_line.get_ydata()[0]
        mask = (self.flux >= lower_bound) & (self.flux <= upper_bound)
        self.time = self.time[mask]
        self.flux = self.flux[mask]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[mask]
        self.plot_data_in_outliers()

    def apply_normalization(self):
        try:
            median_flux = np.median(self.flux)
            self.flux = self.flux / median_flux
            if self.flux_err is not None:
                self.flux_err = self.flux_err / median_flux
            self.plot_data_in_outliers()
            self.on_outlier_data_changed()

            QMessageBox.information(self, "Normalization", "Data normalized by median.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during normalization: {e}")

    def restore_original_data(self):
        if self.original_time is None:
            QMessageBox.warning(self, "Warning", "No original data stored.")
            return
        self.time = self.original_time.copy()
        self.flux = self.original_flux.copy()
        self.flux_err = self.original_flux_err.copy()
        self.plot_data_in_outliers()
        QMessageBox.information(self, "Restore", "Original data restored.")

    def detach_plot(self):
        try:
            if self.time is None or self.flux is None:
                QMessageBox.warning(self, "Warning", "No data available for detaching.")
                return
            fig, ax = plt.subplots()
            if self.outlier_error_checkbox.isChecked() and self.flux_err is not None:
                ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                            markersize=5, capsize=1, elinewidth=1.5, ls="none")
            else:
                ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)
            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_title("Detached Outlier Plot")
            fig.tight_layout()
            fig.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detaching plot: {e}")

    def save_cleaned_data(self):
        try:
            if self.time is None or self.flux is None:
                QMessageBox.warning(self, "Warning", "No data to save.")
                return
            dialog = QDialog(self)
            dialog.setWindowTitle("Save Cleaned Data")
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format:"))
            file_format_combo = QComboBox()
            file_format_combo.addItems(["XLSX", "CSV", "TXT"])
            layout.addWidget(file_format_combo)
            add_2457000_checkbox = QCheckBox("Add 2454833 to Time")
            add_2457000_checkbox.setChecked(True)
            layout.addWidget(add_2457000_checkbox)
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            if dialog.exec_() != QDialog.Accepted:
                return
            file_format = file_format_combo.currentText().lower()
            file_name, _ = QFileDialog.getSaveFileName(
                self, f"Save Cleaned Data as {file_format.upper()}", "",
                f"{file_format.upper()} Files (*.{file_format})"
            )
            if not file_name:
                return
            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"
            time_data = self.time.copy()
            if add_2457000_checkbox.isChecked():
                time_data = self.time + 2454833
            df = pd.DataFrame({
                "Time": time_data,
                "Flux": self.flux,
                "Error": self.flux_err if self.flux_err is not None else np.zeros_like(self.flux)
            })
            if file_format == "csv":
                df.to_csv(file_name, index=False, float_format="%.10f")
            elif file_format == "xlsx":
                with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                    df.to_excel(writer, sheet_name="Cleaned Data", index=False, float_format="%.10f")
            elif file_format == "txt":
                np.savetxt(
                    file_name,
                    df.values,
                    fmt="%.10f",
                    header="\t".join(df.columns),
                    delimiter="\t",
                    comments=""
                )
            QMessageBox.information(self, "Data Saved", f"Cleaned data saved to: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data: {e}")

class KeplerDetrendingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = None
        self.flux = None
        self.flux_err = None


        self.original_time = None
        self.original_flux = None
        self.original_flux_err = None

        self.manual_segments = []
        self.automated_segments = []
        self.cleaned_time = None
        self.cleaned_flux = None
        self.cleaned_flux_err = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        manual_group = QGroupBox("Manual Detrending")
        manual_layout = QVBoxLayout()
        self.select_detrending_region_button = QPushButton("Select Region for Manual Detrending")
        self.select_detrending_region_button.clicked.connect(self.enable_detrending_region_selection)
        manual_layout.addWidget(self.select_detrending_region_button)
        manual_buttons_layout = QHBoxLayout()
        self.show_manual_trend_button = QPushButton("Show Trend")
        self.show_manual_trend_button.clicked.connect(self.show_manual_trend)
        self.remove_manual_trend_button = QPushButton("Remove Trend")
        self.remove_manual_trend_button.clicked.connect(self.remove_manual_trend)
        manual_buttons_layout.addWidget(self.show_manual_trend_button)
        manual_buttons_layout.addWidget(self.remove_manual_trend_button)
        manual_layout.addLayout(manual_buttons_layout)
        manual_group.setLayout(manual_layout)
        manual_group.setFixedHeight(100)


        poly_group = QGroupBox("Polynomial Fit")
        poly_layout = QVBoxLayout()
        self.degree_combo = QComboBox()
        self.degree_combo.addItems([
            "1st Degree Polynomial", "2nd Degree Polynomial",
            "3rd Degree Polynomial", "4th Degree Polynomial"
        ])
        self.degree_combo.setCurrentIndex(1)
        poly_layout.addWidget(self.degree_combo)
        poly_group.setLayout(poly_layout)
        poly_group.setFixedHeight(100)


        auto_group = QGroupBox("Automated Detrending")
        auto_layout = QVBoxLayout()
        threshold_layout = QHBoxLayout()
        self.detect_gaps_button = QPushButton("Detect Gaps")
        self.detect_gaps_button.clicked.connect(self.detect_gaps)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0, 10)
        self.threshold_input.setValue(1.0)
        self.threshold_input.setSingleStep(0.1)
        self.threshold_input.setDecimals(2)
        threshold_layout.addWidget(self.detect_gaps_button)
        threshold_layout.addWidget(self.threshold_input)
        auto_layout.addLayout(threshold_layout)
        auto_buttons_layout = QHBoxLayout()
        self.show_automated_trend_button = QPushButton("Show Trend")
        self.show_automated_trend_button.clicked.connect(self.show_automated_trend)
        self.remove_automated_trend_button = QPushButton("Remove Trend")
        self.remove_automated_trend_button.clicked.connect(self.remove_automated_trend)
        auto_buttons_layout.addWidget(self.show_automated_trend_button)
        auto_buttons_layout.addWidget(self.remove_automated_trend_button)
        auto_layout.addLayout(auto_buttons_layout)
        auto_group.setLayout(auto_layout)
        auto_group.setFixedHeight(100)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(manual_group)
        controls_layout.addWidget(poly_group)
        controls_layout.addWidget(auto_group)
        main_layout.addLayout(controls_layout)


        self.restore_button = QPushButton("Restore Data")
        self.restore_button.clicked.connect(self.restore_data)
        main_layout.addWidget(self.restore_button)


        detrending_group = QGroupBox("Light Curve")
        detrending_layout = QVBoxLayout()
        self.figure_detrending = Figure()
        self.canvas_detrending = FigureCanvas(self.figure_detrending)
        self.toolbar_detrending = NavigationToolbar(self.canvas_detrending, self)
        self.detrending_error_checkbox = QCheckBox("Display with Errors")
        self.detrending_error_checkbox.setChecked(True)
        self.detrending_error_checkbox.stateChanged.connect(self.plot_data_in_detrending)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.toolbar_detrending)
        toolbar_layout.addWidget(self.detrending_error_checkbox)
        toolbar_layout.setAlignment(Qt.AlignLeft)
        detrending_layout.addWidget(self.canvas_detrending)
        detrending_layout.addLayout(toolbar_layout)
        detrending_group.setLayout(detrending_layout)
        main_layout.addWidget(detrending_group)

        bottom_layout = QHBoxLayout()
        self.detach_detrending_plot_button = QPushButton("Detached")
        self.detach_detrending_plot_button.clicked.connect(self.detach_detrending_plot)
        bottom_layout.addWidget(self.detach_detrending_plot_button)
        self.save_detrending_data_button = QPushButton("Save Detrended Data")
        self.save_detrending_data_button.clicked.connect(self.save_detrending_data)
        bottom_layout.addWidget(self.save_detrending_data_button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.setMinimumSize(800, 600)

    def _lock_y_format(self, ax, fmt="%.2f"):
        """Y ekseni etiket biçimini sabitle (.2f vb.)."""
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))

    def _apply_axis_style(self, ax):
        """Apply y-label/inversion based on optional magnitude selection."""
        ylabel = getattr(self, "_flux_label", "Flux ($electrons^{-1}$)")
        ax.set_ylabel(ylabel)
        if getattr(self, "_invert_y", False):
            ax.invert_yaxis()

    def showEvent(self, event):
        super().showEvent(event)
        try:
            if self.figure_detrending is not None and self.figure_detrending.axes:
                self.figure_detrending.tight_layout()
                self.canvas_detrending.draw_idle()
        except Exception:
            pass

    def setData(self, time, flux, flux_err):
        self.time = time.copy()
        self.flux = flux.copy()
        self.flux_err = flux_err.copy()
        if self.original_time is None:
            self.original_time = self.time.copy()
            self.original_flux = self.flux.copy()
            self.original_flux_err = self.flux_err.copy()
        self.plot_data_in_detrending()

    def plot_data_in_detrending(self):
        ax = self.figure_detrending.gca()
        ax.clear()
        if self.time is not None and self.flux is not None:
            if self.detrending_error_checkbox.isChecked() and self.flux_err is not None:
                ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                            markersize=5, capsize=1, elinewidth=1.5, ls="none")
            else:
                ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)
            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_title("Detrending of Light Curve")
            self._apply_axis_style(ax)

        self._lock_y_format(ax, "%.2f")  # << eklenen satır
        self.figure_detrending.tight_layout()
        self.canvas_detrending.draw()

    def enable_detrending_region_selection(self):
        try:
            ax = self.figure_detrending.gca()
            if hasattr(self, 'detrending_selector'):
                self.detrending_selector.set_active(False)
                del self.detrending_selector
            self.detrending_selector = RectangleSelector(
                ax, self.on_select_detrending, useblit=True, interactive=True,
                button=[1], minspanx=0.1, minspany=0.1, spancoords='data'
            )
            QMessageBox.information(self, "Detrending Selection", "Drag to select the region for manual detrending.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error enabling detrending region: {e}")

    def on_select_detrending(self, eclick, erelease):
        start_time = eclick.xdata
        end_time = erelease.xdata
        if start_time is not None and end_time is not None:
            self.manual_segments.append((min(start_time, end_time), max(start_time, end_time)))
            QMessageBox.information(self, "Manual Segment",
                                    f"Manual segment added from {start_time:.2f} to {end_time:.2f}.")
        else:
            QMessageBox.warning(self, "Warning", "No valid region selected.")

    def show_manual_trend(self):
        try:
            if not self.manual_segments:
                QMessageBox.warning(self, "Warning", "No manual segments selected for detrending.")
                return
            degree = int(self.degree_combo.currentText().split()[0][0])
            ax = self.figure_detrending.gca()
            ax.clear()

            eb = ax.errorbar(
                self.time, self.flux, yerr=self.flux_err,
                fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5,
                zorder=1
            )

            for start, end in self.manual_segments:
                mask = (self.time >= start) & (self.time <= end)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]
                if len(time_segment) == 0:
                    continue
                p_coef = np.polyfit(time_segment, flux_segment, deg=degree, w=1 / flux_err_segment)
                poly_fit = np.poly1d(p_coef)


                ax.plot(
                    time_segment, poly_fit(time_segment),
                    "--", color="red", linewidth=1.5, zorder=10
                )

            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_title("Detrending of Light Curve")
            self._apply_axis_style(ax)
            self.canvas_detrending.draw()
            self.figure_detrending.tight_layout()
            self._lock_y_format(ax, "%.2f")

            QMessageBox.information(self, "Manual Detrending Applied",
                                    f"Polynomial fit (degree {degree}) applied to manual segments.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing manual trend: {e}")

    def remove_manual_trend(self):
        try:
            if not hasattr(self, 'pre_detrend_time'):

                self.pre_detrend_time = self.time.copy()
                self.pre_detrend_flux = self.flux.copy()
                self.pre_detrend_flux_err = self.flux_err.copy()
            if not self.manual_segments:
                QMessageBox.warning(self, "Warning", "No manual segments selected for detrending removal.")
                return
            degree = int(self.degree_combo.currentText().split()[0][0])
            updated_flux = self.flux.copy()
            updated_flux_err = self.flux_err.copy()
            for start, end in self.manual_segments:
                mask = (self.time >= start) & (self.time <= end)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]
                if len(time_segment) == 0:
                    continue
                p_coef = np.polyfit(time_segment, flux_segment, deg=degree, w=1/flux_err_segment)
                poly_fit = np.poly1d(p_coef)
                detrended_flux = flux_segment - poly_fit(time_segment) + 1.0
                updated_flux[mask] = detrended_flux
                updated_flux_err[mask] = flux_err_segment
            self.flux = updated_flux
            self.flux_err = updated_flux_err

            median_flux = np.median(self.flux)
            self.flux /= median_flux
            self.flux_err /= median_flux
            self.manual_segments.clear()
            self.plot_data_in_detrending()
            QMessageBox.information(self, "Manual Detrending Removed",
                                    "Polynomial trend removed from manual segments and data normalized.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing manual detrending: {e}")

    def detect_gaps(self):
        try:
            threshold = self.threshold_input.value()
            time_diffs = np.diff(self.time)
            gap_indices = np.where(time_diffs > threshold)[0] + 1
            self.time_segments = np.split(self.time, gap_indices)
            self.flux_segments = np.split(self.flux, gap_indices)
            self.flux_err_segments = np.split(self.flux_err, gap_indices)
            self.automated_segments = [(segment[0], segment[-1]) for segment in self.time_segments if len(segment) > 0]
            ax = self.figure_detrending.gca()
            for start, end in self.automated_segments:
                ax.axvspan(start, end, color="gray", alpha=0.3)
            self.canvas_detrending.draw()
            QMessageBox.information(self, "Gaps Detected",
                                    f"{len(self.automated_segments)} segment(s) detected (threshold={threshold}).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detecting gaps: {e}")

    def show_automated_trend(self):
        try:
            if not self.automated_segments:
                QMessageBox.warning(self, "Warning", "No automated segments detected for detrending.")
                return
            degree = int(self.degree_combo.currentText().split()[0][0])
            ax = self.figure_detrending.gca()
            ax.clear()


            eb = ax.errorbar(
                self.time, self.flux, yerr=self.flux_err,
                fmt=".", color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5,
                zorder=1
            )

            for start, end in self.automated_segments:
                mask = (self.time >= start) & (self.time <= end)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]
                if len(time_segment) == 0:
                    continue
                p_coef = np.polyfit(time_segment, flux_segment, deg=degree, w=1 / flux_err_segment)
                poly_fit = np.poly1d(p_coef)

                ax.plot(
                    time_segment, poly_fit(time_segment),
                    "--", color="red", linewidth=1.5, zorder=10
                )

            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_title("Detrending of Light Curve")
            self._apply_axis_style(ax)
            self.canvas_detrending.draw()
            self.figure_detrending.tight_layout()
            self._lock_y_format(ax, "%.2f")

            QMessageBox.information(self, "Automated Detrending Applied",
                                    f"Automated polynomial trend (degree {degree}) displayed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing automated trend: {e}")

    def remove_automated_trend(self):
        try:
            if not self.automated_segments:
                QMessageBox.warning(self, "Warning", "No automated segments for detrending removal.")
                return
            degree = int(self.degree_combo.currentText().split()[0][0])
            updated_flux = self.flux.copy()
            updated_flux_err = self.flux_err.copy()
            for start, end in self.automated_segments:
                mask = (self.time >= start) & (self.time <= end)
                time_segment = self.time[mask]
                flux_segment = self.flux[mask]
                flux_err_segment = self.flux_err[mask]
                if len(time_segment) == 0:
                    continue
                p_coef = np.polyfit(time_segment, flux_segment, deg=degree, w=1/flux_err_segment)
                poly_fit = np.poly1d(p_coef)
                detrended_flux = flux_segment - poly_fit(time_segment) + 1.0
                updated_flux[mask] = detrended_flux
                updated_flux_err[mask] = flux_err_segment
            self.flux = updated_flux
            self.flux_err = updated_flux_err
            median_flux = np.median(self.flux)
            self.flux /= median_flux
            self.flux_err /= median_flux
            self.automated_segments.clear()
            self.plot_data_in_detrending()
            QMessageBox.information(self, "Automated Detrending Removed",
                                    "Automated polynomial trend removed and data normalized.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing automated detrending: {e}")

    def restore_data(self):
        try:
            if hasattr(self, 'pre_detrend_flux'):
                self.time = self.pre_detrend_time.copy()
                self.flux = self.pre_detrend_flux.copy()
                self.flux_err = self.pre_detrend_flux_err.copy()
                del self.pre_detrend_time, self.pre_detrend_flux, self.pre_detrend_flux_err
            self.plot_data_in_detrending()
            QMessageBox.information(self, "Data Restored",
                                    "Detrending data has been restored from current outlier data.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error restoring detrending data: {e}")

    def update_outlier_data(self, time, flux, flux_err):
        self.cleaned_time = time
        self.cleaned_flux = flux
        self.cleaned_flux_err = flux_err
        ax = self.figure_outliers.gca()
        ax.clear()
        ax.errorbar(self.cleaned_time, self.cleaned_flux, yerr=self.cleaned_flux_err, fmt=".", color="dodgerblue",
                    markersize=5, capsize=1, elinewidth=1.5)
        ax.set_xlabel("Time - 2454833 (day)")
        ax.set_ylabel("Normalized Flux ($electrons^{-1}$)")
        ax.set_title("Normalized Light Curve")
        self.canvas_outliers.draw()
        self.plot_data_in_detrending()
        self._lock_y_format(ax, "%.2f")

    def restore_original_data(self):
        try:
            if self.original_time is None:
                QMessageBox.warning(self, "Warning", "No original data stored.")
                return
            self.time = self.original_time.copy()
            self.flux = self.original_flux.copy()
            self.flux_err = self.original_flux_err.copy()
            self.plot_data_in_outliers()
            QMessageBox.information(self, "Data Restored",
                                    "Original data restored in Outlier and Detrending tabs.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error restoring original data: {e}")

    def detach_detrending_plot(self):
        try:
            if self.cleaned_time is not None and self.cleaned_flux is not None and self.cleaned_flux_err is not None:
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
            ax.errorbar(time, flux, yerr=flux_err, fmt=".", color="dodgerblue",
                        markersize=5, capsize=1, elinewidth=1.5)
            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_title("Detached Detrended Light Curve")
            fig.tight_layout()
            fig.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detaching detrending plot: {e}")

    def save_detrending_data(self):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Save Detrended Data")
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select file format:"))
            file_format_combo = QComboBox()
            file_format_combo.addItems(["XLSX", "CSV", "TXT"])
            layout.addWidget(file_format_combo)
            add_checkbox = QCheckBox("Add 2454833 to Time")
            add_checkbox.setChecked(True)
            layout.addWidget(add_checkbox)
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            if dialog.exec_() != QDialog.Accepted:
                return
            file_format = file_format_combo.currentText().lower()
            file_name, _ = QFileDialog.getSaveFileName(self, f"Save Detrended Data as {file_format.upper()}",
                                                       "", f"{file_format.upper()} Files (*.{file_format})")
            if not file_name:
                return
            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"
            time_data = self.time.copy()
            if add_checkbox.isChecked():
                time_data = self.time + 2454833
            df = pd.DataFrame({
                "Time": time_data,
                "Flux": self.flux,
                "Error": self.flux_err if self.flux_err is not None else np.zeros_like(self.flux)
            })
            if file_format == "csv":
                df.to_csv(file_name, index=False, float_format="%.10f")
            elif file_format == "xlsx":
                with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                    df.to_excel(writer, sheet_name="Detrended Data", index=False, float_format="%.10f")
            elif file_format == "txt":
                np.savetxt(file_name, df.values, fmt="%.10f",
                           header="\t".join(df.columns), delimiter="\t", comments="")
            QMessageBox.information(self, "Data Saved", f"Detrended data saved to: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving detrended data: {e}")

    def plot_data_in_outliers(self):
        """Replot data in Outlier tab (used when restoring original data)."""
        ax = self.figure_outliers.gca()
        ax.clear()
        if self.time is not None and self.flux is not None:
            if self.flux_err is not None:
                ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                            markersize=5, capsize=1, elinewidth=1.5)
            else:
                ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)
            ax.set_xlabel("Time - 2454833 (day)")
            ax.set_ylabel("Flux ($electrons^{-1}$)")
            ax.set_title("Original Light Curve from Query Data")
        self.canvas_outliers.draw()

# def auto_phase_fold_kepler(time, flux, flux_err=None, canvas=None, display_fields=None):
#
#     mask = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask]
#     flux = np.asarray(flux)[mask]
#     if flux_err is not None:
#         flux_err = np.asarray(flux_err)[mask]
#     else:
#         flux_err = np.ones_like(flux)
#
#     if len(time) < 10:
#         raise ValueError("Phase folding için yeterli veri yok.")
#
#     med = np.median(flux)
#     if med == 0 or not np.isfinite(med):
#         med = 1.0
#     flux = flux / med
#     flux_err = flux_err / med
#
#     inverted = -flux
#     prominence_value = 0.05
#     height_value = -0.98
#     peaks, props = find_peaks(inverted, prominence=prominence_value, height=height_value)
#     minima_t = time[peaks]
#     minima_f = flux[peaks]
#
#     if len(minima_t) < 2:
#         peaks, props = find_peaks(inverted, prominence=0.02)
#         minima_t = time[peaks]
#         minima_f = flux[peaks]
#
#     if len(minima_t) < 2:
#
#         freq = np.linspace(1/100, 1/0.05, 50000)
#         ls = LombScargle(time, flux)
#         power = ls.power(freq)
#         best_freq = freq[np.argmax(power)]
#         best_period = 1 / best_freq
#         t0_refined = time[np.argmin(flux)]
#         period_estimate = best_period
#         phase_final = ((time - t0_refined) / best_period) % 1.0
#         extended_phase = np.concatenate([phase_final, phase_final + 1.0])
#         extended_flux = np.concatenate([flux, flux])
#         extended_err = np.concatenate([flux_err, flux_err])
#         if canvas:
#             canvas.figure.clear()
#             ax = canvas.figure.add_subplot(111)
#             ax.errorbar(extended_phase, extended_flux, yerr=extended_err, fmt='.',
#                         color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
#             ax.axvline(1, color='g', linestyle='--', label="Primary Minima")
#             ax.set_xlabel('Phase'); ax.set_ylabel('Normalized Flux ($electrons^{-1}$)')
#             ax.set_title('Phase-Folded Light Curve')
#             ax.set_xlim(0.2, 1.2); ax.legend(loc=3)
#             canvas.draw()
#         # if display_fields:
#         #     display_fields["best_t0_display"].setText(f"{t0_refined:.6f}")
#         #     display_fields["best_period_display"].setText(f"{best_period:.6f}")
#         #     dd = display_fields["period_dropdown"]
#         #     dd.clear(); dd.addItem(f"Lomb-Scargle: {best_period:.6f}")
#         #     dd.addItem(f"Estimate: {period_estimate:.6f}")
#         return t0_refined, period_estimate, best_period, extended_phase, extended_flux, extended_err
#
#     try:
#         gmm = GaussianMixture(n_components=2, random_state=42)
#         gmm.fit(minima_f.reshape(-1, 1))
#         means = np.sort(gmm.means_.flatten())
#         thr = (means[0] + means[1]) / 2.0
#         primary_idx = np.where(minima_f <= thr)[0]
#     except Exception:
#         order = np.argsort(minima_f)
#         primary_idx = order[: max(2, len(order)//2)]
#
#     primary_t = minima_t[primary_idx]
#     if len(primary_t) < 2:
#         primary_t = minima_t
#
#     primary_t = np.sort(primary_t)
#     diffs = np.diff(primary_t)
#     if len(diffs) == 0 or not np.isfinite(np.median(diffs)):
#         freq = np.linspace(1/100, 1/0.05, 50000)
#         ls = LombScargle(time, flux)
#         power = ls.power(freq)
#         best_freq = freq[np.argmax(power)]
#         period_estimate = best_period = 1 / best_freq
#     else:
#         period_estimate = float(np.median(diffs))
#         fg = 1.0 / period_estimate
#         freq = np.linspace(fg*0.5, fg*1.5, 10000)
#         ls = LombScargle(time, flux)
#         power = ls.power(freq)
#         best_freq = freq[np.argmax(power)]
#         best_period = 1 / best_freq
#
#     t0_initial = primary_t[0]
#     delta_t = max(0.02, 0.05 * best_period)
#     mfit = (time >= t0_initial - delta_t) & (time <= t0_initial + delta_t)
#     tfit = time[mfit]; ffit = flux[mfit]; efit = flux_err[mfit]
#     if len(tfit) >= 5:
#         try:
#             w = 1.0 / np.maximum(efit, 1e-6)
#             spline = UnivariateSpline(tfit, ffit, w=w, k=3, s=0)
#             tfine = np.linspace(tfit.min(), tfit.max(), 5000)
#             ffine = spline(tfine)
#             t0_refined = tfine[np.argmin(ffine)]
#         except Exception:
#             t0_refined = t0_initial
#     else:
#         t0_refined = t0_initial
#
#     def fourier3(x, a0, a1, b1, a2, b2, a3, b3):
#         return (a0 +
#                 a1*np.cos(2*np.pi*1*x) + b1*np.sin(2*np.pi*1*x) +
#                 a2*np.cos(2*np.pi*2*x) + b2*np.sin(2*np.pi*2*x) +
#                 a3*np.cos(2*np.pi*3*x) + b3*np.sin(2*np.pi*3*x))
#
#     phase0 = ((time - t0_refined) / best_period) % 1.0
#     ext_phase = np.concatenate([phase0, phase0 + 1.0])
#     ext_flux  = np.concatenate([flux, flux])
#
#     mph = (ext_phase >= 0.9) & (ext_phase <= 1.1)
#     ph_fit = ext_phase[mph]; fl_fit = ext_flux[mph]
#     t0_final = t0_refined
#     if len(ph_fit) >= 8:
#         try:
#             p0 = [np.mean(fl_fit)] + [0.0]*6
#             pars, _ = curve_fit(fourier3, ph_fit, fl_fit, p0=p0, maxfev=20000)
#             ph_fine = np.linspace(ph_fit.min(), ph_fit.max(), 2000)
#             fl_fine = fourier3(ph_fine, *pars)
#             ph_min = ph_fine[np.argmin(fl_fine)]
#             t0_final = t0_refined + (ph_min * best_period)
#         except Exception:
#             pass
#
#     phase_final = ((time - t0_final) / best_period) % 1.0
#     extended_phase = np.concatenate([phase_final, phase_final + 1.0])
#     extended_flux  = np.concatenate([flux, flux])
#     extended_err   = np.concatenate([flux_err, flux_err])
#
#     if canvas:
#         canvas.figure.clear()
#         ax = canvas.figure.add_subplot(111)
#         ax.errorbar(extended_phase, extended_flux, yerr=extended_err, fmt='.',
#                     color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
#         ax.axvline(1, color='g', linestyle='--', label="Primary Minima")
#         ax.set_xlabel('Phase')
#         ax.set_ylabel("Normalized Flux ($electrons^{-1}$)")
#         ax.set_title('Phase-Folded Light Curve')
#         ax.set_xlim(0.2, 1.2)
#         ax.legend(loc=3)
#         canvas.draw()
#
#     if display_fields:
#         display_fields["best_t0_display"].setText(f"{t0_refined:.6f}")
#         display_fields["best_period_display"].setText(f"{best_period:.6f}")
#         dd = display_fields["period_dropdown"]
#         dd.clear()
#         dd.addItem(f"Lomb-Scargle: {best_period:.6f}")
#         dd.addItem(f"Estimate: {period_estimate:.6f}")
#
#     return t0_refined, period_estimate, best_period, extended_phase, extended_flux, extended_err


def _adjusted_minima_time_via_spline(time, flux, flux_err=None,
                                     prominence=0.05, height=-0.98,
                                     delta_t=0.05, min_pts=15):

    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    e = None if flux_err is None else np.asarray(flux_err, float)

    inv = -f
    peaks, props = find_peaks(inv, prominence=prominence, height=height)

    if len(peaks) == 0:
        return float(t[np.argmin(f)])

    depths = inv[peaks]
    qcut = np.percentile(depths, 75)
    prim = peaks[depths >= qcut]
    t0_initial = np.min(t[prim]) if len(prim) else t[peaks[np.argmax(depths)]]

    half = delta_t
    for _ in range(6):
        sel = (t >= t0_initial - half) & (t <= t0_initial + half)
        if np.count_nonzero(sel) >= min_pts:
            break
        half *= 2.0
    sel = (t >= t0_initial - half) & (t <= t0_initial + half)
    if np.count_nonzero(sel) < 3:
        return float(t0_initial)

    tf, ff = t[sel], f[sel]
    w = None
    if e is not None:
        ef = np.copy(e[sel])
        ef[~np.isfinite(ef)] = np.nanmedian(ef[np.isfinite(ef)]) or 1.0
        w = 1.0 / np.maximum(ef, 1e-6)

    spl = UnivariateSpline(tf, ff, w=w, k=3, s=0)
    t_fine = np.linspace(tf.min(), tf.max(), 10000)
    f_fine = spl(t_fine)
    return float(t_fine[np.argmin(f_fine)])


def _align_t0_to_phase_one(t, f, e, P, t0_raw):
    phase_tmp = ((t - t0_raw) / P) % 1.0
    idx = np.argsort(phase_tmp)
    ph, fl = phase_tmp[idx], f[idx]
    ph_ext = np.concatenate([ph, ph + 1.0])
    fl_ext = np.concatenate([fl, fl])
    w = None
    if e is not None:
        w = 1.0 / np.maximum(e[idx], 1e-6)
        w = np.concatenate([w, w])

    try:
        spl = UnivariateSpline(ph_ext, fl_ext, w=w, k=3, s=0)
        ph_f = np.linspace(0.8, 1.2, 2000)
        fl_f = spl(ph_f)
        ph_min = float(ph_f[np.argmin(fl_f)] % 1.0)
    except Exception:
        ph_min = float(ph_ext[np.argmin(fl_ext)] % 1.0)

    t0_aligned = float(t0_raw + (ph_min - 1.0) * P)
    return t0_aligned, ph_min

def _mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _sigma_clip_asymmetric(time, flux, flux_err=None, high_sigma=6.0):
    med = np.median(flux)
    s = _mad(flux)
    if s == 0 or not np.isfinite(s):
        s = np.std(flux) if np.std(flux) > 0 else 1.0
    mask = (flux <= med + high_sigma*s) & np.isfinite(time) & np.isfinite(flux)
    t = time[mask]; f = flux[mask]
    if flux_err is None:
        e = np.full_like(f, _mad(f) if _mad(f) > 0 else (np.std(f) if np.std(f)>0 else 1.0))
    else:
        e = flux_err[mask]
        e[~np.isfinite(e)] = np.nanmedian(e[np.isfinite(e)]) or 1.0
    return t, f, e

def _pdm_theta(time, flux, period, nbins=20):
    phase = ((time / period) % 1.0)
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]
    bins = np.linspace(0, 1, nbins+1)
    th_num, th_den = 0.0, (len(flux)-1) * np.var(flux) if len(flux)>1 else 1.0
    for i in range(nbins):
        m = (phase >= bins[i]) & (phase < bins[i+1])
        if np.count_nonzero(m) > 1:
            th_num += (np.count_nonzero(m)-1) * np.var(flux[m])
    theta = th_num / (th_den if th_den>0 else 1.0)
    return np.clip(theta, 0, 1e9)

def _conditional_entropy(time, flux, period, nphi=30, nflux=30):
    phase = ((time / period) % 1.0)
    fmin, fmax = np.percentile(flux, [0.5, 99.5])
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        fmin, fmax = np.min(flux), np.max(flux)+1e-6
    Hphi_flux, _, _ = np.histogram2d(phase, np.clip((flux - fmin)/(fmax-fmin), 0, 1), bins=[nphi, nflux], density=True)
    p_joint = Hphi_flux / (np.sum(Hphi_flux) + 1e-16)
    p_phi = np.sum(p_joint, axis=1, keepdims=True)
    H_joint = -np.nansum(p_joint * np.log(p_joint + 1e-16))
    H_phi   = -np.nansum(p_phi * np.log(p_phi + 1e-16))
    H_cond  = H_joint - H_phi
    return float(H_cond)

def _objective(time, flux, period):
    pdm = _pdm_theta(time, flux, period)
    ce  = _conditional_entropy(time, flux, period)
    pdm_n = np.log1p(pdm)
    ce_n  = np.log1p(ce)
    return 0.6 * pdm_n + 0.4 * ce_n

def _refine_period(time, flux, p0, pmin, pmax, width_frac=0.15):
    lo = max(pmin, p0 * (1.0 - width_frac))
    hi = min(pmax, p0 * (1.0 + width_frac))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return p0
    res = minimize_scalar(lambda p: _objective(time, flux, p), bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
    return float(res.x) if res.success else p0
def robust_auto_phase_fold(time, flux, flux_err=None, canvas=None, display_fields=None,
                           Pmin=0.05, Pmax=None, top_k=5, samples_per_peak=10):


    t, f, e = _sigma_clip_asymmetric(time, flux, flux_err, high_sigma=6.0)
    if len(t) < 10:
        raise ValueError("Yeterli nokta yok.")
    med = np.median(f);  f = f / (med if med!=0 else 1.0);  e = e / (med if med!=0 else 1.0)

    span = np.nanmax(t) - np.nanmin(t)
    if Pmax is None:
        Pmax = min(100.0, 0.8*span)
    Pmin = max(Pmin, (np.median(np.diff(np.sort(t))) or 1e-3)*2.5)

    if not np.isfinite(Pmax) or Pmax <= Pmin*1.1:
        Pmax = Pmin*1.2


    freq_min, freq_max = 1.0/Pmax, 1.0/Pmin
    gls = LombScargle(t, f, e, nterms=3, center_data=True, fit_mean=True)
    freq_g, power_g = gls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max,
                                    samples_per_peak=samples_per_peak, normalization='standard')

    peaks_g, _ = find_peaks(power_g, distance=max(1, int(0.25*samples_per_peak)))
    cand_g = (1.0 / freq_g[peaks_g]) if len(peaks_g)>0 else np.array([])


    bls = BoxLeastSquares(t, f, e)
    test_periods = np.linspace(Pmin, Pmax, 5000)
    dfracs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])

    bls_power = np.zeros_like(test_periods)
    nblocks = 6

    period_blocks = np.array_split(test_periods, nblocks)
    index_blocks = np.array_split(np.arange(len(test_periods)), nblocks)

    for block, idx in zip(period_blocks, index_blocks):
        dur_grid = dfracs * block.min()
        try:
            res = bls.power(block, dur_grid)
            power = res.power
            if getattr(power, "ndim", 1) == 2:
                power = power.max(axis=1)
            bls_power[idx] = np.maximum(bls_power[idx], power)
        except Exception:
            continue

    peaks_b, _ = find_peaks(bls_power, distance=5)
    cand_b = block[0] * 0 + test_periods[peaks_b] if len(peaks_b) > 0 else np.array([])

    candidates = []
    for P in np.concatenate([cand_g[:top_k], cand_b[:top_k]]):
        if not np.isfinite(P): continue
        for mult in [0.5, 1.0, 2.0]:
            Pm = P*mult
            if Pmin <= Pm <= Pmax:
                candidates.append(Pm)
    if not candidates:
        P0 = 1.0 / freq_g[np.argmax(power_g)]
        candidates = [np.clip(P0, Pmin, Pmax)]

    candidates = np.array(sorted(set([float(np.clip(P, Pmin, Pmax)) for P in candidates])))

    scores = []
    for P in candidates:
        scores.append(_objective(t, f, P))
    scores = np.array(scores)
    best_idx = int(np.argmin(scores))
    P_best = candidates[best_idx]

    P_ref = _refine_period(t, f, P_best, Pmin, Pmax, width_frac=0.15)


    t0_raw = _adjusted_minima_time_via_spline(t, f, e,
                                              prominence=0.05, height=-0.98,
                                              delta_t=0.05, min_pts=15)
    t0, ph_min = _align_t0_to_phase_one(t, f, e, P_ref, t0_raw)

    def _score_total(P):
        return _objective(t, f, P)
    P_alt = 2.0 * P_ref
    if Pmin <= 0.5*P_ref:
        P_half = 0.5*P_ref
    else:
        P_half = None

    choose_P = P_ref
    score_ref = _score_total(P_ref)
    if P_alt <= Pmax and _score_total(P_alt) < score_ref*0.98:
        choose_P = P_alt
    if P_half is not None and _score_total(P_half) < _score_total(choose_P)*0.98:
        choose_P = P_half

    if choose_P != P_ref:
        P_ref = _refine_period(t, f, choose_P, Pmin, Pmax, width_frac=0.10)
        t0_raw = _adjusted_minima_time_via_spline(t, f, e,
                                                  prominence=0.05, height=-0.98,
                                                  delta_t=0.05, min_pts=15)
        t0, ph_min = _align_t0_to_phase_one(t, f, e, P_ref, t0_raw)

    phase = ((t - t0) / P_ref) % 1.0
    phase_ext = np.concatenate([phase, phase + 1.0])
    flux_ext  = np.concatenate([f, f])
    err_ext   = np.concatenate([e, e])

    if canvas:
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.errorbar(phase_ext, flux_ext, yerr=err_ext, fmt='.',
                    color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
        ax.axvline(1, color='g', linestyle='--', label="Primary Minima")
        ax.set_xlabel('Phase'); ax.set_ylabel("Normalized Flux ($electrons^{-1}$)")
        ax.set_title('Phase-Folded Light Curve')
        ax.set_xlim(0.2, 1.2); ax.legend(loc=3)
        canvas.figure.tight_layout()

        canvas.draw()

    if display_fields:
        display_fields["best_t0_display"].setText(f"{t0:.6f}")
        display_fields["best_period_display"].setText(f"{P_ref:.6f}")
        dd = display_fields["period_dropdown"]
        dd.clear()
        dd.addItem(f"GLS/BLS fused: {P_ref:.6f}")

    P_gls_peak = 1.0 / freq_g[np.argmax(power_g)]
    P_bls_peak = test_periods[np.argmax(bls_power)] if len(bls_power) else None

    alts = []
    if Pmin <= 0.5 * P_ref:
        alts.append(0.5 * P_ref)
    if 2.0 * P_ref <= Pmax:
        alts.append(2.0 * P_ref)

    P_alt = None
    if alts:
        sc = [_objective(t, f, P) for P in alts]
        P_alt = alts[int(np.argmin(sc))]

    periods_out = {
        "auto": P_ref,
        "alt_harm": P_alt,
        "gls": P_gls_peak,
        "bls": P_bls_peak
    }


    return t0, P_ref, phase_ext, flux_ext, err_ext, periods_out

class KeplerPhaseFoldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = None
        self.flux = None
        self.flux_err = None

        self.phase_time = None
        self.phase_flux = None
        self.phase_flux_err = None
        self._period_choices: dict[str, float] = {}
        self.init_ui()

    def init_ui(self):

        phase_layout = QVBoxLayout()

        manual_group = QGroupBox("Manual Phase Folding")
        manual_layout = QVBoxLayout()

        manual_inputs_layout = QHBoxLayout()
        self.t0_input = QLineEdit()
        self.t0_input.setPlaceholderText("e.g., 2451350.024259")
        self.period_input = QLineEdit()
        self.period_input.setPlaceholderText("e.g., 0.2375")

        manual_inputs_layout.addWidget(QLabel("Minima Time:"))
        manual_inputs_layout.addWidget(self.t0_input)
        manual_inputs_layout.addWidget(QLabel("Period:"))
        manual_inputs_layout.addWidget(self.period_input)

        manual_button_layout = QHBoxLayout()
        manual_phase_button = QPushButton("Phase Fold Manually")
        manual_phase_button.clicked.connect(self.manual_phase_fold)
        manual_button_layout.addWidget(manual_phase_button)

        manual_layout.addLayout(manual_inputs_layout)
        manual_layout.addLayout(manual_button_layout)
        manual_group.setLayout(manual_layout)

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
        self.period_dropdown.setEditable(False)
        _add_placeholder(self.period_dropdown, "─ Periods ─")
        self.period_dropdown.setEnabled(False)

        self.period_dropdown.setMinimumWidth(200)
        self.period_dropdown.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)

        # değişiklik sinyalini yakala
        self.period_dropdown.currentIndexChanged.connect(
            self._on_period_choice_changed)





        automate_button = QPushButton("Automate Phase Folding")
        automate_button.clicked.connect(self.automate_phase_fold)
        automate_button_layout.addWidget(self.period_dropdown)
        automate_button_layout.addWidget(automate_button)

        automate_layout.addLayout(automate_inputs_layout)
        automate_layout.addLayout(automate_button_layout)
        automate_group.setLayout(automate_layout)


        phase_controls_layout = QHBoxLayout()
        manual_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        automate_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        phase_controls_layout.addWidget(manual_group)
        phase_controls_layout.addWidget(automate_group)

        phase_layout.addLayout(phase_controls_layout)


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

        self.save_phase_data_button = QPushButton("Save Data")
        self.save_phase_data_button.clicked.connect(self.save_phase_data)
        phase_button_layout.addWidget(self.save_phase_data_button)

        phase_layout.addLayout(phase_button_layout)


        self.setLayout(phase_layout)

    def _on_period_choice_changed(self, index: int):
        label = self.period_dropdown.itemText(index)
        value = self._period_choices.get(label)
        if value is not None:
            self.best_period_display.setText(f"{value:.6f}")

    def manual_phase_fold(self):
        try:
            t0 = float(self.t0_input.text().strip())
            period = float(self.period_input.text().strip())
            phase, flux, flux_err = phase_fold_manually(
                time=self.time,
                flux=self.flux,
                flux_err=self.flux_err,
                t0=t0,
                period=period,
                canvas=self.canvas_phase
            )
            self.phase_time = phase
            self.phase_flux = flux
            self.phase_flux_err = flux_err

            QMessageBox.information(self, "Manual Phase Folding", "Manual phase folding completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during manual phase folding: {e}")

    def automate_phase_fold(self):
        try:
            t0, best_period, phase, flux, flux_err, periods = robust_auto_phase_fold(
                time=self.time,
                flux=self.flux,
                flux_err=self.flux_err,
                canvas=self.canvas_phase,
                display_fields={
                    "best_t0_display": self.best_t0_display,
                    "best_period_display": self.best_period_display,
                    "period_dropdown": self.period_dropdown
                },
                Pmin=0.05,
                Pmax=None,
                top_k=5,
                samples_per_peak=12
            )

            self.phase_time = phase
            self.phase_flux = flux
            self.phase_flux_err = flux_err

            dd = self.period_dropdown

            dd.blockSignals(True)
            dd.clear()
            self._period_choices.clear()

            if periods.get("auto"):
                lbl = f"Auto (GLS/BLS fused): {periods['auto']:.6f}"
                dd.addItem(lbl)
                self._period_choices[lbl] = float(periods["auto"])

            if periods.get("alt_harm"):
                lbl = f"Alternative (harmonic): {periods['alt_harm']:.6f}"
                dd.addItem(lbl)
                self._period_choices[lbl] = float(periods["alt_harm"])

            if periods.get("gls"):
                lbl = f"Raw GLS peak: {periods['gls']:.6f}"
                dd.addItem(lbl)
                self._period_choices[lbl] = float(periods["gls"])

            if periods.get("bls"):
                lbl = f"Raw BLS peak: {periods['bls']:.6f}"
                dd.addItem(lbl)
                self._period_choices[lbl] = float(periods["bls"])

            # ilk öğedeki değeri Best-Period kutucuğuna yaz
            first_lbl = dd.itemText(0)
            self.best_period_display.setText(f"{self._period_choices[first_lbl]:.6f}")

            # sinyali yeniden aç
            dd.blockSignals(False)
            dd.setEnabled(True)  # <<< bu satırı ekleyin

            QMessageBox.information(self, "Automate Phase Folding",
                                    "Automated phase folding completed successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"An error occurred during automated phase folding: {e}")

    def update_phase_plot(self):
        ax = self.figure_phase.gca()
        ax.clear()
        if self.phase_time is not None and self.phase_flux is not None:
            if self.phase_error_checkbox.isChecked() and self.phase_flux_err is not None:
                ax.errorbar(self.phase_time, self.phase_flux, yerr=self.phase_flux_err, fmt='.',
                            color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
            else:
                ax.plot(self.phase_time, self.phase_flux, '.', color="dodgerblue", markersize=5)
            ax.set_xlabel("Phase")
            ax.set_ylabel("Normalized Flux")
            ax.set_title("Phase-Folded Light Curve")
            ax.set_xlim(0.2, 1.2)
            self.figure_phase.tight_layout()  # << EKLENDİ

            self.canvas_phase.draw()

    def detach_phase_plot(self):
        try:
            if self.phase_time is None:
                QMessageBox.warning(self, "Warning", "No phase-folded data available to detach.")
                return
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.errorbar(self.phase_time, self.phase_flux, yerr=self.phase_flux_err, fmt='.',
                        color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
            ax.set_xlabel("Phase")
            ax.set_ylabel("Normalized Flux")
            ax.set_title("Detached Phase-Folded Light Curve")
            ax.set_xlim(0.2, 1.2)
            fig.tight_layout()
            fig.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while detaching phase plot: {e}")

    def save_phase_data(self):
        try:
            if self.phase_time is None:
                QMessageBox.warning(self, "Warning", "No phase-folded data available to save.")
                return
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
                self, f"Save Phase-Folded Data as {file_format.upper()}", "",
                f"{file_format.upper()} Files (*.{file_format})"
            )
            if not file_name:
                return
            if not file_name.endswith(f".{file_format}"):
                file_name += f".{file_format}"
            data = pd.DataFrame({
                "Phase": self.phase_time,
                "Flux": self.phase_flux,
                "Error": self.phase_flux_err
            })
            if file_format == "csv":
                data.to_csv(file_name, index=False, float_format="%.10f")
            elif file_format == "xlsx":
                with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                    data.to_excel(writer, sheet_name="Phase-Folded Data", index=False, float_format="%.10f")
            elif file_format == "txt":
                np.savetxt(
                    file_name,
                    data.values,
                    fmt="%.10f",
                    header="\t".join(data.columns),
                    delimiter="\t",
                    comments=""
                )
            QMessageBox.information(self, "Data Saved", f"Phase-folded data saved successfully to {file_name}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving phase-folded data: {e}")

    def setData(self, time, flux, flux_err):
        self.time = time.copy()
        self.flux = flux.copy()
        self.flux_err = flux_err.copy()
        self.phase_time = None
        self.phase_flux = None
        self.phase_flux_err = None


class K2DownloadThread(QThread):
    resultReady = pyqtSignal(object, object, object, object)  # t, f, e, extra

    def __init__(self, star, qres, cadence, campaign, flux_type, apply_quality_filter, parent=None):
        super().__init__(parent)
        self.star = star
        self.qres = qres
        self.cad = cadence    # e.g. "KSC" veya "KLC"
        self.camp = campaign  # kampanya numarası (int)
        self.ftype = flux_type
        self.qfilt = apply_quality_filter

    def run(self):
        try:
            # 1) Sorgu
            if self.qres is None:
                self.qres = Observations.query_criteria(
                    obs_collection="K2",
                    objectname=self.star,
                    dataproduct_type="timeseries",
                    radius=0
                )
                if len(self.qres) == 0:
                    raise RuntimeError(f"No K2 data for '{self.star}'")

            # 2) Ürünleri al ve filtrele
            prods = Observations.get_product_list(self.qres)
            # get("description") yerine p["description"] kullanıyoruz
            filt = [p for p in prods if self.cad in p["description"]]

            matches = []
            for p in filt:
                desc = p["description"]
                m = campaign_re.search(desc)
                if m and int(m.group(1)) == self.camp:
                    matches.append(p)

            if not matches:
                raise RuntimeError(f"No data for Campaign {self.camp} ({self.cad})")

            # 3) İndirme + okuma
            all_t, all_f, all_e = [], [], []
            for p in matches:
                dl_dir = load_download_dir()
                kwargs = {"mrp_only": True}
                if dl_dir is not None:
                    kwargs["download_dir"] = str(dl_dir)
                dl = Observations.download_products(p, **kwargs)
                path = dl["Local Path"][0]
                with fits.open(path) as hd:
                    t = hd[1].data["TIME"]
                    f = hd[1].data[self.ftype]
                    e = hd[1].data[f"{self.ftype}_ERR"]
                    q = hd[1].data["SAP_QUALITY"]
                    if self.qfilt:
                        mask = (q == 0) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
                    else:
                        mask = np.isfinite(t) & np.isfinite(f)

                    all_t.append(t[mask])
                    all_f.append(f[mask])
                    all_e.append(e[mask])

            # 4) Birleştir & sırala
            t = np.concatenate(all_t)
            f = np.concatenate(all_f)
            e = np.concatenate(all_e)
            idx = np.argsort(t)

            # 5) Sonucu emit et
            self.resultReady.emit(t[idx], f[idx], e[idx], self.qres)

        except Exception as ex:
            self.resultReady.emit(None, None, None, ex)


class K2QueryTab(QWidget):
    def __init__(self, screen_width, screen_height, star_name=None, parent=None):
        super().__init__(parent)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.star_name = star_name

        self.query_result = None
        self.time = self.flux = self.flux_err = None

        self.error_checkbox = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.init_ui()
    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_grp = QGroupBox("Data Query Controls")
        tl = QVBoxLayout(top_grp)

        sr = QHBoxLayout()
        self.star_name_input = QLineEdit()
        self.star_name_input.setPlaceholderText("Object Name")
        sr.addWidget(self.star_name_input)

        self.k2_search_btn = QPushButton("Search K2 Data")
        self.k2_search_btn.setMinimumWidth(300)
        self.k2_search_btn.clicked.connect(self.check_k2_data)
        sr.addWidget(self.k2_search_btn)

        self.quality_nan_checkbox = QCheckBox("Filter Data")
        self.quality_nan_checkbox.setChecked(True)
        sr.addWidget(self.quality_nan_checkbox)
        tl.addLayout(sr)


        cr = QHBoxLayout()
        self.cadence_combo = QComboBox()
        self.cadence_combo.setEditable(True)
        self.cadence_combo.lineEdit().setPlaceholderText("Cadence (KSC = short, KLC = long)")
        self.cadence_combo.currentIndexChanged.connect(self.update_campaign_list)
        cr.addWidget(self.cadence_combo)

        self.campaign_combo = QComboBox()
        self.campaign_combo.setEditable(True)
        self.campaign_combo.lineEdit().setPlaceholderText("Campaign (e.g. 12)")
        cr.addWidget(self.campaign_combo)


        self.flux_type_combo = QComboBox()
        self.flux_type_combo.addItems(["SAP FLUX", "PDCSAP FLUX"])
        cr.addWidget(self.flux_type_combo)

        tl.addLayout(cr)
        top_grp.setFixedHeight(100)
        main_layout.addWidget(top_grp)

        self.download_and_plot_btn = QPushButton("Download and Plot Data")
        self.download_and_plot_btn.clicked.connect(self.download_and_plot)
        self.download_and_plot_btn.setEnabled(False)
        main_layout.addWidget(self.download_and_plot_btn)

        for w in (self.star_name_input, self.cadence_combo.lineEdit(), self.campaign_combo.lineEdit()):
            w.textChanged.connect(self._update_download_button_state)
        self.cadence_combo.currentIndexChanged.connect(self._update_download_button_state)
        self.campaign_combo.currentIndexChanged.connect(self._update_download_button_state)

        lc_grp = QGroupBox("Light Curve")
        ll = QVBoxLayout(lc_grp)
        ll.addWidget(self.canvas)
        ll.addWidget(self.toolbar)
        lc_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.toolbar)

        self.error_checkbox = QCheckBox("Display with Errors", checked=True)
        self.error_checkbox.stateChanged.connect(self.redraw_plot)
        toolbar_row.addWidget(self.error_checkbox)
        toolbar_row.setAlignment(Qt.AlignLeft)

        ll.addLayout(toolbar_row)
        lc_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(lc_grp, stretch=1)

        br = QHBoxLayout()
        btn = QPushButton("Detached"); btn.clicked.connect(self.detach_query_plot)
        br.addWidget(btn)
        btn = QPushButton("Save Data"); btn.clicked.connect(self.save_query_data)
        br.addWidget(btn)
        main_layout.addLayout(br)
    def _update_download_button_state(self):
        ok = (
            bool(self.star_name_input.text().strip()) and
            bool(self.cadence_combo.currentText().strip()) and
            bool(self.campaign_combo.currentText().strip())
        )
        self.download_and_plot_btn.setEnabled(ok)

    def _cad_to_code(self, txt: str) -> str:
        t = txt.strip().upper()
        if "KSC" in t:
            return "KSC"
        if "KLC" in t:
            return "KLC"
        if t.startswith("SHORT"):
            return "KSC"
        if t.startswith("LONG"):
            return "KLC"
        return "KLC"
    def redraw_plot(self):
        if self.time is None:
            return

        ax = self.figure.gca()
        ax.clear()

        if self.error_checkbox.isChecked() and self.flux_err is not None:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err,
                        fmt=".", markersize=5, capsize=1, elinewidth=1.5,
                        color="dodgerblue")
        else:
            ax.plot(self.time, self.flux, ".", color="dodgerblue", markersize=5)

        ax.set_xlabel("Time – 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def check_k2_data(self):
        star_name = self.star_name_input.text().strip()
        if not star_name:
            QMessageBox.warning(self, "Warning", "Please enter a star name.")
            return

        self.k2_search_btn.setText("Searching...")
        self.k2_search_btn.setEnabled(False)
        self.k2_search_btn.setStyleSheet("QPushButton { color: red; }")
        QApplication.processEvents()

        self.search_thread = K2SearchThread(star_name)
        self.search_thread.resultReady.connect(self.on_search_finished)
        self.search_thread.start()


    def on_search_finished(self, result):
        self.k2_search_btn.setEnabled(True)
        self.k2_search_btn.setText("Search K2 Data")
        self.k2_search_btn.setStyleSheet("")  # varsayılan stile dön

        star_txt = self.star_name_input.text().strip() or "this star"
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error", f"An error occurred: {result}")
            return
        if len(result) == 0:
            QMessageBox(self, windowTitle="Search Result",
                        textFormat=Qt.RichText,
                        text=html_notfound("K2", star_txt)).exec_()
            return

        self.query_result = result
        products = Observations.get_product_list(result)

        have_ksc = any("KSC" in p["description"] for p in products)
        have_klc = any("KLC" in p["description"] for p in products)

        self.cadence_combo.blockSignals(True)
        self.cadence_combo.clear()
        if have_ksc: self.cadence_combo.addItem("Short Cadence")
        if have_klc: self.cadence_combo.addItem("Long Cadence")
        self.cadence_combo.setCurrentIndex(0)
        self.cadence_combo.setEnabled(True)
        self.campaign_combo.setEnabled(True)
        self.cadence_combo.blockSignals(False)

        self.update_campaign_list()

    def update_campaign_list(self):
        if self.query_result is None:
            return

        code = self._cad_to_code(self.cadence_combo.currentText())

        products = Observations.get_product_list(self.query_result)
        filt = [p for p in products if code in p["description"]]

        camps = sorted({int(campaign_re.search(p["description"]).group(1))
                        for p in filt if campaign_re.search(p["description"])})
        self.campaign_combo.clear()
        for c in camps:
            self.campaign_combo.addItem(str(c))


    def download_and_plot(self):
        star = self.star_name_input.text().strip()
        cad_code = self._cad_to_code(self.cadence_combo.currentText())
        try:
            camp = int(self.campaign_combo.currentText().strip())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid campaign number.")
            return

        self.download_and_plot_btn.setEnabled(False)
        self.download_and_plot_btn.setText("Downloading and Plotting…")
        # style both normal and disabled so it stays red
        self.download_and_plot_btn.setStyleSheet(
            "QPushButton, QPushButton:disabled { color: red; }"
        )
        QApplication.processEvents()

        ftype = self.flux_type_combo.currentText().replace(" ", "_")
        self._current_ftype = ftype
        self._current_camp  = camp

        self.dl_thread = K2DownloadThread(
            star=star,
            qres=self.query_result,
            cadence=cad_code,  # ⬅️ artık “KSC” veya “KLC”
            campaign=camp,
            flux_type=ftype,
            apply_quality_filter=self.quality_nan_checkbox.isChecked()
        )
        self.dl_thread.resultReady.connect(self._on_download_finished, Qt.QueuedConnection)
        self.dl_thread.start()


    def _on_download_finished(self, t, f, e, extra):
        # restore the button
        self.download_and_plot_btn.setEnabled(True)
        self.download_and_plot_btn.setText("Download and Plot Data")
        self.download_and_plot_btn.setStyleSheet("")

        # if the thread told us there was an error:
        if any(x is None for x in (t, f, e)):
            QMessageBox.critical(self, "Error", str(extra))
            return

        # otherwise, plot
        self.time, self.flux, self.flux_err = t, f, e
        self.query_result = extra   # update cache if the thread re-queried
        self.redraw_plot()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.errorbar(
            self.time,
            self.flux,
            yerr=self.flux_err,
            fmt=".",
            capsize=1,
            elinewidth=1.5,
            markersize=5,
            color="dodgerblue"
        )
        ax.set_xlabel("Time – 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")

        title = f"{self._current_ftype.replace('_',' ')} Light Curve - Campaign: {self._current_camp})"
        ax.set_title(title)

        self.figure.tight_layout()
        self.canvas.draw()
        main_win = self.window()
        if hasattr(main_win, "updateSharedData"):
            main_win.updateSharedData(self.time, self.flux, self.flux_err)

    def detach_query_plot(self):
        if self.time is None or self.flux is None or self.flux_err is None:
            QMessageBox.warning(self, "Warning", "No data available for plotting.")
            return
        fig, ax = plt.subplots()
        ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".", color="dodgerblue",
                    markersize=5, capsize=1, elinewidth=1.5)
        ax.set_xlabel("Time - 2454833 (day)")
        ax.set_ylabel("Flux ($electrons^{-1}$)")
        ax.set_title("Detached Light Curve - Query Data")
        fig.tight_layout()
        fig.show()

    def save_query_data(self):
        if self.time is None or self.flux is None or self.flux_err is None:
            QMessageBox.warning(self, "Warning", "No query data available to save.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Query Data")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select file format to save:"))
        file_format_combo = QComboBox()
        file_format_combo.addItems(["XLSX", "CSV", "TXT"])
        layout.addWidget(file_format_combo)
        add_checkbox = QCheckBox("Add 2454833 to Time")
        add_checkbox.setChecked(True)
        layout.addWidget(add_checkbox)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        if dialog.exec_() != QDialog.Accepted:
            return
        file_format = file_format_combo.currentText().lower()
        file_name, _ = QFileDialog.getSaveFileName(self, f"Save Query Data as {file_format.upper()}",
                                                   "", f"{file_format.upper()} Files (*.{file_format})")
        if not file_name:
            return
        if not file_name.endswith(f".{file_format}"):
            file_name += f".{file_format}"
        time_data = self.time + 2454833 if add_checkbox.isChecked() else self.time
        df = pd.DataFrame({
            "Time": time_data,
            "Flux": self.flux,
            "Error": self.flux_err
        })
        try:
            if file_format == "csv":
                df.to_csv(file_name, index=False, float_format="%.10f")
            elif file_format == "xlsx":
                with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                    df.to_excel(writer, sheet_name="Query Data", index=False, float_format="%.10f")
            elif file_format == "txt":
                np.savetxt(file_name, df.values, fmt="%.10f",
                           header="\t".join(df.columns), delimiter="\t", comments="")
            QMessageBox.information(self, "Data Saved", f"Query data saved successfully to {file_name}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving query data as {file_format.upper()}: {e}")
class KeplerApp(QWidget):
    def __init__(self, star_name=None):
        super().__init__()
        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.screen_width = self.screen_resolution.width()
        self.screen_height = self.screen_resolution.height()
        self.star_name = star_name


        self.shared_time = None
        self.shared_flux = None
        self.shared_flux_err = None

        self.init_ui()
        self.adjust_geometry()

    def init_ui(self):
        self.setWindowTitle("Kepler/K2")
        self.setWindowIcon(QIcon("images/star.ico"))

        self.topTabs = QTabWidget()

        query_tab = QWidget()
        query_layout = QVBoxLayout(query_tab)
        self.querySubTabs = QTabWidget()
        self.kepler_query_tab = KeplerQueryTab(screen_width=self.screen_width, screen_height=self.screen_height,
                                               star_name=self.star_name)
        self.querySubTabs.addTab(self.kepler_query_tab, "Kepler")
        self.k2_query_tab = K2QueryTab(screen_width=self.screen_width, screen_height=self.screen_height,
                                       star_name=self.star_name)
        self.querySubTabs.addTab(self.k2_query_tab, "K2")
        query_layout.addWidget(self.querySubTabs)
        query_tab.setLayout(query_layout)

        self.outlier_tab = KeplerOutlierTab()
        self.detrending_tab = KeplerDetrendingTab()
        self.phase_fold_tab = KeplerPhaseFoldTab()

        self.topTabs.addTab(query_tab, "Query")
        self.topTabs.addTab(self.outlier_tab, "Outlier")
        self.topTabs.addTab(self.detrending_tab, "Detrending")
        self.topTabs.addTab(self.phase_fold_tab, "Phase Fold")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.topTabs)
        self.setLayout(main_layout)

    def adjust_geometry(self):
        if self.screen_width == 2560 and self.screen_height == 1440:
            self.setGeometry(400, 300, 1000, 800)
        elif self.screen_width == 1920 and self.screen_height == 1080:
            self.setGeometry(300, 200, 1000, 800)
        else:
            self.setGeometry(300, 200, 1000, 800)
        self.setMinimumSize(1000, 800)

    def updateSharedData(self, time, flux, flux_err):
        self.shared_time = time
        self.shared_flux = flux
        self.shared_flux_err = flux_err
        self.outlier_tab.setData(time, flux, flux_err)
        self.detrending_tab.setData(time, flux, flux_err)
        self.phase_fold_tab.setData(time, flux, flux_err)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KeplerApp()
    window.show()
    sys.exit(app.exec_())
