from PyQt5.QtCore import QTimer


def show_info(label, text: str, duration_ms: int = 4000, color: str = "green") -> None:
    """Set a temporary info message on a QLabel."""
    if label is None:
        return
    label.setStyleSheet(f"color: {color};")
    label.setText(text)
    QTimer.singleShot(duration_ms, label.clear)
