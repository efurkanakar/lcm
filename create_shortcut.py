"""Creates a menu and desktop shortcut that starts lcm with the Python running this script, without a console window."""
import subprocess
import sys
from pathlib import Path

LCM_DIR = Path(__file__).resolve().parent
SCRIPT = LCM_DIR / "lcm.py"

if sys.platform == "win32":
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    quote = lambda p: str(p).replace("'", "''")
    command = (
        "foreach ($folder in [Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')) {"
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(\"$folder\\lcm.lnk\");"
        f"$s.TargetPath = '{quote(pythonw)}';"
        f"$s.Arguments = '\"{quote(SCRIPT)}\"';"
        f"$s.WorkingDirectory = '{quote(LCM_DIR)}';"
        f"$s.IconLocation = '{quote(LCM_DIR / 'images' / 'star.ico')}';"
        "$s.Save(); Write-Output \"$folder\\lcm.lnk\" }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
elif sys.platform.startswith("linux"):
    entry = Path.home() / ".local" / "share" / "applications" / "lcm.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=lcm\nComment=Light Curve Manager\n"
        f'Exec="{sys.executable}" "{SCRIPT}"\nPath={LCM_DIR}\nIcon={LCM_DIR / "images" / "lc1.png"}\n'
        "Terminal=false\nCategories=Science;Astronomy;\nStartupWMClass=lcm.py\n"
    )
    entry.chmod(0o755)
    print(entry)
else:
    sys.exit(f"Shortcuts are not supported on {sys.platform}; start lcm with: python lcm.py")
