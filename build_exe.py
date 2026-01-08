import PyInstaller.__main__
import os

# Use relative paths to avoid issues with special characters in absolute paths
APP_PY = "app.py"
BEST_PT = "best.pt"
RUN_APP = "run_app.py"

# PyInstaller arguments
args = [
    '--noconfirm',
    '--onedir',
    '--windowed',
    '--name=BottleCounter',
    '--clean',
    '--collect-all=streamlit',
    '--collect-all=ultralytics',
    f'--add-data={APP_PY};.',
    f'--add-data={BEST_PT};.',
    '--hidden-import=streamlit',
    '--hidden-import=ultralytics',
    RUN_APP
]

print("Running PyInstaller with args:")
for arg in args:
    print(arg)

PyInstaller.__main__.run(args)
