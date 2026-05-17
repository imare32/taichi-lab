# Taichi Lab for Blender

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue?logo=python&logoColor=white)
![Blender](https://img.shields.io/badge/Blender-4.x%20%7C%205.1+-E87D0D?logo=blender&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Installation - bundle taichi blender extension with wheels

### 1. Download Taichi Wheels

Before installing the addon, you need to download the appropriate Taichi wheel for your Blender version:

**For Blender 5.1 and later (Python 3.13):**
```bash
pip download taichi==1.7.4 --python-version 3.13 --platform win_amd64 --only-binary=:all: -d ./wheels
```

**For Blender 4.x (Python 3.11):**
```bash
pip download taichi==1.7.4 --python-version 3.11 --platform win_amd64 --only-binary=:all: -d ./wheels
```

You can also run both commands to have wheels for both Python versions — Blender will automatically select the correct one.

Remove the downloaded numpy wheels from the wheels folder. (**Blender has its own built-in numpy. Remove it to avoid conflicts.**)

```
├─taichi_addon_test
    │  auto_load.py
    │  blender_manifest.toml
    │  ... (other feature files)
    │  __init__.py
    │
    ├─wheels (numpy removed)
    │      colorama-0.4.6-py2.py3-none-any.whl
    │      dill-0.4.1-py3-none-any.whl
    │      markdown_it_py-4.0.0-py3-none-any.whl
    │      mdurl-0.1.2-py3-none-any.whl
    │      pygments-2.19.2-py3-none-any.whl
    │      rich-14.3.3-py3-none-any.whl
    │      taichi-1.7.4-cp311-cp311-win_amd64.whl
    │      taichi-1.7.4-cp313-cp313-win_amd64.whl
```

### 2. Install the Addon

1. Open Blender and go to **Edit > Preferences > Add-ons**.
2. Click **Install** and select the `taichi_addon_test` folder.
3. Enable the addon.