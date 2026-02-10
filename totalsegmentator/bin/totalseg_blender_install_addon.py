#!/usr/bin/env python3
"""
Install and enable a Blender add-on from a .py file, then save user prefs.

Usage:
  blender -b -P totalseg_blender_install_addon.py -- --addon-path /path/to/addon.py
"""

import sys
import traceback
from pathlib import Path

import bpy
import addon_utils


def parse_args():
    if "--" not in sys.argv:
        return {}
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


def _make_logger(log_file: str | None):
    log_path = Path(log_file).resolve() if log_file else None

    def _log(msg: str):
        print(msg)
        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg.rstrip() + "\n")
            except Exception:
                pass

    return _log


def main():
    args = parse_args()
    addon_path = args.get("addon-path", "")
    log_file = args.get("log-file", "")
    log = _make_logger(log_file if log_file else None)
    log(f"🧾 Install script: {Path(__file__).resolve()}")
    log(f"🧾 argv: {sys.argv}")
    if not addon_path:
        log("❌ ERROR: --addon-path is required")
        sys.exit(1)

    addon_file = Path(addon_path).resolve()
    if not addon_file.exists():
        log(f"❌ ERROR: Addon file not found: {addon_file}")
        sys.exit(1)

    module_name = addon_file.stem
    log(f"📦 Installing addon: {addon_file}")
    log(f"🔧 Module name: {module_name}")
    try:
        log(f"🧩 Blender version: {bpy.app.version_string}")
    except Exception:
        pass

    # Remove any existing addon file with the same name in the user addons folder
    try:
        scripts_dir = Path(bpy.utils.user_resource("SCRIPTS")).resolve()
        addons_dir = scripts_dir / "addons"
        log(f"📂 User addons dir: {addons_dir}")
        addons_dir.mkdir(parents=True, exist_ok=True)
        existing = addons_dir / addon_file.name
        if existing.exists():
            existing.unlink()
            log(f"🧹 Removed old addon: {existing}")
    except Exception as e:
        log(f"❌ Failed to resolve/prepare user addons dir: {e}")
        log(traceback.format_exc())
        sys.exit(2)

    try:
        bpy.ops.preferences.addon_install(filepath=str(addon_file), overwrite=True)
        log("✅ Addon install op completed")
    except Exception as e:
        log(f"❌ Addon install failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    # Refresh addon list and verify install destination.
    try:
        addon_utils.modules(refresh=True)
        log("✅ addon_utils.modules(refresh=True) completed")
    except Exception as e:
        log(f"⚠️ addon_utils.modules(refresh=True) failed: {e}")

    try:
        installed_path = addons_dir / addon_file.name
        log(f"📄 Installed addon file exists: {installed_path.exists()} ({installed_path})")
        if not installed_path.exists():
            log("❌ Addon file not found in user addons dir after install")
            sys.exit(2)
    except Exception as e:
        log(f"⚠️ Failed to verify installed addon file: {e}")

    try:
        bpy.ops.preferences.addon_enable(module=module_name)
        log("✅ Addon enable op completed")
    except Exception as e:
        log(f"❌ Addon enable failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    try:
        addon_utils.enable(module_name, default_set=True)
        log("✅ addon_utils.enable(default_set=True) completed")
    except Exception as e:
        log(f"⚠️ addon_utils.enable failed: {e}")

    try:
        enabled_keys = list(bpy.context.preferences.addons.keys())
        log(f"📌 Prefs enabled addons: {', '.join(enabled_keys) if enabled_keys else '(none)'}")
        if module_name not in enabled_keys:
            log(f"❌ Addon not found in prefs after enable: {module_name}")
            sys.exit(3)
    except Exception:
        pass

    try:
        mods = [m.__name__ for m in addon_utils.modules()]
        log(f"🧩 addon_utils.modules: {', '.join(sorted(mods)) if mods else '(none)'}")
    except Exception:
        pass

    try:
        bpy.ops.wm.save_userpref()
        log("✅ User preferences saved")
    except Exception as e:
        log(f"⚠️ Failed to save user prefs: {e}")


if __name__ == "__main__":
    main()
