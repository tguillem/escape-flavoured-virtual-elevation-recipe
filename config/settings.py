import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.file_handling import get_config_dir, get_results_dir, load_json, save_json


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas Timestamps and other non-serializable types"""

    def default(self, obj):
        # Handle pandas Timestamp objects
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        # Handle numpy arrays and other numpy types
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle NaN, Inf
        elif pd.isna(obj):
            return None
        else:
            try:
                return super().default(obj)
            except TypeError:
                # Return string representation as fallback
                return str(obj)


class Settings:
    def __init__(self):
        self.app_dir: Path = get_config_dir()  # <── only change
        self.settings_file: Path = self.app_dir / "settings.json"

        # ---------- defaults ----------
        self.last_file: str = ""
        self.last_dem_file: str = ""
        self.last_rho_file: str = ""
        self.result_dir: str = str(get_results_dir())  # <── new sensible default

        self.load_settings()

    def load_settings(self) -> None:
        """
        Load persisted settings (if any) and overwrite the defaults in-place.
        If the file is missing or unreadable we silently keep the defaults.
        """
        data = load_json(self.settings_file)
        if not data:  # None or empty dict
            return

        # Safely pull each key in case the JSON is missing fields
        self.last_file = data.get("last_file", self.last_file)
        self.result_dir = data.get("result_dir", self.result_dir)
        self.last_dem_file = data.get("last_dem_file", self.last_dem_file)
        self.last_rho_file = data.get("last_rho_file", self.last_rho_file)

    def save_settings(self) -> None:
        """
        Persist the current in-memory settings to the JSON file in the
        per-OS config directory.
        """
        data = {
            "last_file": self.last_file,
            "result_dir": self.result_dir,
            "last_dem_file": self.last_dem_file,
            "last_rho_file": self.last_rho_file,
        }
        try:
            save_json(data, self.settings_file)
        except OSError as exc:
            # You can log/print or raise if you have a logger:
            # logger.error("Could not save settings: %s", exc)
            print(f"[Settings] Could not save settings: {exc}")

    def get_file_settings(self, filename):
        """Load settings for a specific FIT file"""
        file_settings_path = self.app_dir / f"{Path(filename).stem}_settings.json"
        default_settings = {
            "system_mass": 90,
            "rho": 1.2,
            "cda": None,
            "crr": None,
            "cda_min": 0.150,
            "cda_max": 0.500,
            "crr_min": 0.0010,
            "crr_max": 0.0300,
            "eta": 0.98,
            "wind_speed": None,
            "wind_direction": None,
            "auto_lap_detection": "None",
        }

        if file_settings_path.exists():
            try:
                with open(file_settings_path, "r") as f:
                    saved_settings = json.load(f)
                    # Update defaults with saved settings
                    default_settings.update(saved_settings)
            except:
                pass

        return default_settings

    def save_file_settings(self, filename, settings):
        """Save settings for a specific FIT file"""
        file_settings_path = self.app_dir / f"{Path(filename).stem}_settings.json"
        with open(file_settings_path, "w") as f:
            json.dump(settings, f, cls=JSONEncoder)
