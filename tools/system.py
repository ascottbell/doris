"""
System information tools for Doris.
Battery status, storage info, and system details via macOS APIs.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


def get_battery_status() -> dict:
    """
    Get battery status using pmset (macOS power management).

    Returns:
        dict with charge_percent, is_charging, power_source, time_remaining
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            return {"error": "Could not get battery status"}

        output = result.stdout

        # Parse output like:
        # Now drawing from 'AC Power'
        #  -InternalBattery-0 (id=...)    100%; charged; 0:00 remaining

        info = {
            "charge_percent": None,
            "is_charging": False,
            "is_charged": False,
            "power_source": "unknown",
            "time_remaining": None
        }

        lines = output.strip().split("\n")

        # First line has power source
        if "AC Power" in lines[0]:
            info["power_source"] = "ac"
            info["is_charging"] = True
        elif "Battery Power" in lines[0]:
            info["power_source"] = "battery"
            info["is_charging"] = False

        # Second line has battery details
        if len(lines) > 1:
            battery_line = lines[1]

            # Extract percentage
            if "%" in battery_line:
                pct_part = battery_line.split("%")[0]
                pct = "".join(filter(str.isdigit, pct_part.split()[-1]))
                if pct:
                    info["charge_percent"] = int(pct)

            # Check charging status
            if "charging" in battery_line.lower():
                info["is_charging"] = True
            if "charged" in battery_line.lower():
                info["is_charged"] = True
                info["is_charging"] = False
            if "discharging" in battery_line.lower():
                info["is_charging"] = False

            # Extract time remaining
            if "remaining" in battery_line:
                # Format is like "2:30 remaining"
                parts = battery_line.split()
                for i, part in enumerate(parts):
                    if part == "remaining" and i > 0:
                        time_str = parts[i-1]
                        if ":" in time_str:
                            info["time_remaining"] = time_str
                        break

        return info

    except Exception as e:
        return {"error": str(e)}


def get_storage_info() -> dict:
    """
    Get disk storage information.

    Returns:
        dict with total_gb, used_gb, free_gb, percent_used
    """
    try:
        # Get disk usage for root filesystem
        usage = shutil.disk_usage("/")

        total_gb = round(usage.total / (1024**3), 1)
        used_gb = round(usage.used / (1024**3), 1)
        free_gb = round(usage.free / (1024**3), 1)
        percent_used = round((usage.used / usage.total) * 100, 1)

        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "percent_used": percent_used
        }

    except Exception as e:
        return {"error": str(e)}


def get_system_info() -> dict:
    """
    Get general system information.

    Returns:
        dict with hostname, os_version, uptime
    """
    import platform

    info = {
        "hostname": platform.node(),
        "os_version": platform.mac_ver()[0],
        "processor": platform.processor(),
        "architecture": platform.machine()
    }

    # Get uptime
    try:
        result = subprocess.run(
            ["uptime"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse uptime output
            output = result.stdout.strip()
            # Format: "18:30  up 5 days, 3:42, 2 users, load averages: 1.23 1.45 1.67"
            if "up" in output:
                up_part = output.split("up")[1].split(",")[0].strip()
                info["uptime"] = up_part
    except:
        pass

    return info


def get_wifi_network() -> Optional[str]:
    """Get the current WiFi network name."""
    try:
        # Use airport command on macOS
        result = subprocess.run(
            ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "SSID:" in line and "BSSID:" not in line:
                    return line.split("SSID:")[1].strip()
        return None

    except Exception:
        return None


def format_battery(battery: dict) -> str:
    """Format battery status as natural language."""
    if battery.get("error"):
        return "I couldn't check the battery status."

    pct = battery.get("charge_percent")

    # Mac Mini or desktop - no battery
    if pct is None and battery.get("power_source") == "ac":
        return "Running on AC power, no battery."

    if pct is None:
        return "Battery status unavailable."

    parts = [f"Battery is at {pct} percent"]

    if battery.get("is_charged"):
        parts.append("and fully charged")
    elif battery.get("is_charging"):
        parts.append("and charging")
        if battery.get("time_remaining"):
            parts.append(f"with {battery['time_remaining']} until full")
    else:
        if battery.get("time_remaining"):
            parts.append(f"with {battery['time_remaining']} remaining")

    return ", ".join(parts) + "."


def format_storage(storage: dict) -> str:
    """Format storage info as natural language."""
    if storage.get("error"):
        return "I couldn't check storage status."

    free = storage.get("free_gb", 0)
    total = storage.get("total_gb", 0)
    pct = storage.get("percent_used", 0)

    if free < 20:
        return f"Storage is getting low, only {free} gigs free out of {total}. {pct} percent used."
    else:
        return f"You have {free} gigs free out of {total} total. {pct} percent used."


if __name__ == "__main__":
    print("Battery Status:")
    battery = get_battery_status()
    print(f"  {battery}")
    print(f"  Formatted: {format_battery(battery)}")

    print("\nStorage Info:")
    storage = get_storage_info()
    print(f"  {storage}")
    print(f"  Formatted: {format_storage(storage)}")

    print("\nSystem Info:")
    system = get_system_info()
    print(f"  {system}")

    print(f"\nWiFi Network: {get_wifi_network()}")
