from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Belgian grid carbon intensity (kgCO2eq/kWh) — matches CodeCarbon's BEL value
_BEL_CARBON_INTENSITY = 0.167

try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:
    EmissionsTracker = None

try:
    from carbontracker.tracker import CarbonTracker as _CarbonTracker  # type: ignore
    import carbontracker.parser as _ct_parser  # type: ignore
except Exception:
    _CarbonTracker = None
    _ct_parser = None


# --- Error metrics ---

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Returns MAPE in percentage points (e.g. 2.5 means 2.5%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# --- Energy measurement ---

@dataclass
class EnergyResult:
    """Result returned by any energy meter's stop() call."""
    energy_kwh: float
    emissions_kg: float
    duration_s: float
    tool: str = "codecarbon"
    details_path: Optional[str] = None
    # CarbonTracker companion fields
    ct_energy_kwh: float = float("nan")
    ct_emissions_kg: float = float("nan")
    ct_details_path: Optional[str] = None
    # nvidia-smi GPU fields
    nv_energy_kwh: float = float("nan")
    nv_emissions_kg: float = float("nan")
    nv_details_path: Optional[str] = None

    def with_ct(self, ct_result: "EnergyResult") -> "EnergyResult":
        return EnergyResult(
            energy_kwh=self.energy_kwh,
            emissions_kg=self.emissions_kg,
            duration_s=self.duration_s,
            tool=f"{self.tool}+carbontracker" if self.tool != "disabled" else "carbontracker",
            details_path=self.details_path,
            ct_energy_kwh=ct_result.energy_kwh,
            ct_emissions_kg=ct_result.emissions_kg,
            ct_details_path=ct_result.details_path,
            nv_energy_kwh=self.nv_energy_kwh,
            nv_emissions_kg=self.nv_emissions_kg,
            nv_details_path=self.nv_details_path,
        )

    def with_nv(self, nv_result: "EnergyResult") -> "EnergyResult":
        if nv_result.tool == "disabled":
            new_tool = self.tool
        elif self.tool != "disabled":
            new_tool = f"{self.tool}+nvidia_smi"
        else:
            new_tool = "nvidia_smi"
        return EnergyResult(
            energy_kwh=self.energy_kwh,
            emissions_kg=self.emissions_kg,
            duration_s=self.duration_s,
            tool=new_tool,
            details_path=self.details_path,
            ct_energy_kwh=self.ct_energy_kwh,
            ct_emissions_kg=self.ct_emissions_kg,
            ct_details_path=self.ct_details_path,
            nv_energy_kwh=nv_result.energy_kwh,
            nv_emissions_kg=nv_result.emissions_kg,
            nv_details_path=nv_result.details_path,
        )


class EnergyMeter:
    """CodeCarbon wrapper. Reads emissions.csv directly to stay robust across API versions."""

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        country_iso_code: str = "BEL",
        enabled: bool = True,
    ):
        self.enabled = bool(enabled and (EmissionsTracker is not None))
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.country_iso_code = country_iso_code
        self._tracker = None
        self._t0: Optional[float] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._emissions_csv = self.output_dir / "emissions.csv"

    def prepare(self) -> None:
        """Build the CodeCarbon tracker without starting it.

        Construction does hardware detection and takes seconds; calling this
        ahead of time keeps that cost out of any window that brackets start().
        Idempotent. No-op when disabled.
        """
        if not self.enabled or self._tracker is not None:
            return

        base_kwargs = dict(
            project_name=self.project_name,
            output_dir=str(self.output_dir),
            log_level="error",
            save_to_file=True,
        )
        # Some CodeCarbon versions don't accept country_iso_code
        try:
            self._tracker = EmissionsTracker(**base_kwargs, country_iso_code=self.country_iso_code)
        except TypeError:
            self._tracker = EmissionsTracker(**base_kwargs)

    def start(self) -> None:
        self._t0 = time.time()
        if not self.enabled:
            return
        if self._tracker is None:
            self.prepare()  # unprepared caller: construction lands inside the window
        self._tracker.start()

    def _read_last_emissions_row(self) -> Tuple[float, float, Optional[str]]:
        if not self._emissions_csv.exists():
            return float("nan"), float("nan"), None
        try:
            df = pd.read_csv(self._emissions_csv)
            if len(df) == 0:
                return float("nan"), float("nan"), str(self._emissions_csv)
            last = df.iloc[-1]
            energy_kwh = float(last["energy_consumed"]) if "energy_consumed" in df.columns else float("nan")
            emissions_kg = float(last["emissions"]) if "emissions" in df.columns else float("nan")
            return energy_kwh, emissions_kg, str(self._emissions_csv)
        except Exception:
            return float("nan"), float("nan"), str(self._emissions_csv)

    def stop(self) -> EnergyResult:
        duration_s = time.time() - (self._t0 or time.time())

        if not self.enabled or self._tracker is None:
            return EnergyResult(
                energy_kwh=float("nan"),
                emissions_kg=float("nan"),
                duration_s=duration_s,
                tool="disabled",
                details_path=None,
            )

        stop_return = None
        try:
            stop_return = self._tracker.stop()
        except Exception:
            pass

        energy_kwh, emissions_kg, details = self._read_last_emissions_row()

        # Fallback if CSV read failed
        if np.isnan(emissions_kg) and isinstance(stop_return, (int, float)):
            emissions_kg = float(stop_return)

        return EnergyResult(
            energy_kwh=energy_kwh,
            emissions_kg=emissions_kg,
            duration_s=duration_s,
            tool="codecarbon",
            details_path=details,
        )

    def cpu_mode(self) -> Optional[str]:
        """CPU tracking mode CodeCarbon chose: 'RAPL', 'cpu_load', 'constant', ...

        Read off the CPU hardware object CodeCarbon attached to the tracker
        (its ResourceTracker is not kept). None when disabled or unknown.
        """
        if not self.enabled or self._tracker is None:
            return None
        try:
            for hw in getattr(self._tracker, "_hardware", []):
                if type(hw).__name__ == "CPU":
                    mode = getattr(hw, "_mode", None)
                    if mode is None:
                        return None
                    return "RAPL" if mode == "intel_rapl" else str(mode)
        except Exception:
            pass
        return None


class CarbonTrackerMeter:
    """CarbonTracker wrapper. Degrades gracefully when sudo powermetrics is unavailable."""

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled and (_CarbonTracker is not None))
        self.project_name = project_name
        # Each meter gets its own subdirectory to avoid log-file conflicts
        self.log_dir = Path(output_dir) / project_name
        self._tracker = None
        self._t0: Optional[float] = None

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> None:
        """Build the CarbonTracker object without opening an epoch.

        Its monitor thread starts at construction but collects nothing until
        epoch_start() raises the epoch counter above zero. Idempotent. No-op
        when disabled.
        """
        if not self.enabled or self._tracker is not None:
            return
        self._tracker = _CarbonTracker(
            epochs=1,
            log_dir=str(self.log_dir),
            log_file_prefix=self.project_name,
            verbose=0,
            ignore_errors=True,
            update_interval=1,  # 1s cadence: Heimdall is Linux/RAPL (no powermetrics overhead) and short fits need sub-15s sampling
        )

    def start(self) -> None:
        self._t0 = time.time()
        if not self.enabled:
            return
        if self._tracker is None:
            self.prepare()  # unprepared caller: construction lands inside the window
        self._tracker.epoch_start()

    def _monitoring_succeeded(self, std_log_path: str) -> bool:
        """Check if any hardware component reported a non-None power reading."""
        import re as _re
        try:
            with open(std_log_path, "r") as fh:
                text = fh.read()
        except OSError:
            return False

        pattern = _re.compile(r"Average power usage \(W\) for \w+:\s*(.+)")
        matches = pattern.findall(text)
        if not matches:
            return False
        return any(m.strip() != "None" for m in matches)

    def _parse_results(self) -> Tuple[float, float, Optional[str]]:
        """Parse energy from CarbonTracker logs. Returns NaN if monitoring failed."""
        try:
            # get_all_logs returns (output_logs, standard_logs)
            _, std_logs = _ct_parser.get_all_logs(str(self.log_dir))
            if not std_logs:
                return float("nan"), float("nan"), str(self.log_dir)

            if not any(self._monitoring_succeeded(f) for f in std_logs):
                return float("nan"), float("nan"), str(self.log_dir)

            total_energy, total_co2_g, _ = _ct_parser.aggregate_consumption(str(self.log_dir))
            energy_kwh = float(total_energy) if total_energy is not None else float("nan")
            # CarbonTracker reports CO2 in grams; convert to kg
            emissions_kg = float(total_co2_g) / 1000.0 if total_co2_g is not None else float("nan")
            return energy_kwh, emissions_kg, str(self.log_dir)
        except Exception:
            return float("nan"), float("nan"), str(self.log_dir)

    def stop(self) -> EnergyResult:
        duration_s = time.time() - (self._t0 or time.time())

        if not self.enabled or self._tracker is None:
            return EnergyResult(
                energy_kwh=float("nan"),
                emissions_kg=float("nan"),
                duration_s=duration_s,
                tool="disabled",
                details_path=None,
            )

        try:
            self._tracker.epoch_end()
            self._tracker.stop()
        except Exception:
            pass

        energy_kwh, emissions_kg, details = self._parse_results()

        return EnergyResult(
            energy_kwh=energy_kwh,
            emissions_kg=emissions_kg,
            duration_s=duration_s,
            tool="carbontracker",
            details_path=details,
        )

    def cpu_avg_w(self) -> Optional[float]:
        """CarbonTracker's average CPU power for this phase, in watts.

        Parsed from the last ``Average power usage (W) for cpu: X`` line of
        the standard log. Returns 0.0 if the err log carries the RAPL
        read-permissions failure, None if X is ``None``, the log is absent,
        or the meter is disabled.
        """
        import re as _re
        if not self.enabled:
            return None
        try:
            for path in self.log_dir.glob("*_carbontracker_err.log"):
                if "lack of read-permissions" in path.read_text(errors="replace"):
                    return 0.0
        except OSError:
            return None
        pattern = _re.compile(r"Average power usage \(W\) for cpu:\s*(\S+)")
        value: Optional[float] = None
        try:
            for path in sorted(self.log_dir.glob("*_carbontracker.log")):
                for m in pattern.findall(path.read_text(errors="replace")):
                    if m == "None":
                        value = None
                    else:
                        try:
                            value = float(m)
                        except ValueError:
                            value = None
        except OSError:
            return None
        return value


class NvidiaSmiMeter:
    """GPU energy meter via nvidia-smi polling. Auto-disables if nvidia-smi is not found."""

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        gpu_index: str = "0",
        poll_interval_ms: int = 100,
        carbon_intensity_kg_per_kwh: float = _BEL_CARBON_INTENSITY,
        enabled: bool = True,
    ):
        self._nvidia_smi_available = shutil.which("nvidia-smi") is not None
        self.enabled = bool(enabled and self._nvidia_smi_available)
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.gpu_index = gpu_index
        self.poll_interval_ms = poll_interval_ms
        self.carbon_intensity = carbon_intensity_kg_per_kwh
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Tuple[float, float]] = []  # (timestamp_s, watts)
        self._lock = threading.Lock()
        self._t0: Optional[float] = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self._t0 = time.time()
        if not self.enabled:
            return

        self._samples = []
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw.instant",
            "--format=csv,noheader,nounits",
            "-lms", str(self.poll_interval_ms),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        for line in self._proc.stdout:
            t_now = time.time()
            line = line.strip()
            if not line:
                continue
            try:
                watts = float(line)
                with self._lock:
                    self._samples.append((t_now, watts))
            except ValueError:
                pass  # "N/A" on first tick or malformed output

    def _integrate(self, samples: List[Tuple[float, float]]) -> float:
        """Integrate (time, watts) pairs into kWh using the trapezoidal rule."""
        if not samples:
            return float("nan")
        if len(samples) == 1:
            return samples[0][1] * (self.poll_interval_ms / 1000.0) / 3_600_000
        times = np.array([s[0] for s in samples])
        watts = np.array([s[1] for s in samples])
        energy_wh = float(np.trapz(watts, times)) / 3600  # W·s → W·h
        return energy_wh / 1000.0  # W·h → kW·h

    def stop(self) -> EnergyResult:
        duration_s = time.time() - (self._t0 or time.time())

        if not self.enabled or self._proc is None:
            return EnergyResult(
                energy_kwh=float("nan"),
                emissions_kg=float("nan"),
                duration_s=duration_s,
                tool="disabled",
            )

        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

        with self._lock:
            samples = list(self._samples)

        raw_path = self.output_dir / f"{self.project_name}_gpu_power.csv"
        try:
            with open(raw_path, "w") as fh:
                fh.write("timestamp_s,power_w\n")
                for ts, w in samples:
                    fh.write(f"{ts:.3f},{w:.3f}\n")
            details = str(raw_path)
        except OSError:
            details = None

        energy_kwh = self._integrate(samples)
        emissions_kg = float("nan") if np.isnan(energy_kwh) else energy_kwh * self.carbon_intensity

        return EnergyResult(
            energy_kwh=energy_kwh,
            emissions_kg=emissions_kg,
            duration_s=duration_s,
            tool="nvidia_smi",
            details_path=details,
        )


class BmcPowerMeter:
    """Chassis-level power via the Dell PowerEdge ACPI power meter (hwmon).

    Reads `power1_average` (microwatts, ~2 s hardware averaging, world-readable)
    on a background thread and integrates by the trapezoidal rule.

    Unlike CodeCarbon and CarbonTracker, this is a *measurement* of the whole
    chassis rather than a model: it therefore includes every other tenant on the
    machine. Two baseline windows — before and after the run — are sampled at the
    same cadence so the workload's own contribution can be separated:

        gross_kwh       integral of P over the run window (whole chassis)
        incremental_kwh integral of max(P - baseline_mean, 0) over the run window

    `incremental_kwh` is the honest figure for "what did this job cost"; it is
    only trustworthy when the baseline is quiet and stable, which is what
    `baseline_std` is reported for. On a busy shared host both numbers should be
    treated as upper bounds.
    """

    _HWMON_ROOT = "/sys/class/hwmon"
    _SENSOR_NAME = "power_meter"
    _SENSOR_FILE = "power1_average"

    @classmethod
    def find_sensor(cls) -> Path:
        """Locate the ACPI power meter by scanning hwmon names, not by hardcoding."""
        root = Path(cls._HWMON_ROOT)
        if not root.is_dir():
            raise RuntimeError(
                f"BMC power meter unavailable: {root} does not exist. "
                f"This host has no hwmon subsystem."
            )
        seen = []
        for name_file in sorted(root.glob("*/name")):
            try:
                name = name_file.read_text().strip()
            except OSError:
                continue
            seen.append(f"{name_file.parent.name}={name}")
            if name == cls._SENSOR_NAME:
                sensor = name_file.parent / cls._SENSOR_FILE
                if not sensor.exists():
                    raise RuntimeError(
                        f"BMC power meter unavailable: found hwmon node "
                        f"'{cls._SENSOR_NAME}' at {name_file.parent} but it has no "
                        f"{cls._SENSOR_FILE}."
                    )
                if not os.access(sensor, os.R_OK):
                    raise RuntimeError(
                        f"BMC power meter unavailable: {sensor} is not readable "
                        f"by this user."
                    )
                return sensor
        raise RuntimeError(
            f"BMC power meter unavailable: no hwmon node named "
            f"'{cls._SENSOR_NAME}' under {root}. Found: {', '.join(seen) or 'none'}. "
            f"Run without --bmc, or check that the ipmi/acpi_power_meter module is loaded."
        )

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        poll_interval_s: float = 2.0,
        baseline_before_s: float = 60.0,
        baseline_after_s: float = 60.0,
        carbon_intensity_kg_per_kwh: float = _BEL_CARBON_INTENSITY,
        enabled: bool = True,
        cooldown_s: float = 0.0,
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.poll_interval_s = float(poll_interval_s)
        self.baseline_before_s = float(baseline_before_s)
        self.baseline_after_s = float(baseline_after_s)
        self.carbon_intensity = carbon_intensity_kg_per_kwh
        self.enabled = bool(enabled)
        # Pause between the end of the run and the after-window, so the
        # after-baseline is not taken while fans and GPU are still winding
        # down. Sampled and labelled "cooldown", excluded from the baseline.
        self.cooldown_s = max(0.0, float(cooldown_s))

        self._sensor: Optional[Path] = None
        self._sensor_error: Optional[str] = None
        if self.enabled:
            # Resolve eagerly so a missing sensor fails at construction, not
            # silently at the end of a long run.
            self._sensor = self.find_sensor()

        # (timestamp_s, watts, dt_s); dt_s is the time since the previous
        # sample, i.e. the interval this reading is charged for.
        self._run_samples: List[Tuple[float, float, float]] = []
        self._before: List[Tuple[float, float, float]] = []
        self._after: List[Tuple[float, float, float]] = []
        self._cooldown: List[Tuple[float, float, float]] = []
        self._prev_t: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None
        self._summary: dict = {}

    # --- sampling ---

    def _read_watts(self) -> Optional[float]:
        try:
            raw = self._sensor.read_text().strip()
        except OSError:
            return None
        try:
            return float(raw) / 1e6  # microwatts -> watts
        except ValueError:
            return None

    def _sample(self) -> Optional[Tuple[float, float, float]]:
        """One (timestamp, watts, dt) reading; dt is the time since the previous sample.

        The sensor's power1_average is a ~2 s hardware average, so each
        reading reflects the preceding 2 s regardless of when it is taken.
        Caller holds _lock.
        """
        w = self._read_watts()
        t_now = time.time()
        if w is None:
            return None
        dt = (t_now - self._prev_t) if self._prev_t is not None else 0.0
        self._prev_t = t_now
        return (t_now, w, dt)

    def _sample_window(self, duration_s: float) -> List[Tuple[float, float, float]]:
        """Blocking sample at the poll cadence, used for the baseline windows."""
        out: List[Tuple[float, float, float]] = []
        if duration_s <= 0:
            return out
        end = time.time() + duration_s
        while time.time() < end:
            with self._lock:
                s = self._sample()
            if s is not None:
                out.append(s)
            time.sleep(self.poll_interval_s)
        return out

    def _run_loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                s = self._sample()
                if s is not None:
                    self._run_samples.append(s)
            self._stop_evt.wait(self.poll_interval_s)

    # --- lifecycle ---

    def start(self) -> None:
        self._t0 = time.time()
        if not self.enabled:
            return
        self._prev_t = None
        self._before = self._sample_window(self.baseline_before_s)
        # Boundary sample: closes the "before" window at this instant so the
        # first run sample's dt starts here.
        with self._lock:
            s = self._sample()
            if s is not None:
                self._before.append(s)
            self._run_start = time.time()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _integrate(samples: List[Tuple[float, float, float]], offset_w: float = 0.0) -> float:
        """Sum of max(p - offset, 0) * dt over (t, watts, dt_s) samples -> kWh."""
        joules = sum(max(w - offset_w, 0.0) * dt for _, w, dt in samples)
        return joules / 3.6e6  # W·s -> kWh

    def _write_csv(self) -> Optional[str]:
        path = self.output_dir / "bmc_power.csv"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            # Appended (not overwritten) so several models or phases sharing a
            # run directory each keep their samples; `phase` and `project_name`
            # disambiguate them.
            with open(path, "a") as fh:
                if new:
                    fh.write("project_name,phase,timestamp_s,elapsed_s,power_w\n")
                t0 = self._run_start or self._t0 or 0.0
                for phase, rows in (("baseline_before", self._before),
                                    ("run", self._run_samples),
                                    ("cooldown", self._cooldown),
                                    ("baseline_after", self._after)):
                    for ts, w, _dt in rows:
                        fh.write(f"{self.project_name},{phase},{ts:.3f},"
                                 f"{ts - t0:.3f},{w:.3f}\n")
            return str(path)
        except OSError:
            return None

    def stop(self) -> EnergyResult:
        duration_s = time.time() - (self._t0 or time.time())

        if not self.enabled:
            self._summary = {}
            return EnergyResult(
                energy_kwh=float("nan"), emissions_kg=float("nan"),
                duration_s=duration_s, tool="disabled",
            )

        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_s * 2))
        # Boundary sample: closes the run window at this instant.
        with self._lock:
            s = self._sample()
            if s is not None:
                self._run_samples.append(s)
            self._run_end = time.time()
            run = list(self._run_samples)

        # Cool-down: sampled for the record, never part of the baseline. A
        # boundary sample closes it so the after-window starts at that instant.
        self._cooldown = self._sample_window(self.cooldown_s)
        if self.cooldown_s > 0:
            with self._lock:
                s = self._sample()
                if s is not None:
                    self._cooldown.append(s)

        self._after = self._sample_window(self.baseline_after_s)

        base = [w for _, w, _ in self._before] + [w for _, w, _ in self._after]
        baseline_mean = float(np.mean(base)) if base else float("nan")
        baseline_std = float(np.std(base, ddof=1)) if len(base) > 1 else float("nan")

        gross_kwh = self._integrate(run)
        if base and len(run) >= 1:
            incremental_kwh = self._integrate(run, offset_w=baseline_mean)
        else:
            incremental_kwh = float("nan")

        watts = [w for _, w, _ in run]
        mean_w = float(np.mean(watts)) if watts else float("nan")
        peak_w = float(np.max(watts)) if watts else float("nan")
        run_window_s = ((self._run_end - self._run_start)
                        if (self._run_start and self._run_end) else float("nan"))

        details = self._write_csv()

        self._summary = {
            "bmc_gross_kwh":       gross_kwh,
            "bmc_incremental_kwh": incremental_kwh,
            "bmc_baseline_w":      baseline_mean,
            "bmc_baseline_std_w":  baseline_std,
            "bmc_mean_w":          mean_w,
            "bmc_peak_w":          peak_w,
            "bmc_n_samples":       len(run),
            "bmc_n_baseline_samples": len(base),
            "bmc_run_window_s":    run_window_s,
            "bmc_baseline_before_s": self.baseline_before_s,
            "bmc_baseline_after_s":  self.baseline_after_s,
            "bmc_cooldown_s":      self.cooldown_s,
            "bmc_poll_interval_s": self.poll_interval_s,
            "bmc_sensor":          str(self._sensor),
            "bmc_details_path":    details,
        }

        emissions_kg = (float("nan") if np.isnan(incremental_kwh)
                        else incremental_kwh * self.carbon_intensity)
        return EnergyResult(
            energy_kwh=gross_kwh,
            emissions_kg=emissions_kg,
            duration_s=duration_s,
            tool="bmc",
            details_path=details,
        )

    def summary(self) -> dict:
        """Full BMC statistics. Empty dict if disabled or stop() not yet called."""
        return dict(self._summary)


class RaplPowerMeter:
    """Socket-level CPU energy via Intel RAPL (powercap sysfs), packages only.

    Reads ``energy_uj`` of ``intel-rapl:0`` (package-0) and ``intel-rapl:1``
    (package-1) every ``poll_interval_s`` on one background thread, never the
    ``core`` subdomains. Consecutive readings become watts per socket with the
    counter wrap handled via ``max_energy_range_uj``; the two sockets are summed.

    Unlike :class:`BmcPowerMeter`, this meter never blocks. The thread samples
    continuously and each sample carries a phase label:

        start()      thread starts, phase "before"   (idle baseline)
        begin_run()  phase -> "run"                  (the workload)
        end_run()    phase -> "cooldown" for cooldown_s seconds, then "after"
                     (straight to "after" when cooldown_s is 0)
        stop()       thread stops, statistics computed

    Baseline mean / std come from "before" + "after"; gross and incremental
    energy from "run"; "cooldown" samples are written to the trace and used
    for nothing else. This lets the RAPL and BMC meters share identical
    windows when the BMC meter's blocking baselines bracket the run.
    Like RAPL itself, the reading covers every process on both sockets.
    """

    _RAPL_DOMAINS = (
        ("/sys/class/powercap/intel-rapl:0", "package-0"),
        ("/sys/class/powercap/intel-rapl:1", "package-1"),
    )

    @classmethod
    def find_sensors(cls) -> List[Tuple[Path, int]]:
        """Validate both package domains; return [(energy_uj path, max_range_uj)].

        Raises RuntimeError on the first domain that is missing, misnamed,
        unreadable, or non-numeric. No fallback.
        """
        out: List[Tuple[Path, int]] = []
        for d, expected in cls._RAPL_DOMAINS:
            dom = Path(d)
            if not dom.is_dir():
                raise RuntimeError(
                    f"RAPL unavailable: {dom} does not exist. Run without --rapl."
                )
            try:
                name = (dom / "name").read_text().strip()
            except OSError as e:
                raise RuntimeError(f"RAPL unavailable: cannot read {dom / 'name'}: {e}")
            if name != expected:
                raise RuntimeError(
                    f"RAPL unavailable: {dom} is named '{name}', expected '{expected}'."
                )
            energy = dom / "energy_uj"
            if not os.access(energy, os.R_OK):
                raise RuntimeError(
                    f"RAPL unavailable: {energy} is not readable by this user."
                )
            try:
                int(energy.read_text().strip())
                max_range = int((dom / "max_energy_range_uj").read_text().strip())
            except (OSError, ValueError) as e:
                raise RuntimeError(f"RAPL unavailable: cannot read {dom}: {e}")
            out.append((energy, max_range))
        return out

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        poll_interval_s: float = 2.0,
        baseline_before_s: float = 60.0,
        baseline_after_s: float = 60.0,
        carbon_intensity_kg_per_kwh: float = _BEL_CARBON_INTENSITY,
        enabled: bool = True,
        cooldown_s: float = 0.0,
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.poll_interval_s = float(poll_interval_s)
        # Nominal window lengths; the summary reports the *measured* phase
        # durations, since the caller decides when phases switch.
        self.baseline_before_s = float(baseline_before_s)
        self.baseline_after_s = float(baseline_after_s)
        self.carbon_intensity = carbon_intensity_kg_per_kwh
        self.enabled = bool(enabled)
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._t_cooldown_end: Optional[float] = None
        self._t_after_start: Optional[float] = None

        self._sensors: List[Tuple[Path, int]] = []
        if self.enabled:
            # Fail at construction, not at the end of a long run.
            self._sensors = self.find_sensors()

        # (timestamp_s, phase, package0_w, package1_w, sum_w, dt_s); dt_s is the
        # interval the power point averages over, ending at timestamp_s.
        self._samples: List[Tuple[float, str, float, float, float, float]] = []
        self._phase = "before"
        # Previous raw counter reading; shared by the thread and the
        # phase-switch samples, hence guarded by _lock.
        self._prev_t: Optional[float] = None
        self._prev_raw: Optional[Tuple[int, int]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._t_run_start: Optional[float] = None
        self._t_run_end: Optional[float] = None
        self._t_end: Optional[float] = None
        self._summary: dict = {}

    # --- sampling ---

    def _read_raw(self) -> Optional[Tuple[int, int]]:
        try:
            vals = tuple(int(p.read_text().strip()) for p, _ in self._sensors)
        except (OSError, ValueError):
            return None
        return vals  # type: ignore[return-value]

    def _take_sample(self, phase: str) -> None:
        """Read the counters now and append one power point labelled `phase`.

        The point covers the interval since the previous reading, so a
        sample taken at a phase switch closes the outgoing phase exactly at
        the switch instant and the next sample starts there. Caller holds
        _lock. The very first reading only seeds the previous value.
        """
        raw = self._read_raw()
        t_now = time.time()
        if raw is None:
            return
        if self._prev_raw is not None and t_now > self._prev_t:
            dt = t_now - self._prev_t
            watts = []
            for (_, max_range), e_new, e_old in zip(self._sensors, raw, self._prev_raw):
                d_uj = (e_new - e_old) % max_range  # counter wrap
                watts.append(d_uj / 1e6 / dt)
            self._samples.append((t_now, phase, watts[0], watts[1], watts[0] + watts[1], dt))
        self._prev_t, self._prev_raw = t_now, raw

    def _run_loop(self) -> None:
        while not self._stop_evt.is_set():
            wait_s = self.poll_interval_s
            with self._lock:
                if self._phase == "cooldown" and self._t_cooldown_end is not None:
                    if time.time() >= self._t_cooldown_end:
                        self._take_sample("cooldown")   # close "cooldown" at this instant
                        self._phase = "after"
                        self._t_after_start = time.time()
                    else:
                        self._take_sample("cooldown")
                        # wake exactly at the switch instant
                        wait_s = min(wait_s, max(self._t_cooldown_end - time.time(), 0.0))
                else:
                    self._take_sample(self._phase)
            self._stop_evt.wait(wait_s)

    # --- lifecycle ---

    def start(self) -> None:
        self._t0 = time.time()
        if not self.enabled:
            return
        self._samples = []
        self._phase = "before"
        self._prev_t, self._prev_raw = None, None
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def begin_run(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._take_sample("before")   # close "before" at this instant
            self._phase = "run"
            self._t_run_start = time.time()

    def end_run(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._take_sample("run")      # close "run" at this instant
            self._t_run_end = time.time()
            if self.cooldown_s > 0:
                self._phase = "cooldown"
                self._t_cooldown_end = self._t_run_end + self.cooldown_s
            else:
                self._phase = "after"
                self._t_after_start = self._t_run_end

    @staticmethod
    def _integrate(samples: List[Tuple[float, float]], offset_w: float = 0.0) -> float:
        """Sum of max(p - offset, 0) * dt over (watts, dt_s) pairs -> kWh.

        Each point is the counter delta over its own interval, so this is the
        exact energy between the phase-switch samples. No trapezoid, no
        assumed cadence.
        """
        joules = sum(max(w - offset_w, 0.0) * dt for w, dt in samples)
        return joules / 3.6e6  # W·s -> kWh

    def _write_csv(self, samples) -> Optional[str]:
        path = self.output_dir / "rapl_power.csv"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            # Appended, like bmc_power.csv, so several models sharing a run
            # directory each keep their samples.
            with open(path, "a") as fh:
                if new:
                    fh.write("project_name,phase,timestamp_s,elapsed_s,"
                             "package0_w,package1_w,sum_w\n")
                t0 = self._t_run_start or self._t0 or 0.0
                for ts, phase, p0, p1, ps, _dt in samples:
                    fh.write(f"{self.project_name},{phase},{ts:.3f},{ts - t0:.3f},"
                             f"{p0:.3f},{p1:.3f},{ps:.3f}\n")
            return str(path)
        except OSError:
            return None

    def stop(self) -> EnergyResult:
        duration_s = time.time() - (self._t0 or time.time())

        if not self.enabled:
            self._summary = {}
            return EnergyResult(
                energy_kwh=float("nan"), emissions_kg=float("nan"),
                duration_s=duration_s, tool="disabled",
            )

        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_s * 2))

        with self._lock:
            # close the current phase at this instant ("after" in normal use;
            # "cooldown" only if stop() arrives before the cool-down has elapsed)
            self._take_sample("cooldown" if self._phase == "cooldown" else "after")
            self._t_end = time.time()
            samples = list(self._samples)

        run = [(ps, dt) for _, ph, _, _, ps, dt in samples if ph == "run"]
        base = [ps for _, ph, _, _, ps, _ in samples if ph in ("before", "after")]
        baseline_mean = float(np.mean(base)) if base else float("nan")
        baseline_std = float(np.std(base, ddof=1)) if len(base) > 1 else float("nan")

        gross_kwh = self._integrate(run)
        if base and len(run) >= 1:
            incremental_kwh = self._integrate(run, offset_w=baseline_mean)
        else:
            incremental_kwh = float("nan")

        watts = [w for w, _ in run]
        mean_w = float(np.mean(watts)) if watts else float("nan")
        peak_w = float(np.max(watts)) if watts else float("nan")
        run_window_s = ((self._t_run_end - self._t_run_start)
                        if (self._t_run_start and self._t_run_end) else float("nan"))
        before_s = ((self._t_run_start - self._t0)
                    if (self._t0 and self._t_run_start) else float("nan"))
        after_s = ((self._t_end - self._t_after_start)
                   if (self._t_after_start and self._t_end) else float("nan"))

        details = self._write_csv(samples)

        self._summary = {
            "rapl_gross_kwh":       gross_kwh,
            "rapl_incremental_kwh": incremental_kwh,
            "rapl_baseline_w":      baseline_mean,
            "rapl_baseline_std_w":  baseline_std,
            "rapl_mean_w":          mean_w,
            "rapl_peak_w":          peak_w,
            "rapl_n_samples":       len(run),
            "rapl_n_baseline_samples": len(base),
            "rapl_run_window_s":    run_window_s,
            "rapl_baseline_before_s": before_s,
            "rapl_baseline_after_s":  after_s,
            "rapl_cooldown_s":      self.cooldown_s,
            "rapl_poll_interval_s": self.poll_interval_s,
            "rapl_sensor":          ";".join(str(p) for p, _ in self._sensors),
            "rapl_details_path":    details,
        }

        emissions_kg = (float("nan") if np.isnan(incremental_kwh)
                        else incremental_kwh * self.carbon_intensity)
        return EnergyResult(
            energy_kwh=gross_kwh,
            emissions_kg=emissions_kg,
            duration_s=duration_s,
            tool="rapl",
            details_path=details,
        )

    def summary(self) -> dict:
        """Full RAPL statistics. Empty dict if disabled or stop() not yet called."""
        return dict(self._summary)


def sample_host_conditions() -> dict:
    """Snapshot of host load at a moment in time, for the record.

    Returns 1-minute load average, machine-wide CPU utilisation, and max GPU
    utilisation. `cpu_util_percent` is psutil.cpu_percent — the same quantity
    CodeCarbon feeds its cpu_load power model — so a row's energy figure can be
    read against the conditions that produced it.

    Every field degrades to NaN rather than raising: this is diagnostic metadata
    and must never break a benchmark run.
    """
    load1 = float("nan")
    try:
        load1 = float(os.getloadavg()[0])
    except (OSError, AttributeError):
        pass

    cpu_util = float("nan")
    try:
        import psutil  # type: ignore
        cpu_util = float(psutil.cpu_percent(interval=0.3))
    except Exception:
        pass

    gpu_util = float("nan")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            vals = [float(v) for v in out.stdout.split() if v.strip().isdigit()]
            if vals:
                gpu_util = max(vals)
        except Exception:
            pass

    return {"load1": load1, "cpu_util_percent": cpu_util, "gpu_util_percent": gpu_util}


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary), p < 0.05 rejects H0."""
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna().astype(float)
    res = adfuller(s, autolag="AIC")
    return {
        "adf_stat":        float(res[0]),
        "p_value":         float(res[1]),
        "used_lag":        int(res[2]),
        "n_obs":           int(res[3]),
        "critical_values": {k: float(v) for k, v in res[4].items()},
    }


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Align by DatetimeIndex and compute MAE, RMSE, MAPE."""
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return {
        "MAE":          mae(y_true.values, y_pred.values),
        "RMSE":         rmse(y_true.values, y_pred.values),
        "MAPE_percent": mape(y_true.values, y_pred.values),
    }


def aic_bic_grid_search(
    train: pd.Series,
    p_values: list,
    d_values: list,
    q_values: list,
    seasonal: tuple,
    trend: str = "n",
    max_evals: int = 500,
) -> pd.DataFrame:
    """Grid search over ARIMA/SARIMA orders ranked by AIC."""
    import warnings as _warnings
    from src.models import ArimaConfig, fit_sarimax

    P_list, D_list, Q_list, s = seasonal
    rows = []
    evals = 0

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_list:
                    for D in D_list:
                        for Q in Q_list:
                            if evals >= max_evals:
                                break
                            cfg = ArimaConfig(
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                trend=trend,
                            )
                            try:
                                with _warnings.catch_warnings():
                                    _warnings.simplefilter("ignore")
                                    res = fit_sarimax(train, cfg, maxiter=30)
                                rows.append({
                                    "p": p, "d": d, "q": q,
                                    "P": P, "D": D, "Q": Q, "s": s,
                                    "AIC": float(res.aic),
                                    "BIC": float(res.bic),
                                })
                            except Exception as e:
                                rows.append({
                                    "p": p, "d": d, "q": q,
                                    "P": P, "D": D, "Q": Q, "s": s,
                                    "AIC": np.nan,
                                    "BIC": np.nan,
                                    "error": str(e)[:200],
                                })
                            evals += 1

    return pd.DataFrame(rows).sort_values(["AIC", "BIC"])
