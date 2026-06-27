"""
power_model.py — Component- and state-based power consumption simulation
for the LIGHT-UP Vision Assistant.

Design rationale
----------------
The previous battery_sim.py used heuristic percentage-drain functions with
no physical meaning. This module replaces that with a power model in the
style of PowerTutor / PowerBooter (Zhang et al., CODES+ISSS 2010):

    P_total = sum over components of  beta_i * state_i  +  P_idle      (1)

i.e. total power is a sum of independent per-component contributions, each
driven by an observable system state (CPU utilisation, camera on/off,
network FSM state, inference activity). Battery state-of-discharge (SOD)
is then updated from average power using the energy-balance equation of
Zhang et al. (their Eq. 4):

    P * (t2 - t1) = E * (SOD(t2) - SOD(t1))                            (2)

where E is the rated battery energy. The terminal voltage shown in the UI
is derived from SOD through a piecewise-linear Li-ion discharge curve
(Zhang et al., Sec. 5).

Asynchronous power behaviour (Pathak et al., EuroSys 2012, "eprof") is
modelled explicitly:
  * The network interface has a finite-state machine LOW -> HIGH -> TAIL
    -> LOW. After a request finishes, the radio stays in a TAIL state and
    keeps drawing power.
  * Tail energy is attributed to the *last trigger* (last module that used
    the network) — eprof's last-trigger accounting policy.
  * The camera is treated as an "exotic component": once switched on by a
    detection mode it drains power continuously until switched off.
  * Per-module energy is reported as an energy tuple (u, n) = (utilisation
    energy, tail energy), as in eprof.

All device coefficients are *profile parameters*. The shipped defaults are
order-of-magnitude values assembled from published measurements (HTC
Dream/Magic models in Zhang et al.; public SoC datasheets) scaled to the
listed devices. They are placeholders to be calibrated; see
`calibrate_speed_factor()` and the report, Sec. "Testing".

Units: power in mW, energy in mJ, time in s, battery capacity in mWh.
"""

import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Device profiles (selectable target systems, project goal 1)
# --------------------------------------------------------------------------
# beta_cpu_active : extra power when CPU is busy, scaled by utilisation [mW]
# p_cpu_idle      : baseline power of SoC + always-on logic            [mW]
# p_camera        : camera sensor + ISP while streaming                [mW]
# p_display       : display at default brightness (0 for headless)     [mW]
# p_net_high      : network interface in HIGH/active state             [mW]
# p_net_tail      : network interface in TAIL state                    [mW]
# t_net_tail      : tail duration after last transfer                  [s]
# e_inference     : energy per inference per module on this device     [mJ]
#                   (NPU/GPU/CPU package energy for one forward pass)
# speed_factor    : how much faster/slower than the host laptop the
#                   device executes one inference (t_dev = t_host / sf)

DEVICE_PROFILES = {
    "Flagship phone (Snapdragon 8 Gen 3 class)": {
        "battery_capacity_mWh": 5000 * 3.85,      # 5000 mAh @ 3.85 V
        "p_cpu_idle": 350.0,
        "beta_cpu_active": 2200.0,                # at 100 % util
        "p_camera": 700.0,
        "p_display": 900.0,
        "p_net_high": 800.0,
        "p_net_tail": 250.0,
        "t_net_tail": 4.0,                        # Wi-Fi tail; 3G/LTE longer
        "e_inference": {"object": 35.0, "face": 60.0, "scene": 450.0},
        "speed_factor": 0.8,
    },
    "Mid-range phone (Snapdragon 6 class)": {
        "battery_capacity_mWh": 5000 * 3.85,
        "p_cpu_idle": 300.0,
        "beta_cpu_active": 1500.0,
        "p_camera": 650.0,
        "p_display": 800.0,
        "p_net_high": 750.0,
        "p_net_tail": 230.0,
        "t_net_tail": 4.0,
        "e_inference": {"object": 70.0, "face": 110.0, "scene": 900.0},
        "speed_factor": 0.35,
    },
    "Microprocessor board (Raspberry Pi 5 class)": {
        "battery_capacity_mWh": 10000 * 3.7,      # USB power bank
        "p_cpu_idle": 2400.0,
        "beta_cpu_active": 5500.0,
        "p_camera": 550.0,
        "p_display": 0.0,                         # headless
        "p_net_high": 600.0,
        "p_net_tail": 150.0,
        "t_net_tail": 2.5,
        "e_inference": {"object": 220.0, "face": 350.0, "scene": 2600.0},
        "speed_factor": 0.15,
    },
    "Mobile laptop (demo host, x86)": {
        "battery_capacity_mWh": 56000.0,          # 56 Wh pack
        "p_cpu_idle": 6000.0,
        "beta_cpu_active": 18000.0,
        "p_camera": 900.0,
        "p_display": 3500.0,
        "p_net_high": 1200.0,
        "p_net_tail": 300.0,
        "t_net_tail": 2.0,
        "e_inference": {"object": 600.0, "face": 900.0, "scene": 7000.0},
        "speed_factor": 1.0,
    },
}

MODULES = ("object", "face", "scene")


# --------------------------------------------------------------------------
# Piecewise-linear Li-ion discharge curve  (SOD in [0,1] -> cell voltage)
# Shape follows Zhang et al., Fig. 7 (4.2 V full ... ~3.4 V empty).
# --------------------------------------------------------------------------
_DISCHARGE_CURVE = [
    (0.00, 4.20), (0.05, 4.06), (0.20, 3.95), (0.50, 3.80),
    (0.80, 3.70), (0.95, 3.55), (1.00, 3.40),
]


def voltage_from_sod(sod: float) -> float:
    """Inverse lookup on the piecewise-linear discharge curve."""
    sod = min(max(sod, 0.0), 1.0)
    for (s0, v0), (s1, v1) in zip(_DISCHARGE_CURVE, _DISCHARGE_CURVE[1:]):
        if s0 <= sod <= s1:
            f = (sod - s0) / (s1 - s0) if s1 > s0 else 0.0
            return v0 + f * (v1 - v0)
    return _DISCHARGE_CURVE[-1][1]


# --------------------------------------------------------------------------
# Network FSM (eprof-style asynchronous power behaviour)
# --------------------------------------------------------------------------
@dataclass
class NetworkFSM:
    """LOW -> HIGH (during transfer) -> TAIL (t_net_tail s) -> LOW."""
    state: str = "LOW"
    busy_until: float = 0.0          # HIGH until this wall-clock time
    tail_until: float = 0.0
    last_trigger: str = ""           # module that last used the radio

    def on_transfer(self, module: str, duration: float, now: float,
                    tail_time: float):
        self.state = "HIGH"
        self.busy_until = max(self.busy_until, now) + duration
        self.tail_until = self.busy_until + tail_time
        self.last_trigger = module   # last-trigger accounting policy

    def advance(self, now: float):
        if now < self.busy_until:
            self.state = "HIGH"
        elif now < self.tail_until:
            self.state = "TAIL"
        else:
            self.state = "LOW"
        return self.state


# --------------------------------------------------------------------------
# Power simulator
# --------------------------------------------------------------------------
@dataclass
class PowerSimulator:
    device_name: str = "Flagship phone (Snapdragon 8 Gen 3 class)"
    sod: float = 0.0                                  # state of discharge 0..1
    net: NetworkFSM = field(default_factory=NetworkFSM)
    camera_on: bool = False
    # per-module energy tuples (u, n): utilisation energy, tail energy [mJ]
    energy_u: dict = field(default_factory=lambda: {m: 0.0 for m in MODULES})
    energy_n: dict = field(default_factory=lambda: {m: 0.0 for m in MODULES})
    energy_baseline: float = 0.0                      # idle+display+camera [mJ]
    last_breakdown: dict = field(default_factory=dict)
    _last_tick: float = field(default_factory=time.time)

    # -- configuration -----------------------------------------------------
    @property
    def profile(self) -> dict:
        return DEVICE_PROFILES[self.device_name]

    def set_device(self, device_name: str):
        if device_name != self.device_name:
            self.device_name = device_name  # keep SOD: same mission, new HW

    # -- event hooks (called from client.py) --------------------------------
    def record_inference(self, module: str, host_latency_s: float,
                         net_duration_s: float = 0.15):
        """Account one frame->server->result cycle.

        host_latency_s : measured wall time of inference on the host
                         (server processing_time if available, else RTT)
        net_duration_s : time radio is actively transmitting/receiving
        """
        if module not in MODULES:
            return
        p = self.profile
        now = time.time()
        # 1) inference energy on the simulated device (energy/inference table)
        self.energy_u[module] += p["e_inference"][module]
        # 2) CPU pre/post-processing: device-scaled busy time at full util
        t_dev = host_latency_s / max(p["speed_factor"], 1e-3)
        cpu_overhead = 0.15 * t_dev                      # encode/decode share
        self.energy_u[module] += p["beta_cpu_active"] * cpu_overhead
        # 3) drive the network FSM; tail charged to last trigger on advance
        self.net.on_transfer(module, net_duration_s, now, p["t_net_tail"])
        self.energy_u[module] += p["p_net_high"] * net_duration_s

    # -- periodic update -----------------------------------------------------
    def tick(self, cpu_util_percent: float = 0.0, mode_active: bool = False):
        """Advance simulation by wall-clock elapsed time. Returns breakdown."""
        now = time.time()
        dt = min(max(now - self._last_tick, 0.0), 5.0)   # clamp long gaps
        self._last_tick = now
        p = self.profile

        self.camera_on = mode_active
        net_state = self.net.advance(now)

        # instantaneous component powers [mW]
        p_cpu = p["p_cpu_idle"] + p["beta_cpu_active"] * (cpu_util_percent / 100.0) * 0.2
        p_cam = p["p_camera"] if self.camera_on else 0.0
        p_disp = p["p_display"]
        p_net = {"LOW": 0.0, "HIGH": p["p_net_high"], "TAIL": p["p_net_tail"]}[net_state]

        p_total = p_cpu + p_cam + p_disp + p_net

        # energy accounting [mJ]
        self.energy_baseline += (p_cpu + p_cam + p_disp) * dt
        if net_state == "TAIL" and self.net.last_trigger in MODULES:
            # eprof last-trigger policy: tail energy -> last triggering module
            self.energy_n[self.net.last_trigger] += p_net * dt
        elif net_state == "HIGH":
            self.energy_baseline += 0.0  # active net energy booked per event

        # battery update: P*dt = E*dSOD  (Zhang et al., Eq. 4)
        e_capacity_mJ = p["battery_capacity_mWh"] * 3600.0  # 1 mWh = 3600 mJ
        self.sod = min(self.sod + (p_total * dt) / e_capacity_mJ, 1.0)

        self.last_breakdown = {
            "P_total_mW": p_total, "P_cpu_mW": p_cpu, "P_camera_mW": p_cam,
            "P_display_mW": p_disp, "P_net_mW": p_net, "net_state": net_state,
            "battery_percent": (1.0 - self.sod) * 100.0,
            "voltage_V": voltage_from_sod(self.sod),
        }
        return self.last_breakdown

    # -- reporting -----------------------------------------------------------
    def energy_report(self) -> dict:
        """Per-module energy tuples (u, n) in mJ plus baseline energy."""
        return {
            "modules": {m: (round(self.energy_u[m], 1),
                            round(self.energy_n[m], 1)) for m in MODULES},
            "baseline_mJ": round(self.energy_baseline, 1),
        }

    def battery_lifetime_estimate_h(self) -> float:
        """Naive remaining-lifetime projection at current total power."""
        p = self.last_breakdown.get("P_total_mW", 0.0)
        if p <= 0:
            return float("inf")
        remaining_mWh = self.profile["battery_capacity_mWh"] * (1.0 - self.sod)
        return remaining_mWh / p


# --------------------------------------------------------------------------
# Optional calibration helpers (battery-based model generation idea,
# Zhang et al. Sec. 5: use the built-in battery sensor instead of a meter)
# --------------------------------------------------------------------------
def read_host_battery():
    """Read the real laptop battery via psutil, if present.

    Returns (percent, secs_left, plugged) or None. Comparing the simulated
    'Mobile laptop' profile against this reading over a long interval is the
    software analogue of PowerBooter's battery-sensor-based validation.
    """
    try:
        import psutil
        b = psutil.sensors_battery()
        if b is None:
            return None
        return (b.percent, b.secsleft, b.power_plugged)
    except Exception:
        return None


def calibrate_speed_factor(host_latency_s: float, device_latency_s: float) -> float:
    """speed_factor = t_host / t_device, from measured per-inference latency."""
    if device_latency_s <= 0:
        return 1.0
    return host_latency_s / device_latency_s
