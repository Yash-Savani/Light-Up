"""Unit tests for power_model.py — run with: python test_power_model.py"""
import time
from power_model import (
    PowerSimulator, DEVICE_PROFILES, voltage_from_sod, NetworkFSM,
)


def test_energy_balance():
    """SOD increase must equal integrated power / capacity (Eq. 4)."""
    sim = PowerSimulator(device_name="Mobile laptop (demo host, x86)")
    sim._last_tick -= 4.0                        # pretend 4 s elapsed (< clamp)
    bd = sim.tick(cpu_util_percent=0.0, mode_active=False)
    p_expected = (DEVICE_PROFILES[sim.device_name]["p_cpu_idle"]
                  + DEVICE_PROFILES[sim.device_name]["p_display"])
    assert abs(bd["P_total_mW"] - p_expected) < 1e-6
    e_cap = DEVICE_PROFILES[sim.device_name]["battery_capacity_mWh"] * 3600.0
    assert abs(sim.sod - (p_expected * 4.0) / e_cap) < 1e-9


def test_tail_last_trigger():
    """Tail energy must be charged to the last module using the radio."""
    sim = PowerSimulator()
    sim.record_inference("object", 0.05, net_duration_s=0.01)
    sim.record_inference("face", 0.05, net_duration_s=0.01)   # last trigger
    sim.net.busy_until = time.time() - 0.1                    # force TAIL
    sim.net.tail_until = time.time() + 10.0
    sim._last_tick -= 2.0
    sim.tick(0.0, True)
    assert sim.energy_n["face"] > 0.0 and sim.energy_n["object"] == 0.0


def test_voltage_curve_monotonic():
    v = [voltage_from_sod(s / 100.0) for s in range(101)]
    assert all(a >= b for a, b in zip(v, v[1:]))
    assert abs(v[0] - 4.20) < 1e-6 and abs(v[-1] - 3.40) < 1e-6


def test_fsm_transitions():
    fsm = NetworkFSM()
    now = time.time()
    fsm.on_transfer("scene", duration=0.2, now=now, tail_time=1.0)
    assert fsm.advance(now + 0.1) == "HIGH"
    assert fsm.advance(now + 0.5) == "TAIL"
    assert fsm.advance(now + 5.0) == "LOW"


def test_device_switch_preserves_sod():
    sim = PowerSimulator()
    sim.sod = 0.25
    sim.set_device("Microprocessor board (Raspberry Pi 5 class)")
    assert sim.sod == 0.25


if __name__ == "__main__":
    for fn in [test_energy_balance, test_tail_last_trigger,
               test_voltage_curve_monotonic, test_fsm_transitions,
               test_device_switch_preserves_sod]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("All power-model tests passed.")
