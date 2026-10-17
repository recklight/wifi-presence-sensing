"""Synthetic captures and tables for tests, examples and pipeline dry runs.

Nothing here is measurement data.  The simulator produces beamforming reports
with the properties that make real captures hard to analyse, so every
safeguard of the pipeline has something to act on:

* a static multipath channel (a few reflected paths with their own delays,
  angles and gains) over the 30 reported subcarriers;
* an optional person: a scattered path whose length follows the body's small
  movements and breathing, plus partial shadowing of the line of sight;
* AGC steps that change the raw amplitude without changing the channel;
* 8-bit quantization, per-packet antenna permutations, occasional
  single-transmit-chain packets, implausible noise-floor values, jittered
  packet timing with optional pauses, and a 32-bit timestamp that can wrap.

The model is deliberately simple; it is a test bench for the software, not a
claim about how strongly real rooms respond to people.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path

import numpy as np

from .capture.beamform_report import write_beamform_file
from .capture.dataset import DatasetManifest, RecordingEntry
from .capture.rate_flags import decode_rate_flags, encode_rate_flags
from .capture.snapshot import TIMESTAMP_PERIOD_US, ChannelSnapshot, RadioMeta
from .capture.subcarriers import subcarrier_offsets_hz

_LOGGER = logging.getLogger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0

#: Antenna permutations a report can carry (0-based physical antenna per column).
_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(permutations(range(3)))


@dataclass(frozen=True)
class RoomGeometry:
    """A rectangular room with one transmitter and one receiver.

    Attributes
    ----------
    width_m, depth_m : float
        Room size.
    transmitter_xy, receiver_xy : tuple of float
        Antenna array positions.
    carrier_hz : float
        Channel centre frequency.
    received_power_dbm : float
        Average received power of an empty room.
    n_static_paths : int
        Reflected paths besides the line of sight.
    """

    width_m: float = 8.0
    depth_m: float = 6.0
    transmitter_xy: tuple[float, float] = (0.5, 3.0)
    receiver_xy: tuple[float, float] = (7.5, 3.0)
    carrier_hz: float = 2.437e9
    received_power_dbm: float = -50.0
    n_static_paths: int = 6


@dataclass(frozen=True)
class ChannelEnvironment:
    """Static propagation paths of one session (furniture, walls).

    Attributes
    ----------
    lengths_m : numpy.ndarray
        Path lengths, line of sight first.
    gains : numpy.ndarray
        Complex path gains (relative).
    departure_rad, arrival_rad : numpy.ndarray
        Angles at the transmit and receive arrays.
    power_offset_db : float
        Session-to-session level difference.
    """

    lengths_m: np.ndarray
    gains: np.ndarray
    departure_rad: np.ndarray
    arrival_rad: np.ndarray
    power_offset_db: float = 0.0


@dataclass(frozen=True)
class CaptureScenario:
    """What happens during one simulated recording.

    Attributes
    ----------
    duration_s : float
        Recording length.
    packet_rate_hz : float
        Mean packet rate.
    person_xy : tuple of float or None
        Position of a person in the room; ``None`` for an empty room.
    motion_amplitude_m : float
        Amplitude of the person's small movements.
    n_tx, n_rx : int
        Chains.
    bandwidth_mhz : int
        20 or 40.
    snr_db : float
        Thermal SNR of the channel estimate.
    agc_step_probability : float
        Per-packet probability that the AGC changes setting.
    single_chain_fraction : float
        Fraction of packets reported with one transmit chain.
    noise_outlier_fraction : float
        Fraction of packets with an implausible noise-floor value.
    pause_probability : float
        Per-packet probability of a 1.5-3 s pause after the packet.
    start_timestamp_us : int
        First NIC timestamp (set near ``2**32`` to exercise the wrap).
    """

    duration_s: float = 30.0
    packet_rate_hz: float = 20.0
    person_xy: tuple[float, float] | None = None
    motion_amplitude_m: float = 0.05
    n_tx: int = 2
    n_rx: int = 3
    bandwidth_mhz: int = 20
    snr_db: float = 30.0
    agc_step_probability: float = 0.02
    single_chain_fraction: float = 0.02
    noise_outlier_fraction: float = 0.05
    pause_probability: float = 0.0
    start_timestamp_us: int = 0


def make_environment(geometry: RoomGeometry, seed: int) -> ChannelEnvironment:
    """Draw the static multipath of a session."""
    rng = np.random.default_rng(seed)
    tx = np.asarray(geometry.transmitter_xy)
    rx = np.asarray(geometry.receiver_xy)
    direct = float(np.linalg.norm(rx - tx))
    extra = rng.uniform(1.0, 12.0, geometry.n_static_paths)
    lengths = np.concatenate([[direct], direct + extra])
    reflection = np.concatenate([[1.0], rng.uniform(0.25, 0.7, geometry.n_static_paths)])
    phases = rng.uniform(0, 2 * np.pi, lengths.size)
    gains = reflection * (direct / lengths) * np.exp(1j * phases)
    departure = np.concatenate([[0.0], rng.uniform(-np.pi / 2, np.pi / 2, geometry.n_static_paths)])
    arrival = np.concatenate([[0.0], rng.uniform(-np.pi / 2, np.pi / 2, geometry.n_static_paths)])
    return ChannelEnvironment(
        lengths_m=lengths,
        gains=gains,
        departure_rad=departure,
        arrival_rad=arrival,
        power_offset_db=float(rng.normal(0.0, 1.0)),
    )


def _steering(n_tx: int, n_rx: int, departure: np.ndarray, arrival: np.ndarray) -> np.ndarray:
    """Half-wavelength array phases, shaped ``(..., n_tx, n_rx)``."""
    t = np.arange(n_tx)
    r = np.arange(n_rx)
    phase_tx = np.pi * np.sin(departure)[..., None] * t
    phase_rx = np.pi * np.sin(arrival)[..., None] * r
    return np.exp(-1j * (phase_tx[..., :, None] + phase_rx[..., None, :]))


def _channel(
    scenario: CaptureScenario,
    geometry: RoomGeometry,
    environment: ChannelEnvironment,
    times: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Relative channel ``(n_packets, n_tx, n_rx, n_sc)`` and the empty room's mean power.

    The empty-room power is the reference that fixes the absolute scale, so a
    person's shadowing and scattering still change the received power.
    """
    frequencies = geometry.carrier_hz + subcarrier_offsets_hz(scenario.bandwidth_mhz)
    n = times.size
    # Static paths with a slow, small gain drift (temperature, air flow).
    drift = 1.0 + 0.004 * np.cumsum(
        rng.normal(0.0, 1.0, (n, environment.gains.size)), axis=0
    ) / np.sqrt(max(n, 1))
    gains = environment.gains[None, :] * drift  # (n, paths)
    shadow = np.ones(n)
    person_term = np.zeros((n, scenario.n_tx, scenario.n_rx, frequencies.size), dtype=np.complex128)
    if scenario.person_xy is not None:
        tx = np.asarray(geometry.transmitter_xy)
        rx = np.asarray(geometry.receiver_xy)
        base = np.asarray(scenario.person_xy, dtype=np.float64)
        f_sway = rng.uniform(0.2, 0.5, 2)
        phase = rng.uniform(0, 2 * np.pi, 2)
        sway = scenario.motion_amplitude_m * np.sin(2 * np.pi * f_sway * times[:, None] + phase)
        wander = 0.02 * np.cumsum(rng.normal(0.0, 1.0, (n, 2)), axis=0) / np.sqrt(max(n, 1))
        position = base + sway + wander
        breathing = 0.005 * np.sin(2 * np.pi * rng.uniform(0.2, 0.33) * times)
        to_tx = np.linalg.norm(position - tx, axis=1)
        to_rx = np.linalg.norm(position - rx, axis=1)
        length = to_tx + to_rx + 2.0 * breathing
        direct = float(np.linalg.norm(rx - tx))
        amplitude = 0.45 * direct / length
        departure = np.arctan2(position[:, 1] - tx[1], position[:, 0] - tx[0])
        arrival = np.arctan2(position[:, 1] - rx[1], rx[0] - position[:, 0])
        delay_phase = np.exp(-2j * np.pi * np.outer(length, frequencies) / SPEED_OF_LIGHT_M_S)
        steering = _steering(scenario.n_tx, scenario.n_rx, departure, arrival)
        person_term = (amplitude[:, None, None, None] * steering[..., None]) * delay_phase[
            :, None, None, :
        ]
        # Partial shadowing of the line of sight when the body is near it.
        segment = rx - tx
        along = np.clip(((position - tx) @ segment) / (segment @ segment), 0.0, 1.0)
        closest = tx + along[:, None] * segment
        distance = np.linalg.norm(position - closest, axis=1)
        shadow = 1.0 - 0.5 * np.exp(-((distance / 0.6) ** 2))
    gains = gains.copy()
    gains[:, 0] *= shadow

    path_phase = np.exp(
        -2j * np.pi * np.outer(environment.lengths_m, frequencies) / SPEED_OF_LIGHT_M_S
    )  # (paths, sc)
    steering = _steering(
        scenario.n_tx, scenario.n_rx, environment.departure_rad, environment.arrival_rad
    )
    static = np.einsum("np,pab,ps->nabs", gains, steering, path_phase)
    empty_room = np.einsum("p,pab,ps->abs", environment.gains, steering, path_phase)
    reference_power = float((np.abs(empty_room) ** 2).sum() / frequencies.size)
    return static + person_term, reference_power


def simulate_capture(
    scenario: CaptureScenario,
    geometry: RoomGeometry,
    environment: ChannelEnvironment,
    *,
    seed: int,
) -> list[ChannelSnapshot]:
    """Simulate the beamforming reports of one recording.

    Parameters
    ----------
    scenario : CaptureScenario
        What happens during the recording.
    geometry : RoomGeometry
        Room and antennas.
    environment : ChannelEnvironment
        Static multipath of the session.
    seed : int
        Seed of every random draw.

    Returns
    -------
    list of ChannelSnapshot
        Reports with integer CSI, ready for
        :func:`~wifi_presence_sensing.capture.beamform_report.write_beamform_file`.
    """
    rng = np.random.default_rng(seed)
    expected = int(scenario.duration_s * scenario.packet_rate_hz)
    intervals = rng.gamma(16.0, 1.0 / (16.0 * scenario.packet_rate_hz), expected)
    pauses = rng.random(expected) < scenario.pause_probability
    intervals = intervals + pauses * rng.uniform(1.5, 3.0, expected)
    times = np.concatenate([[0.0], np.cumsum(intervals)[:-1]])
    times = times[times <= scenario.duration_s]
    n = times.size

    channel, reference_power = _channel(scenario, geometry, environment, times, rng)
    target_mw = 10.0 ** ((geometry.received_power_dbm + environment.power_offset_db) / 10.0)
    channel *= np.sqrt(target_mw / reference_power)
    entry_power = np.mean(np.abs(channel) ** 2)
    sigma = np.sqrt(entry_power / 10.0 ** (scenario.snr_db / 10.0) / 2.0)
    channel = channel + sigma * (
        rng.normal(size=channel.shape) + 1j * rng.normal(size=channel.shape)
    )

    # Per-chain received power (mW) and the AGC that keeps raw values in 8 bits.
    chain_mw = (np.abs(channel) ** 2).sum(axis=(1, 3)) / channel.shape[-1]  # (n, n_rx)
    mean_chain_dbm = 10.0 * np.log10(chain_mw.mean())
    agc_nominal = int(np.clip(round(30.0 - 44.0 - mean_chain_dbm), 10, 60))
    steps = rng.random(n) < scenario.agc_step_probability
    offsets = np.cumsum(np.where(steps, rng.choice([-3, -2, 2, 3], n), 0))
    agc = np.clip(agc_nominal + np.clip(offsets, -6, 6), 0, 255).astype(np.int64)
    entry_rms = np.sqrt(entry_power)
    raw_gain = 22.0 / entry_rms * 10.0 ** ((agc - agc_nominal) / 20.0)
    raw = channel * raw_gain[:, None, None, None]
    raw = np.clip(np.rint(raw.real), -128, 127) + 1j * np.clip(np.rint(raw.imag), -128, 127)
    rssi = np.clip(np.rint(10.0 * np.log10(chain_mw) + 44.0 + agc[:, None]), 1, 255).astype(int)

    noise = np.rint(rng.normal(-92.0, 1.0, n)).astype(int)
    outliers = rng.random(n) < scenario.noise_outlier_fraction
    noise[outliers] = rng.integers(-80, -59, int(outliers.sum()))
    single = rng.random(n) < scenario.single_chain_fraction
    permutation_index = rng.integers(0, len(_PERMUTATIONS), n)
    timestamps = (
        scenario.start_timestamp_us + np.rint(times * 1e6).astype(np.int64)
    ) % TIMESTAMP_PERIOD_US

    snapshots: list[ChannelSnapshot] = []
    for i in range(n):
        n_tx = 1 if (single[i] and scenario.n_tx > 1) else scenario.n_tx
        mcs = 8 * (n_tx - 1) + 7
        rssi_triplet = tuple(int(v) for v in rssi[i]) + (0,) * (3 - scenario.n_rx)
        meta = RadioMeta(
            report_counter=i & 0xFFFF,
            n_tx=n_tx,
            n_rx=scenario.n_rx,
            rssi_db=rssi_triplet[:3],  # type: ignore[arg-type]
            noise_dbm=int(noise[i]),
            agc_db=int(agc[i]),
            antenna_permutation=_PERMUTATIONS[permutation_index[i]][: scenario.n_rx]
            if scenario.n_rx == 3
            else tuple(range(scenario.n_rx)),
            permutation_valid=True,
            rate=decode_rate_flags(encode_rate_flags(mcs, bandwidth_mhz=scenario.bandwidth_mhz)),
        )
        snapshots.append(
            ChannelSnapshot(csi=raw[i, :n_tx], timestamp_us=int(timestamps[i]), meta=meta)
        )
    return snapshots


def simple_snapshot(
    *, seed: int = 0, n_tx: int = 2, n_rx: int = 3, timestamp_us: int = 1_000_000
) -> ChannelSnapshot:
    """A single report with random 8-bit CSI, for doctests and unit tests.

    Examples
    --------
    >>> s = simple_snapshot(seed=1)
    >>> s.csi.shape, s.meta.rate.mcs
    ((2, 3, 30), 15)
    """
    rng = np.random.default_rng(seed)
    csi = rng.integers(-60, 61, (n_tx, n_rx, 30)) + 1j * rng.integers(-60, 61, (n_tx, n_rx, 30))
    permutation = tuple(int(v) for v in rng.permutation(n_rx))
    meta = RadioMeta(
        report_counter=seed & 0xFFFF,
        n_tx=n_tx,
        n_rx=n_rx,
        rssi_db=(30, 32, 31),
        noise_dbm=-92,
        agc_db=35,
        antenna_permutation=permutation,
        permutation_valid=True,
        rate=decode_rate_flags(encode_rate_flags(8 * (n_tx - 1) + 7)),
    )
    return ChannelSnapshot(csi=csi, timestamp_us=timestamp_us, meta=meta)


def _session_environment(
    base: ChannelEnvironment, seed: int, variation: float
) -> ChannelEnvironment:
    """The same room on another day: a few paths changed, a slightly different level.

    ``variation`` scales the change: 1 moves two reflected paths to arbitrary
    phases (furniture rearranged), 0 reproduces the base room exactly.
    """
    rng = np.random.default_rng(seed)
    gains = base.gains.copy()
    moved = rng.choice(np.arange(1, gains.size), size=min(2, gains.size - 1), replace=False)
    phase = rng.uniform(-np.pi, np.pi, moved.size) * variation
    scale = rng.uniform(1.0 - 0.3 * variation, 1.0 + 0.3 * variation, moved.size)
    gains[moved] *= np.exp(1j * phase) * scale
    offset = float(rng.normal(0.0, 0.5)) * variation
    return replace(base, gains=gains, power_offset_db=base.power_offset_db + offset)


def write_synthetic_dataset(
    output_dir: str | Path,
    *,
    kind: str = "occupancy",
    n_sessions: int = 3,
    duration_s: float = 40.0,
    packet_rate_hz: float = 20.0,
    grid_shape: tuple[int, int] = (3, 3),
    seed: int = 0,
    geometry: RoomGeometry | None = None,
    session_variation: float | None = None,
) -> Path:
    """Write simulated capture files and a dataset manifest.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Destination folder.
    kind : {"occupancy", "localization"}, optional
        ``occupancy``: one empty and one occupied recording per session.
        ``localization``: one recording per grid position per session.
    n_sessions : int, optional
        Sessions (cross-validation groups).
    duration_s : float, optional
        Length of each recording.
    packet_rate_hz : float, optional
        Mean packet rate.
    grid_shape : (int, int), optional
        Reference positions along x and y (localization).
    seed : int, optional
        Master seed.
    geometry : RoomGeometry, optional
        Room; defaults to :class:`RoomGeometry`.
    session_variation : float, optional
        How much the static multipath changes between sessions, from 0 (none)
        to 1 (reflectors rearranged).  Defaults to 1 for occupancy and 0.05 for
        localization, where a radio map is only useful if the room it was
        surveyed in still exists.

    Returns
    -------
    pathlib.Path
        The manifest file.
    """
    if kind not in ("occupancy", "localization"):
        raise ValueError(f"kind must be 'occupancy' or 'localization', got {kind!r}")
    if n_sessions < 1:
        raise ValueError("n_sessions must be at least 1")
    room = geometry or RoomGeometry()
    default_variation = 1.0 if kind == "occupancy" else 0.05
    variation = default_variation if session_variation is None else session_variation
    if not 0.0 <= variation <= 1.0:
        raise ValueError(f"session_variation must lie in [0, 1], got {variation}")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(seed).generate_state(4 + 64 * n_sessions)
    base = make_environment(room, int(seeds[0]))
    rng = np.random.default_rng(int(seeds[1]))

    entries: list[RecordingEntry] = []
    cursor = 2
    for session in range(n_sessions):
        environment = _session_environment(base, int(seeds[cursor]), variation)
        cursor += 1
        session_name = f"session{session + 1}"
        if kind == "occupancy":
            plans = [
                ("empty", None),
                (
                    "occupied",
                    (
                        float(rng.uniform(1.5, room.width_m - 1.5)),
                        float(rng.uniform(1.0, room.depth_m - 1.0)),
                    ),
                ),
            ]
        else:
            xs = np.linspace(1.5, room.width_m - 1.5, grid_shape[0])
            ys = np.linspace(1.0, room.depth_m - 1.0, grid_shape[1])
            plans = [
                (f"position{index:02d}", (float(x), float(y)))
                for index, (x, y) in enumerate((x, y) for x in xs for y in ys)
            ]
        for name, person in plans:
            scenario = CaptureScenario(
                duration_s=duration_s,
                packet_rate_hz=packet_rate_hz,
                person_xy=person,
                # Occupancy: a person moving about.  Localization: standing still,
                # swaying a centimetre or so and breathing.
                motion_amplitude_m=0.05 if kind == "occupancy" else 0.01,
                start_timestamp_us=int(TIMESTAMP_PERIOD_US - rng.integers(1_000_000, 20_000_000)),
                pause_probability=0.001,
            )
            snapshots = simulate_capture(scenario, room, environment, seed=int(seeds[cursor]))
            cursor += 1
            path = folder / session_name / f"{name}.dat"
            write_beamform_file(path, snapshots)
            label = name if kind == "occupancy" else "occupied"
            entries.append(
                RecordingEntry(
                    path=path.resolve(),
                    label=label,
                    session=session_name,
                    position=person if kind == "localization" else None,
                )
            )
    manifest = DatasetManifest(
        name=f"synthetic-{kind}",
        recordings=tuple(entries),
        description=f"simulated {kind} captures ({n_sessions} sessions, seed {seed})",
    )
    destination = manifest.to_yaml(folder / "manifest.yaml")
    _LOGGER.info("wrote %d synthetic recording(s) and %s", len(entries), destination)
    return destination


def synthetic_amplitude_table(
    *,
    label_values: tuple[tuple[int, ...], tuple[int, ...]] = ((2, 3, 4), (1, 2, 3)),
    rows_per_block: int = 40,
    n_streams: int = 6,
    n_subcarriers: int = 30,
    n_sentinels: int = 10,
    sentinel: float = 65535.0,
    seed: int = 0,
) -> np.ndarray:
    """A headerless amplitude table in the layout read by ``load_amplitude_table``.

    Rows come in contiguous blocks, one per combination of the two label
    columns.  The class in the first label column changes how strongly the
    amplitudes fluctuate over time (its mean level barely), and ``n_sentinels``
    cells are overwritten with the sentinel.

    Returns
    -------
    numpy.ndarray
        ``(n_blocks * rows_per_block, 2 + n_streams * n_subcarriers)``.
    """
    rng = np.random.default_rng(seed)
    first, second = label_values
    profile = 18.0 + 6.0 * rng.normal(size=(n_streams, n_subcarriers))
    rows = []
    for level, a in enumerate(first):
        for b in second:
            t = np.arange(rows_per_block)[:, None, None]
            frequency = rng.uniform(0.05, 0.2)
            swing = 0.3 + 1.2 * level
            motion = swing * np.sin(
                2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi, (1, n_streams, 1))
            )
            values = (
                profile
                + 0.1 * level
                + motion
                + 0.3 * rng.normal(size=(rows_per_block, n_streams, n_subcarriers))
            )
            labels = np.tile([a, b], (rows_per_block, 1))
            rows.append(np.hstack([labels, values.reshape(rows_per_block, -1)]))
    table = np.vstack(rows)
    if n_sentinels:
        r = rng.integers(0, table.shape[0], n_sentinels)
        c = rng.integers(2, table.shape[1], n_sentinels)
        table[r, c] = sentinel
    return table
