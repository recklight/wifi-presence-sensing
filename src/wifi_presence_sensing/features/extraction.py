"""From calibrated amplitude series to a labelled feature table.

The unit of analysis is a window of a uniformly sampled amplitude series.  Each
window yields four named feature groups, so every downstream model declares
which evidence it uses:

``power__``
    Mean and spread of the received power per chain (needs raw captures).
``response__``
    Time-averaged dB amplitude of every (antenna pair, subcarrier), plus the
    overall mean amplitude ``response__mean_db`` used by the single-feature
    baseline.
``profile__``
    Cross-subcarrier fluctuation metrics per antenna pair
    (:mod:`wifi_presence_sensing.features.subcarrier_profile`).
``modal__``
    Statistics of the IMFs of each antenna pair's reduced series
    (:mod:`wifi_presence_sensing.features.modal_statistics`).

Each window also records its recording, session, time block and position, so
the evaluation can keep temporally adjacent windows on the same side of every
split.  Every decomposition draws its noise from a seed derived from
``(seed, series, window)``: the table is identical whatever ``n_jobs`` is.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..decompose import decompose
from .modal_statistics import modal_feature_vector
from .subcarrier_profile import profile_feature_vector, reduce_subcarriers

if TYPE_CHECKING:
    from ..config import DecompositionConfig, FeatureConfig

_LOGGER = logging.getLogger(__name__)

#: Metadata columns of a feature table, in file order.
META_COLUMNS: tuple[str, ...] = (
    "label",
    "recording",
    "session",
    "block",
    "window_start_s",
    "window_s",
    "position_x",
    "position_y",
)

#: Metadata columns a table read from disk may lack (it is then hand-made).
OPTIONAL_META_COLUMNS: frozenset[str] = frozenset({"window_s"})

#: Separator between a feature group and the rest of a feature name.
GROUP_SEPARATOR = "__"

#: Fewest samples per window for which a decomposition is meaningful.
MIN_WINDOW_SAMPLES = 16


def stream_names(n_tx: int, n_rx: int) -> tuple[str, ...]:
    """Names of the antenna pairs in transmit-major order.

    Examples
    --------
    >>> stream_names(2, 3)
    ('tx0rx0', 'tx0rx1', 'tx0rx2', 'tx1rx0', 'tx1rx1', 'tx1rx2')
    """
    return tuple(f"tx{t}rx{r}" for t in range(n_tx) for r in range(n_rx))


@dataclass(eq=False)
class AmplitudeSeries:
    """A uniformly sampled amplitude series of one recording segment.

    Attributes
    ----------
    amplitude_db : numpy.ndarray
        ``(n_time, n_streams, n_subcarriers)`` in dB.
    sampling_rate_hz : float
        Uniform sampling rate.
    stream_names : tuple of str
        One name per antenna pair.
    label : str
        Class label of the recording.
    recording : str
        Recording identifier.
    session : str
        Session identifier.
    start_s : float
        Time of the first sample relative to the recording start.
    position : tuple of float or None
        Ground-truth position in metres.
    power_dbm : numpy.ndarray or None
        ``(n_time, n_power)`` received power per chain and in total.
    power_names : tuple of str
        Names of the power columns.
    """

    amplitude_db: np.ndarray
    sampling_rate_hz: float
    stream_names: tuple[str, ...]
    label: str
    recording: str
    session: str
    start_s: float = 0.0
    position: tuple[float, float] | None = None
    power_dbm: np.ndarray | None = None
    power_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.amplitude_db.ndim != 3:
            raise ValueError("amplitude_db must be (n_time, n_streams, n_subcarriers)")
        if self.amplitude_db.shape[1] != len(self.stream_names):
            raise ValueError("one stream name is needed per antenna pair")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.power_dbm is not None and (
            self.power_dbm.shape[0] != self.amplitude_db.shape[0]
            or self.power_dbm.shape[1] != len(self.power_names)
        ):
            raise ValueError("power_dbm must be (n_time, len(power_names))")

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.amplitude_db.shape[0])


@dataclass(eq=False)
class FeatureTable:
    """Feature matrix with per-window metadata.

    Attributes
    ----------
    features : numpy.ndarray
        ``(n_windows, n_features)``.
    feature_names : list of str
        ``"<group>__<name>"`` for every column.
    labels, recordings, sessions, blocks : numpy.ndarray
        String metadata per window.
    window_start_s : numpy.ndarray
        Start of each window relative to its recording.
    positions : numpy.ndarray
        ``(n_windows, 2)``, NaN where unknown.
    window_s : float or None
        Length of every window in seconds, recorded by
        :func:`build_feature_table` so that a later block-grouped evaluation
        purges with the length the windows really have; ``None`` for a table
        assembled by hand.
    """

    features: np.ndarray
    feature_names: list[str]
    labels: np.ndarray
    recordings: np.ndarray
    sessions: np.ndarray
    blocks: np.ndarray
    window_start_s: np.ndarray
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    window_s: float | None = None

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise ValueError("features must be (n_windows, len(feature_names))")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")
        for name in ("labels", "recordings", "sessions", "blocks", "window_start_s"):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have one entry per window")
        if self.positions.shape != (n, 2):
            raise ValueError("positions must be (n_windows, 2)")
        if self.window_s is not None and not self.window_s > 0:
            raise ValueError(f"window_s must be positive, got {self.window_s}")

    @property
    def n_windows(self) -> int:
        """Number of windows."""
        return int(self.features.shape[0])

    @property
    def has_positions(self) -> bool:
        """True when every window has a ground-truth position."""
        return bool(self.n_windows) and bool(np.all(np.isfinite(self.positions)))

    def groups(self, by: str) -> np.ndarray:
        """Cross-validation group of every window: ``recording``, ``session`` or ``block``."""
        if by == "recording":
            return self.recordings
        if by == "session":
            return self.sessions
        if by == "block":
            return self.blocks
        raise ValueError(f"unknown grouping {by!r}; choose recording, session or block")

    def column_indices(self, groups: Sequence[str] | None = None) -> list[int]:
        """Indices of the columns whose name starts with one of ``groups``."""
        if groups is None:
            return list(range(len(self.feature_names)))
        prefixes = tuple(f"{g}{GROUP_SEPARATOR}" for g in groups)
        return [i for i, name in enumerate(self.feature_names) if name.startswith(prefixes)]

    def feature_frame(self, groups: Sequence[str] | None = None) -> pd.DataFrame:
        """Features as a DataFrame with named columns, optionally restricted to groups."""
        columns = self.column_indices(groups)
        return pd.DataFrame(
            self.features[:, columns], columns=[self.feature_names[i] for i in columns]
        )

    def to_frame(self) -> pd.DataFrame:
        """Metadata and features in one DataFrame."""
        window_length = np.nan if self.window_s is None else self.window_s
        meta = pd.DataFrame(
            {
                "label": self.labels,
                "recording": self.recordings,
                "session": self.sessions,
                "block": self.blocks,
                "window_start_s": self.window_start_s,
                "window_s": np.full(self.n_windows, window_length),
                "position_x": self.positions[:, 0],
                "position_y": self.positions[:, 1],
            }
        )
        return pd.concat([meta, self.feature_frame()], axis=1)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> FeatureTable:
        """Inverse of :meth:`to_frame`.

        Raises
        ------
        ValueError
            If a required metadata column is missing or the windows disagree
            on their length.
        """
        missing = [
            c for c in META_COLUMNS if c not in frame.columns and c not in OPTIONAL_META_COLUMNS
        ]
        if missing:
            raise ValueError(f"feature table is missing metadata column(s) {missing}")
        names = [c for c in frame.columns if c not in META_COLUMNS]
        window_s: float | None = None
        if "window_s" in frame.columns:
            lengths = frame["window_s"].to_numpy(dtype=np.float64)
            known = np.unique(lengths[np.isfinite(lengths)])
            if known.size > 1:
                raise ValueError(f"windows of different lengths in one table: {known.tolist()} s")
            window_s = float(known[0]) if known.size else None
        return cls(
            features=frame[names].to_numpy(dtype=np.float64),
            feature_names=list(names),
            labels=frame["label"].astype(str).to_numpy(),
            recordings=frame["recording"].astype(str).to_numpy(),
            sessions=frame["session"].astype(str).to_numpy(),
            blocks=frame["block"].astype(str).to_numpy(),
            window_start_s=frame["window_start_s"].to_numpy(dtype=np.float64),
            positions=frame[["position_x", "position_y"]].to_numpy(dtype=np.float64),
            window_s=window_s,
        )

    def select(self, mask: np.ndarray) -> FeatureTable:
        """Rows selected by a boolean mask or an index array."""
        return FeatureTable(
            features=self.features[mask],
            feature_names=list(self.feature_names),
            labels=self.labels[mask],
            recordings=self.recordings[mask],
            sessions=self.sessions[mask],
            blocks=self.blocks[mask],
            window_start_s=self.window_start_s[mask],
            positions=self.positions[mask],
            window_s=self.window_s,
        )

    def with_block_purge(self, window_s: float, block_s: float) -> FeatureTable:
        """Assign windows to time blocks and drop those that straddle a block boundary.

        Needed before cross-validating with ``group_by: block``: a window
        overlapping two blocks shares samples with both, which would leak
        across the split.

        Parameters
        ----------
        window_s : float
            Window length the table was built with (:attr:`window_s` when the
            table records it).
        block_s : float
            Block length.

        Returns
        -------
        FeatureTable
            The retained windows, with block labels ``"<recording>#b<k>"``.
        """
        if block_s < window_s:
            raise ValueError("block_s must be at least one window long")
        first = np.floor(self.window_start_s / block_s)
        last = np.floor((self.window_start_s + window_s - 1e-9) / block_s)
        keep = first == last
        dropped = int((~keep).sum())
        if dropped:
            _LOGGER.info(
                "%d of %d window(s) straddle a %.3g s block boundary and were dropped",
                dropped,
                self.n_windows,
                block_s,
            )
        purged = self.select(keep)
        purged.blocks = np.array(
            [f"{r}#b{int(b)}" for r, b in zip(purged.recordings, first[keep], strict=True)],
            dtype=object,
        ).astype(str)
        return purged

    def to_csv(self, path: str | Path) -> Path:
        """Write the table as CSV."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False)
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> FeatureTable:
        """Read a table written by :meth:`to_csv`."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"feature table not found: {source}")
        frame = pd.read_csv(
            source, dtype={"label": str, "recording": str, "session": str, "block": str}
        )
        return cls.from_frame(frame)


def window_starts(n_samples: int, window: int, hop: int) -> np.ndarray:
    """Start indices of full windows.

    Examples
    --------
    >>> window_starts(10, 4, 3).tolist()
    [0, 3, 6]
    """
    if window < 1 or hop < 1:
        raise ValueError("window and hop must be positive")
    if n_samples < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - window + 1, hop, dtype=np.int64)


def window_features(
    window_db: np.ndarray,
    stream_labels: Sequence[str],
    sampling_rate_hz: float,
    decomposition: DecompositionConfig,
    *,
    reduction: str = "pca",
    seed: int = 0,
    power_dbm: np.ndarray | None = None,
    power_names: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """All feature groups of one window.

    Parameters
    ----------
    window_db : numpy.ndarray
        ``(n_time, n_streams, n_subcarriers)`` in dB.
    stream_labels : sequence of str
        Antenna-pair names.
    sampling_rate_hz : float
        Sampling rate of the window.
    decomposition : DecompositionConfig
        Decomposition settings.
    reduction : {"pca", "mean"}, optional
        Subcarrier reduction before decomposition.
    seed : int, optional
        Seed of this window's decompositions.
    power_dbm : numpy.ndarray, optional
        ``(n_time, n_power)`` received power.
    power_names : sequence of str, optional
        Names of the power columns.

    Returns
    -------
    values : numpy.ndarray
        Feature values.
    names : list of str
        Feature names, ``"<group>__..."``.
    """
    data = np.asarray(window_db, dtype=np.float64)
    values: list[np.ndarray] = []
    names: list[str] = []

    if power_dbm is not None:
        power = np.asarray(power_dbm, dtype=np.float64)
        values.append(np.concatenate([power.mean(axis=0), power.std(axis=0)]))
        names.extend(f"power__{name}_mean" for name in power_names)
        names.extend(f"power__{name}_std" for name in power_names)

    profile_mean = data.mean(axis=0)  # (n_streams, n_subcarriers)
    values.append(np.append(profile_mean.ravel(), profile_mean.mean()))
    names.extend(
        f"response__{stream}_k{k:02d}_db" for stream in stream_labels for k in range(data.shape[2])
    )
    names.append("response__mean_db")

    profile_values, profile_names = profile_feature_vector(data, list(stream_labels))
    values.append(profile_values)
    names.extend(profile_names)

    criterion = decomposition.stopping_criterion()
    stream_seeds = np.random.SeedSequence(seed).generate_state(len(stream_labels))
    for index, stream in enumerate(stream_labels):
        series = reduce_subcarriers(data[:, index, :], reduction)
        result = decompose(
            series,
            decomposition.method,
            n_ensemble=decomposition.n_ensemble,
            noise_scale=decomposition.noise_scale,
            max_modes=decomposition.max_modes,
            stopping=criterion,
            max_sift_iterations=decomposition.max_sift_iterations,
            seed=int(stream_seeds[index]),
            paired_noise=decomposition.paired_noise,
        )
        modal_values, modal_names = modal_feature_vector(
            result.imfs,
            result.residual,
            sampling_rate_hz,
            n_modes=decomposition.max_modes,
            prefix=f"modal__{stream}",
        )
        values.append(modal_values)
        names.extend(modal_names)

    return np.concatenate(values), names


@dataclass(frozen=True)
class _WindowTask:
    series_index: int
    start: int
    start_s: float
    block: int


def _plan_windows(
    series_list: Sequence[AmplitudeSeries], features: FeatureConfig
) -> list[_WindowTask]:
    tasks: list[_WindowTask] = []
    for index, series in enumerate(series_list):
        rate = series.sampling_rate_hz
        length = round(features.window_s * rate)
        if length < MIN_WINDOW_SAMPLES:
            raise ValueError(
                f"{series.recording}: a {features.window_s:g} s window at {rate:g} Hz holds "
                f"{length} samples; at least {MIN_WINDOW_SAMPLES} are needed for a meaningful "
                "decomposition (lengthen features.window_s)"
            )
        hop = max(1, round(features.hop_s * rate))
        for start in window_starts(series.n_samples, length, hop):
            start_s = series.start_s + start / rate
            block = int(np.floor(start_s / features.block_s))
            tasks.append(_WindowTask(index, int(start), float(start_s), block))
    return tasks


def build_feature_table(
    series_list: Sequence[AmplitudeSeries],
    *,
    decomposition: DecompositionConfig,
    features: FeatureConfig,
    seed: int,
) -> FeatureTable:
    """Window every series and extract all feature groups.

    Parameters
    ----------
    series_list : sequence of AmplitudeSeries
        Uniformly sampled series sharing antenna pairs and subcarrier count.
    decomposition : DecompositionConfig
        Decomposition settings.
    features : FeatureConfig
        Windowing settings.
    seed : int
        Master seed.

    Returns
    -------
    FeatureTable
        One row per window.

    Raises
    ------
    ValueError
        If the series are inconsistent or no window fits.
    """
    if not series_list:
        raise ValueError("no amplitude series to extract features from")
    reference = series_list[0]
    for series in series_list[1:]:
        if series.stream_names != reference.stream_names:
            raise ValueError(
                f"{series.recording}: antenna pairs {series.stream_names} differ from "
                f"{reference.recording}: {reference.stream_names}"
            )
        if series.amplitude_db.shape[2] != reference.amplitude_db.shape[2]:
            raise ValueError("all series must have the same number of subcarriers")
        if (series.power_dbm is None) != (reference.power_dbm is None) or (
            series.power_names != reference.power_names
        ):
            raise ValueError("either every series carries the same power columns or none does")

    tasks = _plan_windows(series_list, features)
    if not tasks:
        raise ValueError(
            f"no {features.window_s:g} s window fits in any series; shorten features.window_s "
            "or relax resample.max_gap_s"
        )
    _LOGGER.info("extracting features from %d window(s) of %d series", len(tasks), len(series_list))

    def job(task: _WindowTask) -> tuple[np.ndarray, list[str]]:
        series = series_list[task.series_index]
        length = round(features.window_s * series.sampling_rate_hz)
        stop = task.start + length
        window_seed = int(
            np.random.SeedSequence([seed, task.series_index, task.start]).generate_state(1)[0]
        )
        return window_features(
            series.amplitude_db[task.start : stop],
            series.stream_names,
            series.sampling_rate_hz,
            decomposition,
            reduction=features.stream_reduction,
            seed=window_seed,
            power_dbm=None if series.power_dbm is None else series.power_dbm[task.start : stop],
            power_names=series.power_names,
        )

    if features.n_jobs == 1:
        results = [job(task) for task in tasks]
    else:
        results = Parallel(n_jobs=features.n_jobs)(delayed(job)(task) for task in tasks)

    names = results[0][1]
    matrix = np.vstack([values for values, _ in results])
    owners = [series_list[t.series_index] for t in tasks]
    positions = np.array(
        [s.position if s.position is not None else (np.nan, np.nan) for s in owners],
        dtype=np.float64,
    ).reshape(-1, 2)
    return FeatureTable(
        features=matrix,
        feature_names=list(names),
        labels=np.array([s.label for s in owners], dtype=object).astype(str),
        recordings=np.array([s.recording for s in owners], dtype=object).astype(str),
        sessions=np.array([s.session for s in owners], dtype=object).astype(str),
        blocks=np.array(
            [f"{s.recording}#b{t.block}" for s, t in zip(owners, tasks, strict=True)],
            dtype=object,
        ).astype(str),
        window_start_s=np.array([t.start_s for t in tasks]),
        positions=positions,
        # The span the windows really cover once rounded to whole samples.
        window_s=max(
            round(features.window_s * s.sampling_rate_hz) / s.sampling_rate_hz for s in series_list
        ),
    )
