"""Detector-network presets, built on gwfast's detector geometry + shipped PSDs.

The network is a *configurable* input to the whole analysis (per CLAUDE.md): a
preset is just a list of ``(detector_key, psd_relative_path)`` pairs, where
``detector_key`` indexes ``gwfast.gwfastGlobals.detectors`` (geometry) and the
PSD path is relative to the gwfast PSD directory.  ``build_network`` turns a
preset into a ``gwfast.network.DetNet`` of ``gwfast.signal.GWSignal`` objects.
"""

from __future__ import annotations

import os
from typing import Iterable

from . import compat  # noqa: F401

import gwfast.gwfastGlobals as gg
from gwfast.signal import GWSignal
from gwfast.network import DetNet


def psd(*parts: str) -> str:
    """Absolute path to a PSD/ASD file shipped with gwfast."""
    return os.path.join(gg.detPath, *parts)


# Each preset: list of (detector key in gg.detectors, ASD file path).
# All shipped curves here are amplitude spectral densities (is_ASD=True).
NETWORKS: dict[str, list[tuple[str, str]]] = {
    # Single LIGO at design sensitivity -- simplest case, isolates the
    # intrinsic-parameter structure from network/sky effects.
    "L1_design": [
        ("L1", psd("unofficial_curves_all_dets", "aligo_design.txt")),
    ],
    # Current-era HLV network at O4 sensitivity.
    "HLV_O4": [
        ("H1", psd("observing_scenarios_paper", "aligo_O4high.txt")),
        ("L1", psd("observing_scenarios_paper", "aligo_O4high.txt")),
        ("Virgo", psd("observing_scenarios_paper", "avirgo_O4high_NEW.txt")),
    ],
    # A+ / O5-era design network.
    "HLV_design": [
        ("H1", psd("observing_scenarios_paper", "AplusDesign.txt")),
        ("L1", psd("observing_scenarios_paper", "AplusDesign.txt")),
        ("Virgo", psd("observing_scenarios_paper", "avirgo_O5high_NEW.txt")),
    ],
    # Einstein Telescope (triangular) alone.
    "ET": [
        ("ETS", psd("ET_designs_comparison_paper", "HFLF_cryo", "ET-0000A-18.txt"))
        if os.path.exists(
            psd("ET_designs_comparison_paper", "HFLF_cryo", "ET-0000A-18.txt")
        )
        else ("ETS", psd("ET-0000A-18.txt")),
    ],
    # 3G network: ET + two Cosmic Explorers.
    "ET_CE": [
        ("ETS", psd("ET-0000A-18.txt")),
        ("CE1Id", psd("ce_strain", "cosmic_explorer.txt")),
        ("CE2NM", psd("ce_strain", "cosmic_explorer_20km.txt")),
    ],
}


def build_signal(
    det_key: str,
    psd_path: str,
    wf_model,
    *,
    fmin: float = 10.0,
    fmax: float | None = 2048.0,
    use_earth_motion: bool = False,
    verbose: bool = False,
) -> GWSignal:
    """Construct a single-detector GWSignal from a gwfast detector key."""
    det = gg.detectors[det_key]
    return GWSignal(
        wf_model,
        psd_path=psd_path,
        detector_shape=det["shape"],
        det_lat=det["lat"],
        det_long=det["long"],
        det_xax=det["xax"],
        fmin=fmin,
        fmax=fmax,
        useEarthMotion=use_earth_motion,
        verbose=verbose,
        is_ASD=True,
    )


def build_network(
    name: str,
    wf_model_factory,
    *,
    fmin: float = 10.0,
    fmax: float | None = 2048.0,
    use_earth_motion: bool = False,
    verbose: bool = False,
) -> DetNet:
    """Build a ``DetNet`` for a named preset.

    ``wf_model_factory`` is a zero-argument callable returning a *fresh* gwfast
    waveform object (one per detector, since GWSignal stores per-detector state
    on the model).
    """
    if name not in NETWORKS:
        raise KeyError(
            f"unknown network {name!r}; available: {sorted(NETWORKS)}"
        )
    signals = {}
    for det_key, psd_path in NETWORKS[name]:
        signals[det_key] = build_signal(
            det_key,
            psd_path,
            wf_model_factory(),
            fmin=fmin,
            fmax=fmax,
            use_earth_motion=use_earth_motion,
            verbose=verbose,
        )
    return DetNet(signals, verbose=verbose)


def available_networks() -> Iterable[str]:
    return sorted(NETWORKS)
