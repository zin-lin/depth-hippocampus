"""
Universal spatial-proximity index over odometry positions — shared across all
model training pipelines in this project (VPC, JEPATR, CPC, ...), not tied to
any one architecture. Given a pool of recordings' (x, y) positions, finds
"near-odom" frames: physically close to a query frame but excluded from the
temporal window immediately around it, so a match is a genuine revisit (a
different pass through the same physical location) rather than the trivially
-similar next frame.

Motivated by the perceptual-aliasing finding in the stride=20/encode-window=20
evaluation: VPC-Depth collapsed specifically on the perimeter-loop recordings
(rec_17/18/19 — structurally self-similar loop segments) while VPC-RGB held
up, because pure depth geometry can't disambiguate one revisit of a loop from
another the way appearance can. An explicit "same place, different time"
training signal is a direct countermeasure. First consumer: CPC's near-odom
multi-positive InfoNCE (cpc/train.py).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FrameRef:
    recording: str
    frame_idx: int


class NearOdomIndex:
    """
    Args:
        recordings: dict mapping recording name -> (N, 2) xy positions (meters),
                    aligned 1:1 with that recording's frame files (same convention
                    as odometry.npz / odometry_i.npz elsewhere in this project).
    """

    def __init__(self, recordings: Dict[str, np.ndarray]) -> None:
        self.recording_names = list(recordings.keys())
        self._name_to_id = {name: i for i, name in enumerate(self.recording_names)}

        positions, rec_ids, frame_idxs = [], [], []
        for rec_id, name in enumerate(self.recording_names):
            pos = recordings[name]
            positions.append(pos)
            rec_ids.append(np.full(len(pos), rec_id))
            frame_idxs.append(np.arange(len(pos)))

        self.positions = np.concatenate(positions, axis=0).astype(np.float32)
        self.rec_ids = np.concatenate(rec_ids)
        self.frame_idxs = np.concatenate(frame_idxs)
        self.tree = cKDTree(self.positions)

        # per-recording offset into the concatenated arrays, for O(1) query-point lookup
        self._offsets = {}
        cursor = 0
        for name in self.recording_names:
            self._offsets[name] = cursor
            cursor += len(recordings[name])

    def query(
        self,
        recording: str,
        frame_idx: int,
        radius: float = 0.5,
        temporal_exclusion: int = 20,
    ) -> List[FrameRef]:
        """
        Frames within `radius` meters of (recording, frame_idx), excluding
        same-recording frames within +/- temporal_exclusion of frame_idx.
        Cross-recording matches are never temporally excluded (there's no
        shared clock between two separate recordings).
        """
        rec_id = self._name_to_id[recording]
        query_pos = self.positions[self._offsets[recording] + frame_idx]

        neighbor_idxs = self.tree.query_ball_point(query_pos, r=radius)

        matches = []
        for idx in neighbor_idxs:
            r_id = int(self.rec_ids[idx])
            f_idx = int(self.frame_idxs[idx])
            if r_id == rec_id and abs(f_idx - frame_idx) <= temporal_exclusion:
                continue
            matches.append(FrameRef(self.recording_names[r_id], f_idx))
        return matches

    def sample(
        self,
        recording: str,
        frame_idx: int,
        radius: float = 0.5,
        temporal_exclusion: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[FrameRef]:
        """Convenience: one random match, or None if no valid revisit exists."""
        matches = self.query(recording, frame_idx, radius, temporal_exclusion)
        if not matches:
            return None
        rng = rng or np.random.default_rng()
        return matches[int(rng.integers(len(matches)))]


def build_index_from_odom_files(recording_dirs: List, odom_filename: str = "odometry.npz") -> NearOdomIndex:
    """
    Convenience constructor matching this project's on-disk layout:
    data/rec_X/<odom_filename> with 'x'/'y' arrays, aligned to frame order.

    Args:
        recording_dirs: Path-like recording directories (e.g. data/rec_wh_01)
        odom_filename: 'odometry.npz' for depth (frame-count-aligned to depth64/),
                       'odometry_i.npz' for RGB (aligned to images64/) -- these are
                       NOT interchangeable, different frame counts per recording.
    """
    recordings = {}
    for rec_dir in recording_dirs:
        rec_dir = Path(rec_dir)
        odom = np.load(rec_dir / odom_filename)
        positions = np.stack([odom["x"], odom["y"]], axis=1).astype(np.float32)
        recordings[rec_dir.name] = positions
    return NearOdomIndex(recordings)


if __name__ == "__main__":
    print("Testing NearOdomIndex...")

    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    loop = (np.stack([np.cos(t), np.sin(t)], axis=1) * 5.0).astype(np.float32)  # 5m-radius closed loop
    straight = np.stack([np.linspace(0, 20, 100), np.zeros(100)], axis=1).astype(np.float32)  # no revisits

    index = NearOdomIndex({"loop_rec": loop, "straight_rec": straight})

    matches = index.query("loop_rec", 0, radius=0.5, temporal_exclusion=20)
    print(f"loop_rec frame 0: {len(matches)} near-odom matches (expect >0 -- loop closure)")
    for m in matches[:5]:
        print(f"  {m}")

    matches_straight = index.query("straight_rec", 50, radius=0.5, temporal_exclusion=20)
    print(f"straight_rec frame 50: {len(matches_straight)} near-odom matches (expect 0 -- no revisits)")

    sample = index.sample("loop_rec", 0, radius=0.5, temporal_exclusion=20)
    print(f"sample() for loop_rec frame 0: {sample}")

    assert len(matches) > 0 and len(matches_straight) == 0, "sanity check failed"
    print("\nAll checks passed.")
