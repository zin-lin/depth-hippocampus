#!/usr/bin/env python3
"""Playback recorded depth frames at 5.5Hz as a slideshow."""

import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Depth frame playback")
    parser.add_argument('--path', type=str, default='./rec/depth')
    parser.add_argument('--fps', type=float, default=5.5)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.path}/*.npy"))
    print(f"Found {len(files)} frames, playing at {args.fps} Hz")

    fig, ax = plt.subplots()
    img = ax.imshow(np.load(files[0]), cmap='inferno')
    ax.set_axis_off()
    title = ax.set_title(f"Frame 0/{len(files)}")
    plt.colorbar(img, ax=ax)

    interval = 1.0 / args.fps
    for i, f in enumerate(files):
        frame = np.load(f)
        frame = np.clip(frame, 0, np.nanmax(frame[np.isfinite(frame)]))
        img.set_data(frame)
        img.set_clim(frame.min(), frame.max())
        title.set_text(f"Frame {i}/{len(files)}")
        plt.pause(interval)

    plt.show()

if __name__ == "__main__":
    main()
