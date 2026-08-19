import os
import subprocess
from typing import Optional


def create_video(
    input_pattern: str,
    output_path: str,
    framerate: int = 30,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    duration: Optional[int] = None,
    scale: Optional[str] = None,
    overwrite: bool = True,
    suppress_output: bool = True,
    cwd: Optional[str] = None,
):
    """
    Create a video (or .gif) from an image sequence via ffmpeg.
    Glob patterns ("*", "?") in input_pattern are auto-detected; duration is in
    seconds; scale is an ffmpeg scale filter (e.g. "640:trunc(ow/a/2)*2").
    """
    # Input args shared by both GIF passes
    input_args = []
    if overwrite:
        input_args.append("-y")
    if "*" in input_pattern or "?" in input_pattern:
        input_args.extend(["-pattern_type", "glob"])
    input_args.extend(["-framerate", str(framerate), "-i", input_pattern])
    if duration is not None:
        input_args.extend(["-t", str(duration)])

    run_kwargs = {"cwd": cwd}
    if suppress_output:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL

    if output_path.endswith(".gif"):
        # Two-pass GIF: pass 1 builds a 256-color palette, pass 2 encodes with it
        scale_filter = f"scale={scale}," if scale is not None else ""
        palette_filter = f"{scale_filter}palettegen=stats_mode=diff"
        encode_filter = f"{scale_filter}paletteuse=dither=sierra2_4a"

        palette_path = output_path + ".palette.png"
        pass1 = ["ffmpeg"] + input_args + ["-vf", palette_filter, palette_path]
        pass2 = (
            ["ffmpeg"]
            + input_args
            + ["-i", palette_path, "-lavfi", encode_filter, output_path]
        )

        subprocess.run(pass1, **run_kwargs)
        subprocess.run(pass2, **run_kwargs)

        if os.path.exists(palette_path):
            os.remove(palette_path)
    else:
        cmd = ["ffmpeg"] + input_args + ["-c:v", codec, "-pix_fmt", pix_fmt]
        if scale is not None:
            cmd.extend(["-vf", f"scale={scale}"])
        cmd.append(output_path)
        subprocess.run(cmd, **run_kwargs)
