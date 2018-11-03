import argparse
import sys
import subprocess


def main(_=None):
    parser = argparse.ArgumentParser(description='Video Conversion via FFMPEG')
    parser.add_argument('-i', '--inp', required=True, help="input path")
    parser.add_argument('-o', '--out', required=True, help="output path")
    parser.add_argument('--width', default=-1)
    parser.add_argument('--height', default=-1)
    parser.add_argument('--fps', default=24)
    parser.add_argument('--pts', default=1.0)

    args = parser.parse_args()

    palette_path = "/tmp/palette.png"
    filters = f"setpts={args.pts}*PTS,fps={args.fps},scale={args.width}:{args.height}:flags=lanczos"

    subprocess.call(f'ffmpeg -v warning -i "{args.inp}" -vf "{filters},palettegen" -y "{palette_path}"', shell=True)
    subprocess.call(f'ffmpeg -i "{args.inp}" -i "{palette_path}" -lavfi "{filters} [x]; [x][1:v] '
                    f'paletteuse" -y "{args.out}"', shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
