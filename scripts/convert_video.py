import argparse
import sys
import subprocess

# see https://medium.com/abraia/basic-video-editing-for-social-media-with-ffmpeg-commands-1e873801659

def main(_=None):
    parser = argparse.ArgumentParser(description='Video Conversion via FFMPEG')
    parser.add_argument('-i', '--inp', required=True, help="input path")
    parser.add_argument('-o', '--out', required=True, help="output path")
    parser.add_argument('--width', default=-1)
    parser.add_argument('--height', default=-1)
    parser.add_argument('--fps', default=24)
    parser.add_argument('--pts', default=1.0)
    parser.add_argument('--ss', default="")
    parser.add_argument('-t', default="")

    args = parser.parse_args()

    palette_path = "/tmp/palette.png"
    filters = f"setpts={args.pts}*PTS,fps={args.fps},scale={args.width}:{args.height}:flags=lanczos"
    # define start and time args if a value has been passed
    cropping = f"-ss {args.ss}" if args.ss else "" + f"-t {args.t}" if args.t else ""

    subprocess.call(f'ffmpeg -v warning -i "{args.inp}" -vf "{filters},palettegen" -y "{palette_path}"', shell=True)
    subprocess.call(f'ffmpeg -i "{args.inp}" -i "{palette_path}" -movflags +faststart -lavfi "{filters} [x]; [x][1:v] '
                    f'paletteuse" {cropping} -y "{args.out}"', shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
