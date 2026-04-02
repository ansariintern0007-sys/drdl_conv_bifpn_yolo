import subprocess

input_gif = "/home/aid-pc/Documents/stream2.gif"
output_wmv = "/home/aid-pc/Documents/output.wmv"

cmd = [
    "ffmpeg",
    "-i", "/home/aid-pc/Documents/stream2.gif",
    "-c:v", "wmv2",
    "-pix_fmt", "yuv420p",
    "-r", "25",
    output_wmv
]

subprocess.run(cmd)