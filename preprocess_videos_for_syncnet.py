def normalize_sampling_rate_aac_mono(file_path, out_file_path):
    # Change to 22050 for all the videos
    cmd = f"ffmpeg -hide_banner -loglevel error -i {file_path} -ac 1 -ar 16000 -c:a aac -q:a 1.3 {out_file_path} -y"
    proc = subprocess.Popen(cmd, shell=True)
    returncode = proc.wait()
    if returncode != 0:
        raise Exception('Error finetuning the vocoder!')
    else:
        return out_file_path

def change_frame_rate(source_video, target_video):
    cmd = f"ffmpeg -hide_banner -loglevel error -i {source_video} -filter:v fps {target_video} -y" #By default ffmpeg converts to 25fps
    #PAL Framerate
    proc = subprocess.Popen(cmd, shell=True)
    returncode = proc.wait()
    if returncode != 0:
        raise Exception('Error finetuning the vocoder!')
    else:
        return target_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "SyncNet");
    parser.add_argument('--videofiledirectory', type=str, default="", help='');
    parser.add_argument('--save_dir', type=str, default="tmp", help='');
    opt = parser.parse_args();

    