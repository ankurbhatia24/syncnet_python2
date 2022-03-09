import time, pdb, argparse, subprocess, os
from tqdm import tqdm
from SyncNetInstance_calc_scores import *
import shutil
import time
from natsort import natsorted

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

def cut_to_5min(source_video, target_video):
    cmd = f"ffmpeg -ss 00:00:00 -i {source_video} -t 00:05:00 -c copy {target_video}"
    proc = subprocess.Popen(cmd, shell = True)
    returncode = proc.wait()
    if returncode != 0:
        raise Exception("Error cutting video")
    else:
        return target_video


def apply_offset_to_video_fps(source_video_path : str, target_video_path : str, offset : int):
    #Since the video is 25 fps => 0.04 sec for a frame to come
    #for a single frame we have to apply an offset of 40ms (+/-)
    cmd = f"ffmpeg -hide_banner -loglevel error -i {source_video_path} -itsoffset {-1*offset*40}ms -i {source_video_path} -map 1:v -map 0:a -c copy {target_video_path} -y"
    # ffmpeg -i out_25fps_16k.mp4 -itsoffset -80ms -i out_25fps_16k.mp4 -map 1:v -map 0:a -c copy -q out_25fps_16k_00offset.mp4 
    print("Offset command: ", cmd)
    proc = subprocess.Popen(cmd, shell=True)
    returncode = proc.wait()
    if returncode != 0:
        raise Exception('Error finetuning the vocoder!')
    else:
        return target_video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Before Steps Syncnet");
    parser.add_argument('--videofiledirectory', type=str, default="", help='');
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--convert', type = bool, default=False)
    parser.add_argument('--cut', type = bool, default = False)
    parser.add_argument('--metadata_filepath', type=str)
    parser.add_argument('--apply_syncnet', type=bool, default=False, help='');
    parser.add_argument('--apply_offset', type=bool, default=False)
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
    parser.add_argument('--batch_size', type=int, default='20', help='');
    parser.add_argument('--vshift', type=int, default='15', help='');
    parser.add_argument('--reference', type=str, default="demo", help='');
    parser.add_argument('--syncnet_result_file_name', type = str)
    parser.add_argument('--save_offset_corrected_videos_path', type=str)
    parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')

    
    # parser.add_argument('--savefiledirectory', type=str, default="", help='');
    opt = parser.parse_args();
    base_path = opt.videofiledirectory
    tmp_path = "/data3/suvrat/temp"
    os.makedirs(tmp_path, exist_ok = True)
    # os.makedirs(opt.save_directory, exist_ok=True)

    if opt.apply_syncnet:
        s = SyncNetInstance();
        s.loadParameters(opt.initial_model);
        print("Model %s loaded."%opt.initial_model);
        #write file and its offsets to a file
        file = open(opt.syncnet_result_file_name, 'a')


    all_files = natsorted(os.listdir(opt.videofiledirectory))[20000:30000]
    for i in tqdm(all_files):
        try:
            if opt.convert:
                print(opt.convert)
                print("hejhd")
                # current_video_path = os.path.join(base_path, i)
                # print(i)
                # fps_video_path = os.path.join(tmp_path, i[:-4] + '_25fps.mp4')
                # fps_video_path = change_frame_rate(current_video_path, fps_video_path)
                # fps_16k_video_path = normalize_sampling_rate_aac_mono(fps_video_path, fps_video_path[:-4] + '_16k.mp4')
                # os.remove(fps_video_path)
            if opt.cut:
                print("ndhdgbhjnd")
                # current_video_path = os.path.join(base_path, i)
                # print(i)
                # target_video_path = os.path.join(opt.save_directory, i[:-4] + "_5min.mp4")
                # cut_to_5min(current_video_path, target_video_path)
            if opt.apply_syncnet:
                current_video_path = os.path.join(base_path, i)
                print("Running syncnet on: ", current_video_path)
                start_time = time.time()
                offset, confidence, min_distance = s.evaluate(opt, videofile=current_video_path)
                dur = time.time() - start_time
                print("--- %s seconds ---" % (dur))
                print("Offset: ", offset, "Confidence: ", confidence, "min_distance: ", min_distance)
                # os.makedirs(opt.save_offset_corrected_videos_path, exist_ok = True)
                # save_video_path = os.path.join(opt.save_offset_corrected_videos_path, i)
                file.write(os.path.join(opt.save_offset_corrected_videos_path, i) + '\t' + str(offset) + '\t' + str(confidence) + '\t' + str(min_distance) + '\t' + str(dur) + '\n')
            
        except Exception as e:
            with open(opt.metadata_filepath, 'a') as outfile:
                outfile.write(i + " : " + str(e) + '\n')
            pass

    if opt.apply_offset:
        with open(opt.syncnet_result_file_name, 'r') as infile:
            lines = infile.readlines()
            for i, line in enumerate(lines):
                print(i, line)
                line_ = line.split('\t')
                videofile_basename = os.path.basename(line_[0])
                # import pdb; pdb.set_trace();
                save_dir = os.path.dirname(line_[0])
                offset = int(line_[1])
                current_video_path = os.path.join(opt.videofiledirectory, videofile_basename)
                save_video_path = line_[0]
                if offset != 0:
                    apply_offset_to_video_fps(current_video_path, save_video_path, int(offset))
                else:
                    shutil.copyfile(current_video_path, save_video_path)

