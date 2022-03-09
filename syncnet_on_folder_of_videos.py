#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
from tqdm import tqdm
# from SyncNetInstance import *
from SyncNetInstance_calc_scores import *

# offset, confidence, min_distance = s.evaluate(opt, videofile=os.path.join(opt.videofiledirectory, i))
# print("Offset: ", offset, "Confidence: ", confidence, "min_distance: ", min_distance)


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

    parser = argparse.ArgumentParser(description = "SyncNet");
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
    parser.add_argument('--batch_size', type=int, default='20', help='');
    parser.add_argument('--vshift', type=int, default='15', help='');
    # parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
    parser.add_argument('--videofiledirectory', type=str, default="", help='');
    parser.add_argument('--save_offset_corrected_videos_path', type=str)
    parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
    parser.add_argument('--reference', type=str, default="demo", help='');
    parser.add_argument('--apply_syncnet', type=bool, default=False, help='');
    opt = parser.parse_args();

    s = SyncNetInstance();

    s.loadParameters(opt.initial_model);
    print("Model %s loaded."%opt.initial_model);

    #write file and its offsets to a file
    file = open('syncnet_result.txt', 'a')

    list_videos_done = []
    for i in os.listdir(opt.save_offset_corrected_videos_path):
        list_videos_done.append(i.split('_25fps_16k')[0] + i.split('_25fps_16k')[1])

    base_path = opt.videofiledirectory
    save_path = "/data2/ankur/temp"
    os.makedirs(save_path, exist_ok= True)
    for i in tqdm({'WDA_MarkWarner_000.mp4', 'WDA_KirstenGillibrand_000.mp4', 'RD_Radio19_000.mp4', 'WDA_JonTester0_000.mp4', 'RD_Radio20_000.mp4', 'RD_Radio27_000.mp4', 'WRA_CarlyFiorina_000.mp4', 'WRA_LamarAlexander_000.mp4', 'RD_Radio10_000.mp4', 'WDA_XavierBecerra_001.mp4', 'WRA_MichaelSteele_001.mp4', 'WRA_RogerWicker1_001.mp4', 'WRA_JohnThune_000.mp4', 'RD_Radio17_000.mp4', 'WRA_ChuckGrassley_000.mp4', 'WDA_JohnYarmuth1_000.mp4', 'WRA_JohnnyIsakson_001.mp4', 'WRA_DebFischer0_000.mp4', 'WRA_RichardShelby_000.mp4', 'RD_Radio12_000.mp4', 'RD_Radio18_000.mp4', 'WDA_BobCasey1_000.mp4', 'WRA_ShelleyMooreCapito0_000.mp4', 'WRA_KristiNoem2_000.mp4', 'WDA_MazieHirono1_000.mp4', 'RD_Radio23_000.mp4', 'WDA_EliotEngel_000.mp4', 'WRA_AdamKinzinger0_000.mp4', 'WDA_PattyMurray0_000.mp4', 'WRA_JimInhofe_000.mp4', 'WDA_TammyBaldwin0_000.mp4', 'WRA_RandPaul0_000.mp4', 'WRA_TomPrice_000.mp4', 'WRA_KristiNoem2_001.mp4', 'WDA_MaggieHassan_000.mp4', 'WRA_JohnHoeven_000.mp4', 'WRA_ShelleyMooreCapito1_000.mp4', 'WRA_MikeJohanns1_000.mp4', 'WRA_SteveDaines0_000.mp4', 'WDA_JoeNeguse_000.mp4', 'WRA_MichaelSteele_000.mp4', 'RD_Radio16_000.mp4', 'WRA_PeterKing_000.mp4', 'WDA_XavierBecerra_002.mp4'}):
        if i not in list_videos_done and i != 'WRA_PeterKing_000.mp4' and i!= "WDA_MaggieHassan_000.mp4" and i != "WDA_MazieHirono1_000.mp4":
            current_video_path = os.path.join(base_path, i)
            # Change the Framerate
            target_video_path = os.path.join(save_path, i[:-4] + '_25fps.mp4')
            if not os.path.exists(target_video_path):
                target_video_path = change_frame_rate(current_video_path, target_video_path)
            else:
                print("25fps video path exists")

            #Change the audio to 16k and monos
            if not os.path.exists(target_video_path[:-4] + '_16k.mp4'):
                target_video_path = normalize_sampling_rate_aac_mono(target_video_path, target_video_path[:-4] + '_16k.mp4')
            else:
                target_video_path = target_video_path[:-4] + '_16k.mp4'
                print("25fps_16k video exists")

            # if opt.apply_syncnet and not os.path.exists(os.path.join(opt.save_offset_corrected_videos_path, os.path.basename(target_video_path))) :
            #     print("Running syncnet on: ", target_video_path)
            #     offset, confidence, min_distance = s.evaluate(opt, videofile=target_video_path)
            #     print("Offset: ", offset, "Confidence: ", confidence, "min_distance: ", min_distance)
            #     os.makedirs(opt.save_offset_corrected_videos_path, exist_ok = True)
            #     save_video_path = os.path.join(opt.save_offset_corrected_videos_path, os.path.basename(target_video_path))
            #     file.write(i + '\t' + str(offset) + '\t' + str(confidence) + '\t' + str(min_distance) + '\n')
            #     if offset != 0:
            #         apply_offset_to_video_fps(target_video_path, save_video_path, int(offset))
            # else:
            #     print("Not Applying Syncnet")
        else:
            print("Video Already Done")
    print("Success = True")
    file.close()