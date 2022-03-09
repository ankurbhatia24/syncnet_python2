import os
import subprocess
import time
import concurrent
import concurrent.futures
import argparse
from tqdm import tqdm
from natsort import natsorted


#convert to 25fps
def change_frame_rate(i, source_video, target_video):
    
    
    source_video_basename = os.path.basename(source_video)
    timings = source_video_basename.split('_')[-1][:-4].split('-')
    try:
        st = float(timings[0])
        et = float(timings[1])
    except Exception as e:
        print("#"*10, timings, "#"*10, source_video_basename)
        import pdb; pdb.set_trace();
        raise e
    print(source_video, '/data3/soma/avspeech/temp_data_9k_avi/' + source_video_basename, st, et)
    source_video_cut = cut_video(source_video, '/data3/soma/avspeech/temp_data_9k_avi/' + source_video_basename, st, et)
    # import pdb;pdb.set_trace();
    if os.path.exists(target_video):
        return target_video
    s1 = time.time()
    cmd  = f"ffmpeg -y -i {source_video_cut} -qscale:v 2 -async 1 -r 25 {target_video} -y"
    # cmd = f"ffmpeg -hide_banner -loglevel error -i {source_video_cut} -filter:v fps {target_video} -y" #By default ffmpeg converts to 25fps
    #PAL Framerate
    subprocess.call(cmd, shell=True, stdout=None)
    return target_video
    # proc = subprocess.Popen(cmd, shell=True)
    # returncode = proc.wait()
    # e1 = time.time()
    # print(i, e1 - s1, source_video_cut)
    # if returncode != 0:
    #     raise Exception('Error finetuning the vocoder!')
    # else:
    #     return target_video

def cut_video(input, output, st, et):
    command = f"ffmpeg -ss {st} -i {input} -to {et} -c copy {output} -y"
    print(command)
    proc = subprocess.Popen(command, shell=True)
    returncode = proc.wait()
    if returncode != 0:
        raise Exception('Error finetuning the vocoder!')
    else:
        return output

def resample_audio(i, source_video, target_audio):
    os.makedirs(os.path.dirname(target_audio))
    if os.path.exists(target_audio):
        return target_audio
    s1 = time.time()
    command = ("ffmpeg -loglevel error -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (source_video, target_audio))
    #PAL Framerate
    proc = subprocess.Popen(command, shell=True)
    returncode = proc.wait()
    e1 = time.time()
    print(i, e1 - s1, source_video)
    if returncode != 0:
        raise Exception('Error finetuning the vocoder!')
    else:
        return target_audio


# def 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Before Steps Syncnet");
    parser.add_argument('--video_files_text_file', type=str, help='', default=None); #if file names in the text files
    parser.add_argument('--error_txt_file', type=str, help='');
    parser.add_argument('--videofiledirectory', type=str, default=None) #If direclty take the files from the directory
    parser.add_argument('--save_directory', type=str, default=None) #save the result of 25fps in this directory
    parser.add_argument('--convert_fps', type=bool, default=True)
    parser.add_argument('--convert_16k', type=bool, default=False)
    parser.add_argument('--reference', type=str, default="demo", help='');
    parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')

    opt = parser.parse_args();

    #read text file
    # with open(opt.video_files_text_file, 'r') as infile:
    #     list_video_files = infile.read().split('\n')
    
    list_video_files = natsorted(os.listdir(opt.videofiledirectory))[:]

    # import pdb; pdb.set_trace();
    if opt.convert_fps:
        save_files = []
        for file in tqdm(list_video_files):
            current_file = os.path.basename(file)
            save_files.append(os.path.join(opt.save_directory, current_file))

    if opt.convert_16k:
        save_files = []
        for file_ind in tqdm(range(len(list_video_files))):
            save_files.append(os.path.join(opt.tmp_dir, opt.reference + '_' + str(file_ind), 'audio.wav'))

    # import pdb; pdb.set_trace();
    # exit(0);
    #Start conversion
    s1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = {
                executor.submit(
                    change_frame_rate,
                    i,
                    os.path.join(opt.videofiledirectory, list_video_files[i]),
                    save_files[i][:-4] + ".avi"
                ) : i
                for i in range(len(list_video_files))
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    assert res
                    del res
                except Exception as e:
                    with open(opt.error_txt_file, 'a') as outfile:
                        outfile.write(str(e) + '\n')
                    print('Error in do_source_clustering!', e)
    e1 = time.time()
    print(f'Time taken for computation: ', e1 - s1)