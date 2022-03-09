#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

import shutil
from detectors import S3FD
from natsort import natsorted


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(opt,track,cropfile):
  # import pdb; pdb.set_trace();
  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
    
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    try: 
      vOut.write(cv2.resize(face,(224,224)))
    except:
      pdb.set_trace();

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def get_frame_resize_factor_for_video(frame):
  res_to_factor = {'1280_720' : 2, 
                    '1920_1080' : 2, 
                    '1440_1080' : 2, 
                    '854_480' : 2, 
                    '640_480' : 2, 
                    '1728_1080' : 2, 
                    '1270_720' : 2, 
                    '1906_1080' : 2, 
                    '1916_1080' : 2, 
                    '1172_720' : 2, 
                    '900_720' : 2, 
                    '1152_720' : 2,
                    '608_1080' : 2,
                    '854_478' : 2,
                    '1080_720' : 2,
                    '720_480' : 2,
                    '1278_720' : 2,
                    '960_720' : 2}
  image = cv2.imread(frame)
  height = image.shape[0]
  width = image.shape[1]
  if str(width) + '_' + str(height) in res_to_factor:
    return width, height, res_to_factor[str(width) + '_' + str(height)]
  else:
    return width, height, 1

video_to_speaking_face_coords = pickle.load(open("/data3/soma/avspeech/face_center_coordinate.pickle", 'rb'))

def get_face_center_coordinate(video_crop, height, width):
  face_center_coords = video_to_speaking_face_coords[video_crop].split('-')
  x = float(face_center_coords[0])
  y = float(face_center_coords[1])
  x = x*width
  y = y*height
  return (x, y)

def inference_video(opt):

  DET = S3FD(device='cuda')


  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = []

  width, height, resize_factor = get_frame_resize_factor_for_video(flist[0])  #assumed a fixed width and height for each frame
  target_shape = (width//resize_factor, height//resize_factor)
      
  for fidx, fname in enumerate(flist[:10]):

    start_time = time.time()
    
    #lower the res of frame
    image = cv2.imread(fname)
    # image = cv2.resize(image, target_shape, interpolation = cv2.INTER_AREA)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
    #rescale bboxes
    bboxes = bboxes*resize_factor
    dets.append([]);
    for bbox in bboxes:
      dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

    # import pdb; pdb.set_trace();
    elapsed_time = time.time() - start_time

    print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

  #Check which one is the active speaker: CSV(loaded in a dict > pickle) as the center coordinate of the speaking speaker
  
  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')
  savepath_doubt = os.path.join(opt.work_dir, opt.reference,'doubt_faces.pckl')

  # gt_x, gt_y = get_face_center_coordinate("--RIhXUy_3E_219.280000-225.840000", height, width)
  # yt_id = os.path.basename(opt.videofile)[:-4].split('_')
  # print("$$$$$$$ ", yt_id)
  # import pdb; pdb.set_trace();
  # tims = yt_id[-1].split('-')
  # st = float(tims[0])
  # et = float(tims[1])
  # yt_id = "_".join(yt_id[:-1])
  # video_id = yt_id + '_' + str(st) + '-' + str(et)
  # gt_x, gt_y = get_face_center_coordinate(video_id, height, width)
  # import pdb; pdb.set_trace();
  if len(dets[0]) > 1:
    single_dets = []
    # face_dict = {}
    # doubt_frames = []
    # for frame_face_detction in dets:
    #   found = False
    #   for i, face_detected in enumerate(frame_face_detction):
    #     x1, y1, x2, y2 = tuple(face_detected["bbox"])
    #     if gt_x >= x1 and gt_x <= x2 and gt_y >= y1 and gt_y <= y2:
    #       if str(i) in face_dict:
    #         face_dict[str(i)] += 1
    #       else:
    #         face_dict[str(i)] = 1
    #       found = True
    #   if not found:
    #     doubt_frames.append(frame_face_detction)
    # #Frame with max number of center coordinate lying within it
    # try: 
    #   max_key = int(max(face_dict, key=face_dict.get))
    #   single_dets = []
    #   for frame_face_detction in dets:
    #     single_dets.append([frame_face_detction[max_key]])
    # except:
    #   single_dets = []
  else:
    single_dets = dets

  
  # import pdb; pdb.set_trace();


  # with open(savepath, 'wb')

  with open(savepath, 'wb') as fil:
    pickle.dump(single_dets, fil)

  return single_dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_manager = VideoManager([os.path.join(opt.avi_dir,opt.reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(opt.work_dir,opt.reference,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

  return scene_list
    

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========
def main():
    if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
      rmtree(os.path.join(opt.work_dir,opt.reference))

    if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
      rmtree(os.path.join(opt.crop_dir,opt.reference))

    if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
      rmtree(os.path.join(opt.avi_dir,opt.reference))

    if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
      rmtree(os.path.join(opt.frames_dir,opt.reference))

    if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
      rmtree(os.path.join(opt.tmp_dir,opt.reference))

    # ========== MAKE NEW DIRECTORIES ==========

    os.makedirs(os.path.join(opt.work_dir,opt.reference))
    os.makedirs(os.path.join(opt.crop_dir,opt.reference))
    os.makedirs(os.path.join(opt.avi_dir,opt.reference))
    os.makedirs(os.path.join(opt.frames_dir,opt.reference))
    os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

    # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile,os.path.join(opt.avi_dir,opt.reference,'video.avi')))  #converting the video to 25 fps
    output = subprocess.call(command, shell=True, stdout=None)

    # pdb.set_trace();
    # shutil.copyfile(opt.videofile, os.path.join(opt.avi_dir,opt.reference,'video.avi'))

    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg'))) #extract frames
    output = subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav'))) 
    output = subprocess.call(command, shell=True, stdout=None)

    # ========== FACE DETECTION ==========

    faces = inference_video(opt)

    # ========== SCENE DETECTION ==========

    scene = scene_detect(opt)

    # ========== FACE TRACKING ==========

    alltracks = []
    vidtracks = []
    pdb.set_trace();s
    for shot in scene:
      if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
        print("track_shot")
        alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num]))

    # ========== FACE TRACK CROP ==========
    print("Face track crop")
    # pdb.set_trace();
    for ii, track in enumerate(alltracks):
      vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

    # ========== SAVE RESULTS ==========

    savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

    with open(savepath, 'wb') as fil:
      pickle.dump(vidtracks, fil)

    rmtree(os.path.join(opt.tmp_dir,opt.reference))


if __name__ == "__main__":
  # ========== ========== ========== ==========
  # # PARSE ARGS
  # ========== ========== ========== ==========
  print("hdjhdd")
  parser = argparse.ArgumentParser(description = "FaceTracker");
  parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry');
  parser.add_argument('--videofile',      type=str, default='',   help='Input video file');
  parser.add_argument('--videofiledirectory',type=str, default='',   help='Input video file');
  parser.add_argument('--reference',      type=str, default='',   help='Video reference');
  parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection');
  parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box');
  parser.add_argument('--min_track',      type=int, default=5,  help='Minimum facetrack duration');
  parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
  parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped');
  parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels');

  opt = parser.parse_args();

  setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
  setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
  setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
  setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
  setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))
  for file in natsorted(os.listdir(opt.videofiledirectory))[:]: 
    opt.reference = os.path.basename(file)
    opt.videofile = os.path.join(opt.videofiledirectory, "0GUxVrYkjxw_250.550300-260.360100.mp4")
    main()