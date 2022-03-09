#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess

# from SyncNetInstance import *
from SyncNetInstance_calc_scores import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

offset, confidence, min_distance = s.evaluate(opt, videofile=opt.videofile)
print("Offset: ", offset, "Confidence: ", confidence, "min_distance: ", min_distance)

