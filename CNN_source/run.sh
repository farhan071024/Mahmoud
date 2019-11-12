#!/bin/bash
time python -u cnn_model.py --window_size 1 --height 120 --width 90 --batch_size 64 --num_epochs 20 --data_dir ~/tfrecords/pairs_recs/random_pick/ws1 --learning_rate 1e-5 --log_dir ~/log/w1-b64-ep20-depth-pairs | tee ~/log_run/w1-b64-ep20-depth-pairs.out
