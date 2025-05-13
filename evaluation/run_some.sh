#!/bin/bash

S1=8263
S2=10652
S3=29960
S4=19236
S5=4037
S6=21154
S7=13930
S8=18412
S9=26602
S10=5220



model="llava-hf/llava-v1.6-vicuna-13b-hf"

python run.py --model_path $model --task 5 --type lv --seed $S7
python run.py --model_path $model --task 5 --type lv --seed $S8
python run.py --model_path $model --task 5 --type lv --seed $S9
python run.py --model_path $model --task 5 --type lv --seed $S10

model="meta-llama/Llama-3.2-11B-Vision-Instruct"

python run.py --model_path $model --task 5 --type lv --seed $S1
python run.py --model_path $model --task 5 --type lv --seed $S2
python run.py --model_path $model --task 5 --type lv --seed $S3
python run.py --model_path $model --task 5 --type lv --seed $S4
python run.py --model_path $model --task 5 --type lv --seed $S5
python run.py --model_path $model --task 5 --type lv --seed $S6
python run.py --model_path $model --task 5 --type lv --seed $S7
python run.py --model_path $model --task 5 --type lv --seed $S8
python run.py --model_path $model --task 5 --type lv --seed $S9
python run.py --model_path $model --task 5 --type lv --seed $S10