#!/bin/bash

S1=$((1 + $RANDOM % 999999))
S2=$((1 + $RANDOM % 999999))
S3=$((1 + $RANDOM % 999999))
S4=$((1 + $RANDOM % 999999))
S5=$((1 + $RANDOM % 999999))
S6=$((1 + $RANDOM % 999999))
S7=$((1 + $RANDOM % 999999))
S8=$((1 + $RANDOM % 999999))
S9=$((1 + $RANDOM % 999999))
S10=$((1 + $RANDOM % 999999))


declare -a MODELS=( "microsoft/Phi-3.5-vision-instruct" 
					"llava-hf/llava-v1.6-vicuna-7b-hf" 
					"llava-hf/llava-v1.6-vicuna-13b-hf" 
					"meta-llama/Llama-3.2-11B-Vision-Instruct" )


#Qwen seems to be immune to the random seeds...
python run.py --model_path "Qwen/Qwen2-VL-7B-Instruct" --task 4 --seed $S1 #--type lv

for model in "${MODELS[@]}"
do
	python run.py --model_path $model --task 4 --seed $S1 #--type lv
	python run.py --model_path $model --task 4 --seed $S2 #--type lv
	python run.py --model_path $model --task 4 --seed $S3 #--type lv
	python run.py --model_path $model --task 4 --seed $S4 #--type lv
	python run.py --model_path $model --task 4 --seed $S5 #--type lv
	python run.py --model_path $model --task 4 --seed $S6 #--type lv
	python run.py --model_path $model --task 4 --seed $S7 #--type lv
	python run.py --model_path $model --task 4 --seed $S8 #--type lv
	python run.py --model_path $model --task 4 --seed $S9 #--type lv
	python run.py --model_path $model --task 4 --seed $S10 #--type lv
done
