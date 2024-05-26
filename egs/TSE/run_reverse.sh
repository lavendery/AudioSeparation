# Author: Jinchuan Tian, jinchuat@andrew.cmu.edu

# A demo recipe for TSE

. ./path.sh

# pip3 install fairseq==0.12.2 einops==0.6.0 sentencepiece encodec

stage=5
stop_stage=5
ngpu=8
train_set="train"
valid_set="val"
test_set="test"
TASK='Audioset_ex_large510_reverse'


# training config
seed=999
debug=false
batch_scale=4800
learning_rate=0.0005
port=12345
train_opts=
inference_opts=
tag=
inference_tag=default
resume=
data_tag=

if [ ! -d "utils" ]; then
  ln -s ../../tools/kaldi/utils ./
fi
if [ ! -d "data_scripts" ]; then
  ln -s ../../tools/data_scripts ./
fi
if [ ! -d "tokenizer" ]; then
  ln -s ../../tools/tokenizer ./
fi

. utils/parse_options.sh

if [ ! -z $resume ]; then
    train_opts="--resume $resume"
    inference_opts="--resume $resume"
fi

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    # train_opts="$train_opts --print_freq 1 --minibatch_debug 100"
    # train_opts="$train_opts --print_freq 1 --minibatch_debug 100 --n_epoch 1 --save_interval 99"
else
    export HOST_GPU_NUM=8
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
fi
# split_scp=
# for n in `seq 1 $ngpu`; do
#     split_scp="$split_scp /home/v-dongyang/exp_data/SE_new/${ngpu}splits/librilight_se.${n}"
# done
# utils/split_scp.pl /home/v-dongyang/exp_data/librilight_se $split_scp

### stage 1-3: data preparation ###

# # Prepare data following Espnet and split
# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     echo "Prepare Librilight dataset"
#     # bash local/data.sh || exit 1;

#     # valid set is selected from train set without overlap: 2 utts for each speaker
#     utils/subset_data_dir.sh --per-spk data/train-960 2 data/train_holdout
#     cp data/train-960/{wav.scp,spk2utt,utt2spk} data/train
#     filter_scp.pl --exclude -f 1 data/train_holdout/text data/train-960/text \
#       > data/train/text
#     bash utils/fix_data_dir.sh data/train/
# fi

# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     echo "split the data for $ngpu GPUs"

#     for part in $valid_set $train_set; do
#       mkdir -p data/${TASK}/${part}/${ngpu}splits
#       # extra shuf to ensure balance across GPUs
#       # So the generated data cannot be reproduced due to the shuffle randomness
#       cat data/${TASK}/${part}/tar.scp | shuf > data/${TASK}/${part}/tar.scp.shuf

#       split_scp=
#       for n in `seq 1 $ngpu`; do
#           split_scp="$split_scp data/${TASK}/${part}/${ngpu}splits/tar.${n}.scp"
#       done
#       utils/split_scp.pl data/${TASK}/${part}/tar.scp.shuf $split_scp

#     done
# fi

# ncpu=32
# scpu=25
# # TTS requires 3 data keys: phone_seq, prompt_seq, audio_seq
# # stage 2-3 process audio_seq and prompt_seq respectively
# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "Prepare text and audio sequence"
#     for part in $valid_set $train_set; do
#       utils/run.pl JOB=$scpu:$ncpu data/${TASK}/${part}/${ngpu}splits/log/mix_codec_dump.JOB.log \
#         python3 data_scripts/offline_tokenization.py \
#           --tar-file data/${TASK}/${part}/${ngpu}splits/tar.JOB.scp \
#           --tar-key-word mix_speech \
#           --tar-info data/${TASK}/${part}/${ngpu}splits/tar_info_mix_speech.JOB.scp \
#           --output-file data/${TASK}/${part}/${ngpu}splits/mix_codec.JOB.pt \
#           --tokenizer audio --rank JOB \
        
#       utils/run.pl JOB=$scpu:$ncpu data/${TASK}/${part}/${ngpu}splits/log/target_codec_dump.JOB.log \
#         python3 data_scripts/offline_tokenization.py \
#           --tar-file data/${TASK}/${part}/${ngpu}splits/tar.JOB.scp \
#           --tar-key-word target_speech \
#           --tar-info data/${TASK}/${part}/${ngpu}splits/tar_info_target_speech.JOB.scp \
#           --output-file data/${TASK}/${part}/${ngpu}splits/target_codec.JOB.pt \
#           --tokenizer audio --rank JOB \
      
#       utils/run.pl JOB=$scpu:$ncpu data/${TASK}/${part}/${ngpu}splits/log/prompt_codec_dump.JOB.log \
#         python3 data_scripts/offline_tokenization.py \
#           --tar-file data/${TASK}/${part}/${ngpu}splits/tar.JOB.scp \
#           --tar-key-word prompt_speech \
#           --tar-info data/${TASK}/${part}/${ngpu}splits/tar_info_prompt_speech.JOB.scp \
#           --output-file data/${TASK}/${part}/${ngpu}splits/prompt_codec.JOB.pt \
#           --tokenizer audio --rank JOB \
    
#     done
# fi

### stage 1-3: data preparation ###

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare TSE dataset"
    # this part aims to get the information about the dataset. 
    # prepare mix.scp, target.scp and prompt.scp
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"

    for part in $test_set $valid_set $train_set; do
      mkdir -p data/${TASK}/${part}/${ngpu}splits
      # extra shuf to ensure balance across GPUs
      # So the generated data cannot be reproduced due to the shuffle randomness
      cat data/${TASK}/${part}/mix_wav.scp | shuf >  data/${TASK}/${part}/mix_wav.scp.shuf
      split_scp=
      for n in `seq 1 $ngpu`; do
          split_scp="$split_scp data/${TASK}/${part}/${ngpu}splits/mix_wav.${n}.scp"
      done
      utils/split_scp.pl data/${TASK}/${part}/mix_wav.scp.shuf $split_scp


    done
fi

# Plain TTS requires 2 data keys: phone_seq, audio_seq
# stage 2-3 process audio_seq and phone_seq respectively
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare audio sequence"
    for part in $test_set $valid_set $train_set; do
    # for part in $valid_set; do
      echo "prepare $part ... "

      # split target_wav.scp based on mix_wav.scp
      utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits/log/filter_target_wav.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${TASK}/${part}/${ngpu}splits/mix_wav.JOB.scp data/${TASK}/${part}/target_wav.scp \
          data/${TASK}/${part}/${ngpu}splits/target_wav.JOB.scp || exit 1;

      # split prompt_wav.scp based on mix_wav.scp
      utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits/log/filter_prompt_wav.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${TASK}/${part}/${ngpu}splits/mix_wav.JOB.scp data/${TASK}/${part}/prompt_wav.scp \
          data/${TASK}/${part}/${ngpu}splits/prompt_wav.JOB.scp || exit 1;

      # mix Audio
      utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits/log/mix_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${TASK}/${part}/${ngpu}splits/mix_wav.JOB.scp \
          --output-file data/${TASK}/${part}/${ngpu}splits/mix_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;

      # target Audio
      utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits/log/target_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${TASK}/${part}/${ngpu}splits/target_wav.JOB.scp \
          --output-file data/${TASK}/${part}/${ngpu}splits/target_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;

      # prompt Audio
      utils/run.pl JOB=1:$ngpu data/${TASK}/${part}/${ngpu}splits/log/prompt_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${TASK}/${part}/${ngpu}splits/prompt_wav.JOB.scp \
          --output-file data/${TASK}/${part}/${ngpu}splits/prompt_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;
      
    done
fi

# TASK='Spex_large'
# ngpu=32
# TASK='Audioset_ex'
# TASK='Audioset_ex_large515'
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "create data json"
    for part in $train_set $valid_set $test_set; do
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task Spex \
         --out-json   data/${TASK}/${part}/${ngpu}splits/data.${n}.json \
         --noise_seq  data/${TASK}/${part}/${ngpu}splits/mix_codec.$[$n+1].pt \
         --prompt_seq  data/${TASK}/${part}/${ngpu}splits/prompt_codec.$[$n+1].pt \
         --audio_seq  data/${TASK}/${part}/${ngpu}splits/target_codec.$[$n+1].pt \
         & 
      done; wait
    done
fi

### Stage 4: Training ###
# train_data_jsons="data/${TASK}/${train_set}/${ngpu}splits/data.ALL.json"
# valid_data_jsons="data/${TASK}/${valid_set}/${ngpu}splits/data.ALL.json"
#TSE train
# train_data_jsons="data/Audioset_ex_large/train/32splits/data.ALL.json"
# train_data_jsons="data/Spex_large/train/32splits/data.ALL.json"
train_data_jsons="data/Audioset_ex_largeAB_d5/train/32splits/data.ALL.json"
train_data_jsons="$train_data_jsons data/Audioset_ex_largeAB_d10/train/32splits/data.ALL.json"
train_data_jsons="$train_data_jsons data/Audioset_ex_large510/train/32splits/data.ALL.json"
train_data_jsons="$train_data_jsons data/ESC-50/train/8splits/data.ALL.json"
train_data_jsons="$train_data_jsons data/Spex_large/train/32splits/data.ALL.json"
#TSE val
# valid_data_jsons="data/Audioset_ex_large/val/32splits/data.ALL.json"
# valid_data_jsons="data/Spex_large/val/32splits/data.ALL.json"
# valid_data_jsons="data/Audioset_ex_largeAB_d5/val/32splits/data.ALL.json"
# valid_data_jsons="$valid_data_jsons data/Audioset_ex_largeAB_d10/val/32splits/data.ALL.json"
valid_data_jsons="data/Audioset_ex_large510/val/32splits/data.ALL.json"
valid_data_jsons="$valid_data_jsons data/ESC-50/val/8splits/data.ALL.json"
valid_data_jsons="$valid_data_jsons data/Spex_large/val/32splits/data.ALL.json"

# tag="largesound"
# tag="speech"
# tag="largesound_speech"
# tag="largesound_ABd5_ABd10"
# tag="largesound_ABd5_ABd10_speech"
# tag="largesound_ABd5_510_ESC-50_speech"
# tag="largesound_ABd5_510_ESC-50"
# tag="largesound_ABd5_ABd10_510_ESC-50"
# tag="largesound_ABd5_ABd10_510_ESC-50_speech"
tag="largesound_ABd5_ABd10_510_ESC-50_pretrain"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    mkdir -p data/exp
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 5: training..."
    NCCL_DEBUG=TRACE torchrun \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ../../train.py \
        --resume data/exp/pretrain_audioset_libri/ep1-iter90000.checkpoint \
        --exp_dir data/exp/${tag} \
        --seed $seed \
        --cudnn_deterministic \
        --train_data_jsons $train_data_jsons \
        --valid_data_jsons $valid_data_jsons \
        --batch_scale $batch_scale \
        --learning_rate $learning_rate \
        --non-acoustic-repeat 3 \
        --audio-tokenizer "soundstream" \
        --audio-prompt-tokenizer "audio_prompt" \
        --n_layer 12 \
        --n_head 8 \
        --n_embd 1536 \
        $train_opts
fi

# TSE inference
vc_test_sets="tse_test"
tag="largesound_ABd5_ABd10_510_ESC-50"
inference_tag="ESC50"
inference_dir=data/exp/${tag}/inference_${inference_tag}
ngpu=1
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: TTS  inference ..."
    mkdir -p ${inference_dir}
    for part in $vc_test_sets; do
        mkdir -p ${inference_dir}/${part}
        echo "inference on set: ${part}"
        # data_json="data/Audioset_ex_large515/test/8splits/data.0.json" # your val set .json file
        # data_json="data/Audioset_ex_large05/test/8splits/data.0.json"
        # data_json="data/Spex_large/test/8splits/data_voicebank.0.json"
        data_json="data/ESC-50/test/8splits/data.0.json"
        # data_json="data/esc-50-simulation/test/8splits/data.0.json"
        # data_json="data/esc-50-simulation_3s/test/8splits/data.0.json"

        utils/run.pl --max-jobs-run 8 JOB=0:$[${ngpu}-1] \
          ${inference_dir}/${part}/inference.JOB.log \
          python3 ../../infer.py \
            --resume data/exp/largesound_ABd5_ABd10_510_ESC-50/ep1.checkpoint \
            --exp_dir data/exp/${tag} \
            --rank JOB \
	          --inference_mode 'sampling' \
            --n_samples 1 \
            --seed 888 \
	          --data_json $data_json \
            --generate_target audio \
            --fixed_length True \
            --maxlen_ratio -1 \
            --minlen_ratio -1 \
	          --output_dir ${inference_dir}/${part}/JOB \
            $inference_opts
    done
fi