#!/bin/bash
set -xe

function main {
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip install sentencepiece tensorflow_hub

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            #cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            #mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" == "cpu" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        elif [ "${device}" == "cuda" ];then
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
	elif [ "${device}" == "xpu" ];then
	    OOB_EXEC_HEADER=" ZE_AFFINITY_MASK=0.${i} "
        fi
	OOB_EXEC_HEADER+=" ${OOB_EXTRA_HEADER} "
        printf " ${OOB_EXEC_HEADER} \
	    python run_classifier.py --data_dir=${DATASET_DIR} --output_dir=./results  \
                --albert_config_file=./albert_base/albert_config.json   --spm_model_file=./albert_base/30k-clean.model \
                --do_train --do_lower_case --max_seq_length=128 --optimizer=adamw --task_name=MNLI \
                --warmup_step=100 --learning_rate=3e-5 --train_step=400 \
                --save_checkpoints_steps=1001  --train_batch_size=${batch_size} --device ${device} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git -b gpu_oob

# Start
main "$@"
