# A Quick Training Script Guide for the Modal Adapter Learning based Text Image Translation Model.
# Please update the corresponding path and hyper-parameters before running the code in your own environment!
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/E2TIMT
src_lang=${src_language_setting}
tgt_lang=${tgt_language_setting}
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
model_path=${path_of_model_saving}
teacher_model_path=${path_of_loaded_teacher_model}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
batch_size=${batch_size}
task_name=embedding_modal_adapter   # embedding_modal_adapter or sequential_modal_adapter
total_step=${total_training_step}
valid_step=${validate_step_interval}
saved_step=${saving_model_step_interval}

cl_weight=${loss_weight_of_contrastive_learning_loss}
cl_vv_weight=${loss_weight_of_vision_vision_contrastive_learning_loss}
cl_vl_weight=${loss_weight_of_vision_language_contrastive_learning_loss}
cl_ll_weight=${loss_weight_of_language_language_contrastive_learning_loss}

# Path of Text Image Machine Translation Dataset | lmdb file.
train_path=${path_of_timt_train_dataset}
valid_path=${path_of_timt_valid_dataset}

# Path of Textual Machine Translation Dataset | txt file.
txt_train_src=${path_of_text_mt_train_dataset_source_language}
txt_train_tgt=${path_of_text_mt_train_dataset_target_language}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Model Folder.'
if [ -d ${model_path}/${exp_name}/ ];then
  rm -rf ${model_path}/${exp_name}/
fi

echo 'Start to train ...'
${python_path} ${code_path}/trainer.py \
--task ${task_name} \
--imgW ${img_width} \
--train_data ${train_path} \
--valid_data ${valid_path} \
--saved_model ${model_path} \
--exp_name ${exp_name} \
--src_vocab ${vocab_src} --tgt_vocab ${vocab_tgt} \
--batch_size ${batch_size} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \
--sensitive \
--teacher_path ${teacher_model_path}/best_bleu \
--image_encoder_tuning False \
--text_encoder_tuning False \
--ocr_tr_encoder_tuning False \
--mt_tr_encoder_tuning False \
--mt_tr_decoder_tuning False \
--MT_Task --MT_Weight ${mt_weight} --external_mt yes \
--CL_Task --CL_Weight ${cl_weight} \
--CL_vv_Weight ${cl_vv_weight} \
--CL_vl_Weight ${cl_vl_weight} \
--CL_ll_Weight ${cl_ll_weight}
--num_iter ${total_step} \
--valInterval ${valid_step} \
--saveInterval ${saved_step}

echo 'Scripts Done.'
