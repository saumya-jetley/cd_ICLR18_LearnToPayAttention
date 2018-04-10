:<<'END'
#-------------------------------TRAIN-------------------------------------------------------------
# Train cifar-10 (2level_1global_with_2_concatDP)
export model_archi_local1='models/models_CIFARS/2_level_atten_1globalDP/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_1globalDP/2_vgg_local.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_1globalDP/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_1globalDP/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_1globalDP/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_1globalDP/vgg_match.lua'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='train'
export save='#logs/trials/2_level_atten_1global_concatDP'
export test_batchSize=10
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord_concat_2level_dp.txt


# Train cifar-10 (2levels-HL-PAN)
export model_archi_local1='models/models_CIFARS/2_level_atten_higherlevel_PAN/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_higherlevel_PAN/2_vgg_local.lua'
export model_archi_global1='models/models_CIFARS/2_level_atten_higherlevel_PAN/1_vgg_global.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_higherlevel_PAN/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_higherlevel_PAN/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_higherlevel_PAN/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_higherlevel_PAN/vgg_match.lua'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='train'
export save='#logs/trials/2_level_atten_1global_hlll_PAN_10'
export test_batchSize=10
th ./lua_source/main_multiplicativeatt_hl_ll.lua | tee runtimerecord.txt


# Train cifar-10 (3level_1global_with_upprojectedfeatures)
export model_archi_local1='models/models_CIFARS/3_level_atten_1global_upprojecting/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/3_level_atten_1global_upprojecting/2_vgg_local.lua'
export model_archi_local3='models/models_CIFARS/3_level_atten_1global_upprojecting/3_vgg_local.lua'
export model_archi_global3='models/models_CIFARS/3_level_atten_1global_upprojecting/3_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/3_level_atten_1global_upprojecting/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/3_level_atten_1global_upprojecting/2_vgg_atten.lua'
export model_archi_atten3='models/models_CIFARS/3_level_atten_1global_upprojecting/3_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/3_level_atten_1global_upprojecting/vgg_match.lua'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='train'
export save='#logs/trials/3_level_atten_1global_upprojfeats'
export batchSize=64
export test_batchSize=10
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord_cifar10.txt


# Train cifar-10 (2level_1global_with_2_independent_prediction_pipelines)
export model_archi_local1='models/models_CIFARS/2_level_atten_1global_pred-ensemble/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_1global_pred-ensemble/2_vgg_local.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_1global_pred-ensemble/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_1global_pred-ensemble/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_1global_pred-ensemble/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_1global_pred-ensemble/vgg_match.lua'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='train'
export save='#logs/trials/2_level_atten_1global_pred-ensemble'
export test_batchSize=10
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt


# Train cifar-10 (2level_1global_with_2_independent_prediction_pipelines)
export model_archi_local1='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/2_vgg_local.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_1global_pred-ensembleDP/vgg_match.lua'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='train'
export save='#logs/trials/2_level_atten_1global_pred-ensembleDP'
export test_batchSize=10
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord_predEnsemble_2level_dp.txt


# Train cifar-10 (baseline vgg model)
export model_archi='./models/models_CIFARS/vgg_bn_drop.lua'
export dataset='./#dataset/CIFAR10/cifar10_whitened.t7'
export num_classes=10
export mode='train'
export save='#logs/baseline_cifar10'
export test_batchSize=10
th ./lua_source/main.lua | tee runtimerecord.txt


# Train cifar-10 (1level-with DP)
export model_archi_local='models/models_CIFARS/1_level_atten_softmax_conv_DPversion/1_vgg_conv.lua' 
export model_archi_global='models/models_CIFARS/1_level_atten_softmax_conv_DPversion/2_vgg_full.lua'
export model_archi_atten='models/models_CIFARS/1_level_atten_softmax_conv_DPversion/3_atten_softmax_dp.lua'
export model_archi_match='models/models_CIFARS/1_level_atten_softmax_conv_DPversion/4_match_singleimagepred.lua'
export dataset='./#dataset/CIFAR10/cifar10_whitened.t7'
export num_classes=10
export mode='train'
export save='#logs/attention_level1_cifar10'
export test_batchSize=10
th ./lua_source/main_AttLevel1_1global.lua | tee runtimerecord.txt
#


# Train cifar-10 (1level)
export model_archi_local='models/models_CIFARS/1_level_atten_softmax_conv/1_vgg_conv.lua' 
export model_archi_global='models/models_CIFARS/1_level_atten_softmax_conv/2_vgg_full.lua'
export model_archi_atten='models/models_CIFARS/1_level_atten_softmax_conv/3_atten.lua'
export model_archi_match='models/models_CIFARS/1_level_atten_softmax_conv/4_match_singleimagepred.lua'
export dataset='./#dataset/CIFAR10/cifar10_whitened.t7'
export num_classes=10
export mode='train'
th ./lua_source/main_AttLevel1_1global.lua | tee runtimerecord.txt

#----------------------------------TEST----------------------------------------------------------
END

# Test cifar-10 (2level_1global)
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_2.net'
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mglobal_2.net' 
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mmatch.net'
export dataset='/media/sjvision/DATASETDISK/Att_ObjectDiscovery-data/Data/t7_files/objDiscoData_ach_norm.t7' #*IMP*
export num_classes=10   #*IMP*
export mode='test'
export test_batchSize=1
export save_attention=true
export save='att2_cifar10_ach' #*IMP*
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#Test accuracy: not-required-here	



:<<'END'
# Test cifar-10 (2levels-HL-PAN)
export model_wts_local1='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/mlocal_2.net'
export model_wts_global1='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/mglobal_1.net'
export model_wts_global2='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/mglobal_2.net'
export model_wts_atten1='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2_level_atten_1global_hlll_PAN_10/mmatch.net'
export dataset='/media/sjvision/DATASETDISK/Att_ObjectDiscovery-data/Data/t7_files/objDiscoData_ach_norm.t7' #*IMP*
export num_classes=10 #*MP*
export mode='test'
export test_batchSize=1
export save_attention=true
export save='pan_cifar10_ach' #*IMP*
th ./lua_source/main_multiplicativeatt_hl_ll.lua | tee runtimerecord.txt
#Test accuracy: not-required-here




#Test cifar-10 (main)
#export save=logs/${model}_${RANDOM}${RANDOM} # saving done in the code
export model_archi="./models/models_CIFARS/vgg_bn_drop"
export model_wts="./#logs/#logs_CIFAR10/vgg_bn_drop_withHFLIP/model.net" 
export dataset="./#dataset/CIFAR10/cifar10_prov.t7" 
export num_classes=10
export mode="test"
export batchSize=125
th ./lua_source/main.lua | tee runtimerecord.txt
#Test Accuracy: 92.3


#Test cifar-10 (main)
#export save=logs/${model}_${RANDOM}${RANDOM} # saving done in the code
export model_archi="./models/models_CIFARS/vgg_bn_drop"
export model_wts="./#logs/#logs_CIFAR10/vgg_bn_drop_withHFLIP/model.net" 
export dataset="./#dataset/CIFAR10/cifar10_whitened.t7" 
export num_classes=10
export mode="test"
export batchSize=125
th ./lua_source/main.lua | tee runtimerecord.txt
#Test Accuracy: 92.3
# The two copies are probably the same


# Test cifar-10 (1level)
export model_archi_local='models/models_CIFARS/1_level_atten_softmax_conv/1_vgg_conv.lua' 
export model_archi_global='models/models_CIFARS/1_level_atten_softmax_conv/2_vgg_full.lua'
export model_archi_atten='models/models_CIFARS/1_level_atten_softmax_conv/3_atten.lua'
export model_archi_match='models/models_CIFARS/1_level_atten_softmax_conv/4_match_singleimagepred.lua'
export model_wts_local='#logs/#logs_CIFAR10/1atten_softmax_conv/model_local.net'
export model_wts_global='#logs/#logs_CIFAR10/1atten_softmax_conv/model_global.net'
export model_wts_atten='#logs/#logs_CIFAR10/1atten_softmax_conv/model_atten.net' 
export model_wts_match='#logs/#logs_CIFAR10/1atten_softmax_conv/model_match.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel1_1global.lua | tee runtimerecord.txt
#Test accuracy:	94.39


# Test cifar-10 (1level)
export model_archi_local='models/models_CIFARS/1_level_atten_softmax_conv/1_vgg_conv.lua' 
export model_archi_global='models/models_CIFARS/1_level_atten_softmax_conv/2_vgg_full.lua'
export model_archi_atten='models/models_CIFARS/1_level_atten_softmax_conv/3_atten.lua'
export model_archi_match='models/models_CIFARS/1_level_atten_softmax_conv/4_match_singleimagepred.lua'
export model_wts_local='#logs/#logs_CIFAR10/1atten_softmax_conv/model_local.net'
export model_wts_global='#logs/#logs_CIFAR10/1atten_softmax_conv/model_global.net'
export model_wts_atten='#logs/#logs_CIFAR10/1atten_softmax_conv/model_atten.net' 
export model_wts_match='#logs/#logs_CIFAR10/1atten_softmax_conv/model_match.net'
export dataset='./#dataset/CIFAR10/cifar10_whitened.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel1_1global.lua | tee runtimerecord.txt
#Test accuracy:	94.39


# Test cifar-10 (2level_1global)
export model_archi_local1='models/models_CIFARS/2_level_atten_1global/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_1global/2_vgg_local.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_1global/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_1global/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_1global/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_1global/vgg_match.lua'
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_2.net'
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mglobal_2.net' 
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#Test accuracy:	94.58	


# Test cifar-10 (2level_1global)
export model_archi_local1='models/models_CIFARS/2_level_atten_1global/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_1global/2_vgg_local.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_1global/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_1global/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_1global/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_1global/vgg_match.lua'
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mlocal_2.net'
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mglobal_2.net' 
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_1global/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_whitened.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#Test accuracy:	94.58	


# Test cifar-10 (2level_2global)
export model_archi_local1='models/models_CIFARS/2_level_atten_2global/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_2global/2_vgg_local.lua'
export model_archi_global1='models/models_CIFARS/2_level_atten_2global/1_vgg_global.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_2global/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_2global/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_2global/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_2global/vgg_match.lua'
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/mlocal_2.net'
export model_wts_global1='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/mglobal_1.net' 
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/mglobal_2.net' 
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_2global/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel2_2global.lua | tee runtimerecord.txt
#Test accuracy:	94.32


# Test cifar-10 (3level_1global)
export model_archi_local_1='models/models_CIFARS/3_level_atten_1global/1_vgg_local.lua'
export model_archi_local_2='models/models_CIFARS/3_level_atten_1global/2_vgg_local.lua'
export model_archi_local_3='models/models_CIFARS/3_level_atten_1global/3_vgg_local.lua'
export model_archi_global_3='models/models_CIFARS/3_level_atten_1global/3_vgg_global.lua'
export model_archi_atten_1='models/models_CIFARS/3_level_atten_1global/1_vgg_atten.lua'
export model_archi_atten_2='models/models_CIFARS/3_level_atten_1global/2_vgg_atten.lua'
export model_archi_atten_3='models/models_CIFARS/3_level_atten_1global/3_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/3_level_atten_1global/vgg_match.lua'
export model_wts_local_1='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/mlocal_1.net'
export model_wts_local_2='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/mlocal_2.net'
export model_wts_local_3='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/mlocal_3.net'
export model_wts_global_3='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/mglobal_3.net'
export model_wts_atten_1='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/matten_1.net'
export model_wts_atten_2='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/matten_2.net'
export model_wts_atten_3='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/matten_3.net'
export model_wts_match='#logs/#logs_CIFAR10/3atten_e-7_3levels_1global/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord.txt
#Test accuracy:	93.55


# Test cifar-10 (2levels-HL)
export model_archi_local1='models/models_CIFARS/2_level_atten_higherlevel/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_higherlevel/2_vgg_local.lua'
export model_archi_global1='models/models_CIFARS/2_level_atten_higherlevel/1_vgg_global.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_higherlevel/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_higherlevel/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_higherlevel/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_higherlevel/vgg_match.lua'
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/mlocal_2.net'
export model_wts_global1='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/mglobal_1.net'
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/mglobal_2.net'
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_higherlevel/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_multiplicativeatt_hl_ll.lua | tee runtimerecord.txt
#Test accuracy:	81.01


# Test cifar-10 (2Levels-LL)
export model_archi_local1='models/models_CIFARS/2_level_atten_lowerlevel/1_vgg_local.lua'
export model_archi_local2='models/models_CIFARS/2_level_atten_lowerlevel/2_vgg_local.lua'
export model_archi_global1='models/models_CIFARS/2_level_atten_lowerlevel/1_vgg_global.lua'
export model_archi_global2='models/models_CIFARS/2_level_atten_lowerlevel/2_vgg_global.lua'
export model_archi_atten1='models/models_CIFARS/2_level_atten_lowerlevel/1_vgg_atten.lua'
export model_archi_atten2='models/models_CIFARS/2_level_atten_lowerlevel/2_vgg_atten.lua'
export model_archi_match='models/models_CIFARS/2_level_atten_lowerlevel/vgg_match.lua'
export model_wts_local1='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/mlocal_1.net'
export model_wts_local2='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/mlocal_2.net'
export model_wts_global1='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/mglobal_1.net'
export model_wts_global2='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/mglobal_2.net'
export model_wts_atten1='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/matten_1.net'
export model_wts_atten2='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/matten_2.net'
export model_wts_match='#logs/#logs_CIFAR10/2atten_e-7_2levels_lowerlevel/mmatch.net'
export dataset='./#dataset/CIFAR10/cifar10_prov.t7'
export num_classes=10
export mode='test'
export batchSize=125
th ./lua_source/main_multiplicativeatt_hl_ll.lua | tee runtimerecord.txt
#Test accuracy:	61.69
END	
