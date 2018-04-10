#-------------------------------TRAIN-------------------------------------------------------------
:<<'END'
# Train cubs-200 (2level_1global_concatDP)
export model_archi_local1='models/models_CUBS200/2_level_atten_1globalDP/1.1_vgg_local.lua'
export model_archi_local2='models/models_CUBS200/2_level_atten_1globalDP/1.2_vgg_local.lua'
export model_archi_global2='models/models_CUBS200/2_level_atten_1globalDP/2.2_vgg_global.lua'
export model_archi_atten1='models/models_CUBS200/2_level_atten_1globalDP/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_CUBS200/2_level_atten_1globalDP/3.2_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/2_level_atten_1globalDP/4_vgg_match.lua'
export model_wts_local1='#logs/trials/2_level_atten_1global_concatDP100/----------'
export model_wts_local2='#logs/trials/2_level_atten_1global_concatDP100/-----------'
export model_wts_global2='#logs/trials/2_level_atten_1global_concatDP100/---------' 
export model_wts_atten1='#logs/trials/2_level_atten_1global_concatDP100/---------'
export model_wts_atten2='#logs/trials/2_level_atten_1global_concatDP100/---------'
export model_wts_match='#logs/trials/2_level_atten_1global_concatDP100/-----------'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=128
export testbatchSize=2
export learningRate=0.1
export epoch_step='{30,60,90,120,150,180,210,240,270,300}'
export max_epoch=320
export lr_step='{2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}'
export mode='train'
export save='#logs/trials/2_level_atten_1global_cubs_concatDP'
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord_cubs_2level_concatdp.txt
END

# Train cubs-200 (2levels-HL-PAN)
export model_archi_local1='models/models_CUBS200/2_level_atten_higherlevel_PAN/1.1_vgg_local.lua'
export model_archi_local2='models/models_CUBS200/2_level_atten_higherlevel_PAN/1.2_vgg_local.lua'
export model_archi_global1='models/models_CUBS200/2_level_atten_higherlevel_PAN/2.1_vgg_global.lua'
export model_archi_global2='models/models_CUBS200/2_level_atten_higherlevel_PAN/2.2_vgg_global.lua'
export model_archi_atten1='models/models_CUBS200/2_level_atten_higherlevel_PAN/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_CUBS200/2_level_atten_higherlevel_PAN/3.2_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/2_level_atten_higherlevel_PAN/4_vgg_match.lua'
export model_wts_local1='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/mlocal_1.net'
export model_wts_local2='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/mlocal_2.net'
export model_wts_global1='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/mglobal_1.net' 
export model_wts_global2='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/mglobal_2.net' 
export model_wts_atten1='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/matten_1.net'
export model_wts_atten2='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/matten_2.net'
export model_wts_match='#logs/trials/2_level_atten_1global_hlll_PAN_100_re/mmatch.net'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=32
export test_batchSize=2
export learningRate=0.1
export epoch_step='{30,60,90,120,150,180,210,240,270,300}'
export max_epoch=320
export lr_step='{2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}'
export mode='train'
export save='#logs/trials/2_level_atten_1global_hlll_PAN_cubs'
th ./lua_source/main_multiplicativeatt_hl_ll.lua | tee runtimerecord_PAN_cubs.txt


:<<'END'
#Test cubs-200 (main)
export model_archi="./models/models_CUBS200/vgg_bn_drop.lua"
export model_wts="./#logs/#logs_CUBS200/vgg_bn_drop/vgg_bn_drop_rot_aug_0.1_0.2_0.4lr_30all/model.net" 
export dataset="./#dataset/CUBS-200/cubs-200.t7" 
export num_classes=200
export mode="test"
export batchSize=2
th ./lua_source/main.lua | tee runtimerecord.txt
#Test Accuracy:65.188125647221



# Test cubs-200 (2level_1global)
export model_archi_local1='models/models_CUBS200/2_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local2='models/models_CUBS200/2_level_atten_1global/1.2_vgg_local.lua'
export model_archi_global2='models/models_CUBS200/2_level_atten_1global/2.2_vgg_global.lua'
export model_archi_atten1='models/models_CUBS200/2_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_CUBS200/2_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/2_level_atten_1global/4_vgg_match.lua'
export model_wts_local1='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/mlocal_1.net'
export model_wts_local2='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/mlocal_2.net'
export model_wts_global2='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/mglobal_2.net' 
export model_wts_atten1='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/matten_1.net'
export model_wts_atten2='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/matten_2.net'
export model_wts_match='#logs/#logs_CUBS200/vgg_conv_atten_e-7_2levels_1global_0.1_0.2_0.4lr_30all/mmatch.net'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=2
export mode='test'
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#Test accuracy:	73.127373144632



# Test cubs-200 (3level_1global)
export model_archi_local_1='models/models_CUBS200/3_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local_2='models/models_CUBS200/3_level_atten_1global/1.2_vgg_local.lua'
export model_archi_local_3='models/models_CUBS200/3_level_atten_1global/1.3_vgg_local.lua'
export model_archi_global_3='models/models_CUBS200/3_level_atten_1global/2.3_vgg_global.lua'
export model_archi_atten_1='models/models_CUBS200/3_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten_2='models/models_CUBS200/3_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_atten_3='models/models_CUBS200/3_level_atten_1global/3.3_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/3_level_atten_1global/4_vgg_match.lua'
export model_wts_local_1='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/mlocal_1.net'
export model_wts_local_2='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/mlocal_2.net'
export model_wts_local_3='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/mlocal_3.net'
export model_wts_global_3='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/mglobal_3.net'
export model_wts_atten_1='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/matten_1.net'
export model_wts_atten_2='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/matten_2.net'
export model_wts_atten_3='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/matten_3.net'
export model_wts_match='#logs/#logs_CUBS200/vgg_conv_atten_e-7_3levels_1global_0.1_0.2_0.4lr_30all/mmatch.net'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=2
export mode='test'
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord.txt
#Test accuracy:	73.213669313082


# Train cubs-200 (main)
export model_archi="./models/models_CUBS200/vgg_bn_drop.lua"
export model_wts="./#logs/#logs_CUBS200/-----------" 
export dataset="./#dataset/CUBS-200/cubs-200.t7" 
export num_classes=200
export batchSize=128
export learningRate=0.1
export epoch_step='{30,60,90,120,150,180,210,240,270,300}'
export max_epoch=320
export lr_step='{2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}'
export mode="train"
th ./lua_source/main.lua | tee runtimerecord.txt
#Test Accuracy:65.188125647221
END

:<<'END'
# Train cubs-200 (2level_1global)
export model_archi_local1='models/models_CUBS200/2_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local2='models/models_CUBS200/2_level_atten_1global/1.2_vgg_local.lua'
export model_archi_global2='models/models_CUBS200/2_level_atten_1global/2.2_vgg_global.lua'
export model_archi_atten1='models/models_CUBS200/2_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_CUBS200/2_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/2_level_atten_1global/4_vgg_match.lua'
export model_wts_local1='#logs/#logs_CUBS200/-----------'
export model_wts_local2='#logs/#logs_CUBS200/-----------'
export model_wts_global2='#logs/#logs_CUBS200/---------' 
export model_wts_atten1='#logs/#logs_CUBS200/---------'
export model_wts_atten2='#logs/#logs_CUBS200/---------'
export model_wts_match='#logs/#logs_CUBS200/-----------'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=128
export learningRate=0.1
export epoch_step='{30,60,90,120,150,180,210,240,270,300}'
export max_epoch=320
export lr_step='{2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}'
export mode='train'
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#Test accuracy:	73.127373144632



# Train cubs-200 (3level_1global)
export model_archi_local_1='models/models_CUBS200/3_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local_2='models/models_CUBS200/3_level_atten_1global/1.2_vgg_local.lua'
export model_archi_local_3='models/models_CUBS200/3_level_atten_1global/1.3_vgg_local.lua'
export model_archi_global_3='models/models_CUBS200/3_level_atten_1global/2.3_vgg_global.lua'
export model_archi_atten_1='models/models_CUBS200/3_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten_2='models/models_CUBS200/3_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_atten_3='models/models_CUBS200/3_level_atten_1global/3.3_vgg_atten.lua'
export model_archi_match='models/models_CUBS200/3_level_atten_1global/4_vgg_match.lua'
export model_wts_local_1='#logs/#logs_CUBS200/---------'
export model_wts_local_2='#logs/#logs_CUBS200/---------'
export model_wts_local_3='#logs/#logs_CUBS200/---------'
export model_wts_global_3='#logs/#logs_CUBS200/---------'
export model_wts_atten_1='#logs/#logs_CUBS200/----------'
export model_wts_atten_2='#logs/#logs_CUBS200/----------'
export model_wts_atten_3='#logs/#logs_CUBS200/---------'
export model_wts_match='#logs/#logs_CUBS200/---------'
export dataset='./#dataset/CUBS-200/cubs-200.t7'
export num_classes=200
export batchSize=128
export learningRate=0.1
export epoch_step='{30,60,90,120,150,180,210,240,270,300}'
export max_epoch=320
export lr_step='{2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}'
export mode='train'
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord.txt
#Test accuracy:	73.213669313082
END
