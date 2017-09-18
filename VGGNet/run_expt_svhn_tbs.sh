:<<'END'
#Test svhn-10 (main)
export model_archi="./models/models_SVHN/vgg_bn_drop.lua"
export model_wts="./#logs/#logs_SVHN/vgg_bn_drop/----" 
export dataset="./#dataset/SVHN/svhn_provider.t7" 
export num_classes=10
export mode="test"
export batchSize=16
th ./lua_source/main.lua | tee runtimerecord.txt
#


# Test svhn-10 (2level_1global)
export model_archi_local1='models/models_SVHN/2_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local2='models/models_SVHN/2_level_atten_1global/1.2_vgg_local.lua'
export model_archi_global2='models/models_SVHN/2_level_atten_1global/2.2_vgg_global.lua'
export model_archi_atten1='models/models_SVHN/2_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_SVHN/2_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_match='models/models_SVHN/2_level_atten_1global/4_vgg_match.lua'
export model_wts_local1='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_local2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_global2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net' 
export model_wts_atten1='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_atten2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/-------.net'
export model_wts_match='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export dataset='./#dataset/SVHN/svhn_provider.t7'
export num_classes=10
export mode='test'
export batchSize=16
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#


# Test svhn-10 (3level_1global)
export model_archi_local_1='models/models_SVHN/3_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local_2='models/models_SVHN/3_level_atten_1global/1.2_vgg_local.lua'
export model_archi_local_3='models/models_SVHN/3_level_atten_1global/1.3_vgg_local.lua'
export model_archi_global_3='models/models_SVHN/3_level_atten_1global/2.3_vgg_global.lua'
export model_archi_atten_1='models/models_SVHN/3_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten_2='models/models_SVHN/3_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_atten_3='models/models_SVHN/3_level_atten_1global/3.3_vgg_atten.lua'
export model_archi_match='models/models_SVHN/3_level_atten_1global/4_vgg_match.lua'
export model_wts_local_1='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_local_2='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_local_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_global_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_atten_1='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_atten_2='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_atten_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_match='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export dataset='./#dataset/SVHN/svhn_provider.t7'
export num_classes=10
export mode='test'
export batchSize=16
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord.txt
#
END


# Train svhn-10 (main)
export model_archi="./models/models_SVHN/vgg_bn_drop.lua"
export model_wts="./#logs/#logs_SVHN/vgg_bn_drop/----" 
export dataset="./#dataset/SVHN/svhn_provider.t7" 
export num_classes=10
export mode="train"
export batchSize=128
export learningRate=1 #maybe 0.1
export epoch_step='25'
export lr_step='0.5'
export max_epoch=300
export save="#logs/svhn_baseline"
th ./lua_source/main.lua | tee runtimerecord.txt
#

:<<'END'
# Train svhn-10 (2level_1global)
export model_archi_local1='models/models_SVHN/2_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local2='models/models_SVHN/2_level_atten_1global/1.2_vgg_local.lua'
export model_archi_global2='models/models_SVHN/2_level_atten_1global/2.2_vgg_global.lua'
export model_archi_atten1='models/models_SVHN/2_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten2='models/models_SVHN/2_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_match='models/models_SVHN/2_level_atten_1global/4_vgg_match.lua'
export model_wts_local1='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_local2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_global2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net' 
export model_wts_atten1='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export model_wts_atten2='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/-------.net'
export model_wts_match='#logs/#logs_SVHN/vgg_conv_atten_e-7_2levels_1global/------.net'
export dataset='./#dataset/SVHN/svhn_provider.t7'
export num_classes=10
export mode="train"
export batchSize=128
export learningRate=1 #maybe 0.1
export epoch_step='25'
export lr_step='0.5'
export max_epoch=300
th ./lua_source/main_AttLevel2_1global.lua | tee runtimerecord.txt
#


# Train svhn-10 (3level_1global)
export model_archi_local_1='models/models_SVHN/3_level_atten_1global/1.1_vgg_local.lua'
export model_archi_local_2='models/models_SVHN/3_level_atten_1global/1.2_vgg_local.lua'
export model_archi_local_3='models/models_SVHN/3_level_atten_1global/1.3_vgg_local.lua'
export model_archi_global_3='models/models_SVHN/3_level_atten_1global/2.3_vgg_global.lua'
export model_archi_atten_1='models/models_SVHN/3_level_atten_1global/3.1_vgg_atten.lua'
export model_archi_atten_2='models/models_SVHN/3_level_atten_1global/3.2_vgg_atten.lua'
export model_archi_atten_3='models/models_SVHN/3_level_atten_1global/3.3_vgg_atten.lua'
export model_archi_match='models/models_SVHN/3_level_atten_1global/4_vgg_match.lua'
export model_wts_local_1='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_local_2='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_local_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_global_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/-----.net'
export model_wts_atten_1='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_atten_2='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_atten_3='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export model_wts_match='#logs/#logs_SVHN/vgg_conv_atten_e-7_3levels_1global/------.net'
export dataset='./#dataset/SVHN/svhn_provider.t7'
export num_classes=10
export mode="train"
export batchSize=128
export learningRate=1 #maybe 0.1
export epoch_step='25'
export lr_step='0.5'
export max_epoch=300
th ./lua_source/main_AttLevel3_1global.lua | tee runtimerecord.txt
#
END
