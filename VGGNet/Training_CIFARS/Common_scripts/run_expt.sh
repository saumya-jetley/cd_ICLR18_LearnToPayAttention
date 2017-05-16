#Test cifar-10 (main)
#export save=logs/${model}_${RANDOM}${RANDOM} # done in the code
th main.lua -model_archi vgg_bn_drop -model_wts ./logs/vgg_withHFLIP/model.net -dataset provider.t7 -num_classes 10 |& tee runtimerecord.txt
# Test cifar-10 (1level)

# Test cifar-10 (2level_2global)

# Test cifar-10 (2level_1global)

# Test cifar-10 (3level_3global)

# Test cifar-10 (3level-1global)
