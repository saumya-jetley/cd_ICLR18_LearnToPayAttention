#!/usr/bin/env bash
export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt


#!/usr/bin/env bash 
# -- wide resnet routine with 200 epochs --?!!?
export learningRate=0.1
export epoch_step="{60,120,160,200}"
export max_epoch=250
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt


#!/usr/bin/env bash
export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train_attenunit_1level.lua | tee $save/log.txt


#!/usr/bin/env bash
export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train_attenunit_2levels.lua | tee $save/log.txt






