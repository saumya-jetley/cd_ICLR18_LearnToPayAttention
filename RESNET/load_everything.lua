require 'nn'
require 'cunn'
require 'image'                                                           
require 'nnx'
require 'nngraph'
require 'optim'
model_utils = require 'util.model_utils'

function getParameter(nngraph_model, name)
    local params
    nngraph_model:apply( function(m) if m.name==name then params = m end end)
    return params
end
