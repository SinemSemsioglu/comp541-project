using Knet, JLD

import TRAIN, READ_IMAGES, VISUALIZE

function main()
    path = "natural_images/" # will be taken as an argument later since it depends on file organization
    atype =(gpu()>=0 ? KnetArray{Float64} : Array{Float64})
    images = READ_IMAGES.get_square_color_images(path,1,100,1, atype)
    images .= mean(images, (1,2,3))

    progress, models, states, hiddens, recons, stats = TRAIN.main(;images=images[:,:,:,:], debug=1, gradient_lr=0.01, sparsity_lr = 5, max_iterations=[20, 1], save_model_data=0, model_path="kyoto_adam_0.02_300_124_62_grayscale.jld")

    VISUALIZE.visualize_l1_filters(models[1][1], 12, 2)
    VISUALIZE.visualize_l2_filters(models[1][1], models[2][1], 10,10)
end
