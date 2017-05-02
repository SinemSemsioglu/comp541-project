using Knet, JLD

import TRAIN, READ_IMAGES

function main()
    path = "natural_images/" # will be taken as an argument later since it depends on file organization
    images = READ_IMAGES.get_square_color_images(path,1,100,1)
    images .= mean(images, (1,2,3))

    models, states, recons = TRAIN.main(;images=images[:,:,:,:], debug=1, gradient_lr=0.02, sparsity_lr = 300, max_iterations=[124, 62], save_model_data=1, model_path="kyoto_adam_0.02_300_124_62_grayscale.jld")
end
