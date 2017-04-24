using MAT, Knet

import TRAIN

function main()
    path = "natural_images/" # will be taken as an argument later since it depends on file organization
    images = READ_IMAGES.get_square_color_images(path,1,200)
    TRAIN.main(;images=images[:,:,:,1:31],save_model_data=1,model_path="kyoto_natural_images_first_31.jld")
end
