using Knet

import TRAIN, READ_IMAGES

function main()
    path = "natural_images/" # will be taken as an argument later since it depends on file organization
    images = READ_IMAGES.get_square_color_images(path,1,200)

    max_iter = 150
    prev_model = []
    prev_state = []

    for iter=1:max_iter
        model, state = TRAIN.main(;images=images[:,:,:,1:10],save_model_data=1,model_path="kyoto_natural_images_first_10_2.jld", gradient_lr=0.0001, sparsity_lr = 5)
        if iter > 1
            if isapprox(prev_model, model)
                prev_model = model
                prev_state = state
                break
            else
                print("iter: " , iter, " ave diff: ", mean(prev_model .- model))
            end
        end

        prev_state = state
        prev_model = model
    end
end
