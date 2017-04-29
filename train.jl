module TRAIN
using Knet, JLD
import CDBN

export main

function main(;
                images=rand(100,100,1,100), # mock data of 100 instances of 100x100 images
                initial_model=[],
                numlayers=2,
                filtersize=[10,10],
                numfilters=[24,100],
                pool=[2,2],
                sparsity=[0.003,0.005],
                gradient_lr=0.1,
                sparsity_lr=5,
                cd=1,
                batch=1,
                save_model_data=0,
                model_path="model.jld",
                mode=0 #input is real for mode 0, binary for mode 1
                )

    # what is the format of images?
    numImages = size(images,4)

    cdbn_mode = 0
    prev_modelz = []
    prev_statez = []

    for imIndex=1:numImages
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)
        image -= mean(image)

        if imIndex > 1
            prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters, filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0, modelz=prev_modelz)
        elseif size(initial_model,1) >0
             prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters,  filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, modelz=initial_model, mode=0)
        else
            prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters,  filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0)
        end

# gradient_lr = gradient_lr / (1 + 0.1)
    end

    if save_model_data == 1
        save(model_path, "models", prev_modelz, "states", prev_statez)
    end

    return (prev_modelz, prev_statez)
end

end
