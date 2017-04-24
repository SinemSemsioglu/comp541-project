module TRAIN
using Knet, JLD
import CDBN

export main

function main(;
                images=rand(100,100,1,100), # mock data of 100 instances of 100x100 images
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
                save_feature_info=0,
                feature_path="feature.libsvm",
                mode=0 #input is real for mode 0, binary for mode 1
                )
    # what is the format of images?
    numImages = size(images,4)
    models = Any[]
    states = Any[]
    pools = Any[]

    for imIndex=1:numImages
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)
        if imIndex > 1
            prev_modelz = models[imIndex-1]
            modelz, statez, poolz = CDBN.main(input=image, numlayers=numlayers, numfilters=numfilters, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0, modelz=prev_modelz)
        else
            modelz, statez, poolz = CDBN.main(input=image, numlayers=numlayers, numfilters=numfilters, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0)
        end
        push!(models, modelz)
        push!(states, statez)
        push!(pools, poolz)
    end

    if save_model_data == 1
        save(model_path, "models", models, "states", states)
    end

    if save_feature_info == 1
        #TODO write a libsvm file
    end
end

end
