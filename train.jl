module TRAIN
using Knet, JLD
import CDBN, WRITE_LIBSVM

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
                labels=[], #init a 100 length vector
                one_hot=0,
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
        data = convert_to_data_matrix(pools, numlayers)
        WRITE_LIBSVM.write_libsvm(feature_path, data, labels, one_hot)
    end
end

function convert_to_data_matrix(poolz, num_layers)
    num_instances = size(poolz,1);
    vector_size, pool_sizes = get_size_feature_vector(poolz[1], num_layers)
    data_arr = Array{Float64}(num_instances, vector_size)

    for p=1:num_instances
        po = poolz[p]
        f = reshape(po[1], 1, pool_sizes[1])

        for l=2:num_layers
            f = [f reshape(po[l], 1, pool_sizes[l])]
        end

        data_arr[p,:] = f
    end

    return data_arr
end

function get_size_feature_vector(pools, num_layers)
    vector_size = 0
    pool_sizes = Any[]
    for layer_index=1:num_layers
        pool_size = size(pools[layer_index])
        pool_size_mult = multiply_size(pool_size)
        push!(pool_sizes, pool_size_mult)
        vector_size += pool_size_mult
    end
    return vector_size, pool_sizes
end

#im sure there is a better way to do this and i dont know what
#effectively multiplies contents of a tuple
function multiply_size(sizes)
    size_mult = 1
    for this_size in sizes
        size_mult = size_mult * this_size
    end
    return size_mult
end

end
