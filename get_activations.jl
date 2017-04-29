module ACTIVATIONS
using Knet
import CDBN, WRITE_LIBSVM

export main

function main(;
    images=rand(100,100,1,100), # mock data of 100 instances of 100x100 images
    model=[],
    pool=[2,2],
    numlayers=2,
    save_feature_info=0,
    feature_path="feature.libsvm",
    labels=[], #init a 100 length vector
    one_hot=0
)

    # what is the format of images?
    numImages = size(images,4)
    pools = Any[]

    cdbn_mode = 1

    for imIndex=1:numImages
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)

        poolz = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, pool=pool, modelz=model)

        push!(pools, poolz)
    end

    data = convert_to_data_matrix(pools, numlayers)

    if save_feature_info == 1

        WRITE_LIBSVM.write_libsvm(feature_path, data, labels, one_hot)
    end

    return data

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
