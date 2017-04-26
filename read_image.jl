module READ_IMAGES
export get_square_color_images, resize_images

using Images

#file_type 0 for jpg, 1 for png
function get_square_color_images(path, file_type, im_size)
    if file_type == 0
        imFiles = filter(x -> ismatch(r"\.jpg", x), readdir(path));
    elseif file_type == 1
        imFiles = filter(x -> ismatch(r"\.png", x), readdir(path));
    end

    imFiles = map(x-> string(path, x), imFiles)
    numImages = size(imFiles,1)
    images = Array{Float64}(im_size, im_size, 3, numImages)

    #need to normalize image sizes? they are all different
    for imIndex=1:numImages
        # adapted from vgg.jl in Knet repo
        imFile = imFiles[imIndex]
        rawImg = load(imFile)
        newSize = ntuple(i->div(size(rawImg,i)*im_size,minimum(size(rawImg))),2)
        resizedImage = Images.imresize(rawImg, newSize)
        w = div(size(resizedImage,1)-im_size,2)
        h = div(size(resizedImage,2)-im_size,2)
        squaredImage = resizedImage[w+1:w+im_size,h+1:h+im_size]
        channeledImage = permutedims(channelview(squaredImage), (3,2,1))
        floatImage = convert(Array{Float32}, channeledImage)

        images[:,:,:, imIndex] = floatImage;
    end

    return images;
end

function resize_images(images, new_size)
    num_images = size(images, 4)
    resized_images = Array{Float64}(new_size[1], new_size[2], new_size[3], num_images)

    for im_index=1:num_images
        resized_images[:,:,:,im_index] = Images.imresize(images[:,:,:,im_index], new_size)
    end

    return resized_images
end

end
