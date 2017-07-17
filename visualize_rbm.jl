module VISUALIZE_RBM

export visualize_filter, visualize

using ImageView, Knet, MAT

function normalize_filter(filter)
	max_value = maximum(abs(filter))
	normalized = filter / max_value
	normalized += 1
	normalized = normalized /2
	
	return normalized
end

# each row represents an image to be visualized
function visualize(images, im_width, im_height, num_rows, num_cols, isfilter; path="")	
	num_images = size(images, 1)
	
	# check dims 
	#if im_width * im_height != size(filters_,1)
    #    print("incorrect arguments for image width and height, their multiplication should equal to the number of rows in filters")
    #end
	
	#if num_filters !== num_rows * num_cols
    #    print("incorrect arguments for number of rows and columns, their multiplication should equal number of filters in each channel")
    #end
			
	all_images = zeros(num_rows * im_height, num_cols * im_width)
	
	for im = 1:num_images
        r = div(im -1, num_cols)
        c = mod(im -1, num_cols)

        image = reshape(images[im, :], im_width, im_height)
		
		if isfilter
			normalized = normalize_filter(image)
		else 
			normalized = image
		end

        all_images[(r * im_height) + 1: ((r +1) * im_height), c * im_width + 1: (c+1) * im_width] = normalized
    end
	
	imshow(all_images)
	
	#saves as a .mat file
	if path != ""
		file = matopen(path, "w")
		write(file, "images", all_images)
		close(file)
	end
end

function visualize_filter(filter)
    normalized = normalize_filter(filter)
    imshow(normalized)
end


end
