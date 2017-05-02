module VISUALIZE
export visualize_filter, visualize_l1_filters, visualize_l2_filters

using ImageView, Knet

function visualize_filter(filter)
#shifted = filter - minimum(filter)
#   normalized = shifted / maximum(shifted)

    normalized = filter / maximum(abs(filter))
    imshow(normalized)
end

function visualize_l1_filters(filters, num_rows, num_cols)
    numfilters = size(filters, 4)
    numchannels = size(filters, 3)

    if numfilters !== num_rows * num_cols
        print("incorrect arguments for number of rows and columns, their multiplication should equal number of filters in each channel")
    else

    fw = size(filters,1)
    fh = size(filters,2)

    all_filters = zeros(num_rows * fh * numchannels, num_cols * fw)

    for chn = 1:numchannels
        chn_v_off = (chn - 1) * fh * num_rows

        for flt=1:numfilters
print(flt, "\n")
            r = div(flt -1, num_cols)
            c = mod(flt -1, num_cols)

            filter = filters[:,:,chn, flt]
    normalized = filter / maximum(abs(filter))
#shifted = filter - minimum(filter)
#  normalized = shifted / maximum(shifted)
            all_filters[chn_v_off + (r * fh) + 1: chn_v_off + ((r +1) * fh), c * fw + 1: (c+1) * fw] = normalized
        end
    end

    imshow(all_filters)
    end
end

function visualize_l2_filters(l1_filters, l2_filters, num_rows, num_cols)
    numchannels = size(l1_filters, 3)
    comb_w_h = size(l1_filters,1) + size(l2_filters,1) - 1
    padd = Int(round((comb_w_h - (size(l1_filters,1) - size(l2_filters,1) + 1))/2))
    combination = zeros(comb_w_h, comb_w_h, numchannels, size(l2_filters, 4))

    for chn=1:numchannels
        l1_chn = reshape(l1_filters[:,:,chn,:], size(l1_filters,1), size(l1_filters,2), size(l1_filters,4), 1)
        for k=1:size(l2_filters,4)
            l2_k = reshape(l2_filters[:,:,:,k], size(l2_filters,1), size(l2_filters,2), size(l2_filters,3),1)
            combination[:,:,chn,k] = conv4(l2_k, l1_chn ; padding=padd)
        end
    end

    visualize_l1_filters(combination, num_rows, num_cols)
end

end
