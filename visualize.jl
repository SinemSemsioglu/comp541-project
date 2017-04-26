module VISUALIZE
export visualize_filter

using ImageView

function visualize_filter(filter)
    shifted = filter - minimum(filter)
    normalized = shifted / maximum(shifted)
    imshow(normalized)
end

end
