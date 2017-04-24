module VISUALIZE
export visualize_filter

using ImageView

function visualize_filter(filter)
    shifted = filter - minimum(filter)
    normalized = filter / maximum(filter)
    imshow(normalized)
end

end
