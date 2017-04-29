module WRITE_LIBSVM
export write_libsvm, convert_one_hot_to_class

#assume each row of the data is an instance and data is a 2d matrix
# labels is a numInstances x 1 vector of labels

function write_libsvm(filepath, data, labels, one_hot)
    if one_hot == 1
        labels = convert_one_hot_to_class(labels)
    end
    f = open(filepath, "w")

    num_rows = size(data, 1)
    num_cols = size(data, 2)

    for row=1:num_rows
        ce_row = data[row,:]
        write(f, string("+", labels[row]))

        for col=1:num_cols
            value = ce_row[col]
# if (value != 0)
                write(f, " ", string(col), ":", string(value))
#           end
        end
        write(f, "\n")
    end
    close(f)
end

function convert_one_hot_to_class(labels)
    num_labels = size(labels,2)
    labels_vector = Array{Int}(num_labels)
    for label_index=1:num_labels
        label_vector = labels[:,label_index]
        labels_vector[label_index] = Int(find(label_vector)[1])
    end

    return labels_vector
end

end
