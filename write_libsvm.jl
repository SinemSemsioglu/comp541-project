module WRITE_LIBSVM

#assume each row of the data is an instance and data is a 2d matrix
# labels is a numInstances x 1 vector of labels
function write(filepath, data, labels)
f = open(filepath, "w")

num_rows = size(data, 1)
num_cols = size(data, 2)

for row=1:num_rows
    row = data[row,:]
    write(f, labels[row])

    for col=1:num_cols
        value = row[col]
        if (value != 0)
            write(f, " ", col, ":", value)
        end
    end
    write(f, "\n")
end

end
