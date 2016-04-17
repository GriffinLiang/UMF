function new_input = matrix_out_product(input1, input2)

dim1 = size(input1, 1);
dim2 = size(input2, 1);
new_dim = dim1*dim2;
new_input = zeros(new_dim, size(input1, 2));
for ii = 1:size(input1, 2)
    temp = input1(:, ii)*input2(:, ii)';
    new_input(:, ii) = temp(:);
end

end