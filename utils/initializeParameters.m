function theta = initializeParameters(hiddenSize, visibleSize)

r  = sqrt(6) / sqrt(hiddenSize+visibleSize);
W = rand(hiddenSize, visibleSize) * 2 * r - r;
theta = W(:);
end
