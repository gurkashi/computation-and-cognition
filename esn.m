%% sizes
size_x = 1;
size_y = 100;
size_b = 1;
size_h = 500;
size_ts = 10000;
lambda = 0.001; % regularization constant for linear perceptron

%% weight matrices
K_bar = rand(size_x + size_h + size_b, size_x + size_h + size_b)-0.5;
k = max(real(eig(K_bar)));
K = K_bar(1:size_h, :) / k; % weights connected to hidden layer

%% bias
b = ones(size_b, 1);

%% activations
% hidden layer activation - tanh
g = @(x) (exp(x)-exp(-x)) ./ (exp(x)+exp(-x));

%% training set
x = randn(size_ts, size_x); % inputs from random gaussian noise
h = zeros(size_h, 1); % hidden layer values start at 0
%% training though signal propagation
C = zeros(size_x + size_h + size_b);
U = zeros(size_y, size_x + size_h + size_b);
for t = size_y+1:size_ts
    y0 = flip(x(t-size_y:t-1)); % output y0[i] should be the signal at x[t-i]
    C = C + [x(t); h; b] * [x(t); h; b]';
    U = U + y0 * [x(t); h; b]';
    h = g(K * [x(t); h; b]);
end
C = C / (size_ts - (size_y + 1));
U = U / (size_ts - (size_y + 1));

%% optimal weights of output layer
I = eye(size(C));
W = U / (C - lambda * I);
h = zeros(size_h, 1);