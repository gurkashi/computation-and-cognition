close all;
clear;
clc;

%% parameters
size_x = 1;
size_y = 100;
size_b = 1;
size_h = 500;
size_test = 10000;
size_ts = 10000;
lambda = 0.0; % regularization constant for linear perceptron
alpha = 0.00; % decay
beta = 0.0; % sparse

%% weight matrices
%K_bar = rand(size_x + size_h + size_b, size_x + size_h + size_b)-0.5;
K_connected = rand(size_x + size_h + size_b) > beta;
K_bar = K_connected .* randn(size_x + size_h + size_b, size_x + size_h + size_b);
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
    h = alpha * h + (1-alpha) * g(K * [x(t); h; b]);
end
C = C / (size_ts - (size_y + 1));
U = U / (size_ts - (size_y + 1));

%% optimal weights of output layer
I = eye(size(C));
W = U / (C - lambda * I);
h = zeros(size_h, 1);

%% test set
x = randn(size_test, size_x);
Sy = zeros(size_y, 1);
Sy2 = zeros(size_y, 1);
Sy0 = zeros(size_y, 1);
Sy02 = zeros(size_y, 1);
Syy0 = zeros(size_y, 1);
for t = size_y+1:size_test
    % signal propagation in the network
    y0 = flip(x(t-size_y:t-1)); % output y0[i] should be the signal at x[t-i]
    y = W * [x(t); h; b];
    h = alpha * h + (1-alpha) * g(K * [x(t); h; b]);
    
    % pearson correlation coefficeints calculation
    Sy = Sy + y;
    Sy2 = Sy2 + y .^ 2;
    Sy0 = Sy0 + y0;
    Sy02 = Sy02 + y0 .^ 2;
    Syy0 = Syy0 + y .* y0;
end

n = size_test - (size_y + 1);
r = (n * Syy0 - Sy .* Sy0) ./ sqrt((n * Sy2 - Sy .^ 2) .* (n * Sy02 - Sy0 .^ 2));
r2 = r .^ 2;
plot(r2);
title(strcat('Memory capacity = ' , num2str(sum(r2))));
