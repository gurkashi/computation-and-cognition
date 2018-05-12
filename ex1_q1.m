close all;
clc;
clear;

%% load pre-computed data
if exist(fullfile('esn.mat'), 'file')
    load('esn');
else
    esn;
    save('esn');
end

%% sizes
size_test = 10000;

%% test set
x = randn(size_test, size_x);
r = zeros(size_y, 1);
for t = size_y+1:size_test
    y0 = flip(x(t-size_y:t-1)); % output y0[i] should be the signal at x[t-i]
    y = W * [x(t); h; b];
    h = g(K * [x(t); h; b]);
    
    r = r + (y .* y0);
end
r = r / (size_test - (size_y + 1));
r2 = r .^ 2;

plot(r2);
title(strcat('Memory capacity = ' , num2str(sum(r2))));
