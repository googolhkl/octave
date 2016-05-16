function [J, grad] = costFunctionReg(theta, X, y, lambda)
%m = length(y); % number of training examples
[m, n] = size(X);
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);

J = (sum(-y.*log(h)-(1-y).*log(1-h))/m) + (lambda/2*m)*sum(theta.^2);
grad = (1/m) * ((sigmoid(X*theta) -y)' *X);
[r,c] = size(grad);
THETA = theta'; % 1*28  grad도 1*28
for i= 2:c,
    grad(1,c) = grad(1,c) + (lambda .* THETA(1,c))/m;
end
end
