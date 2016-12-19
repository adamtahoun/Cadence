function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

siggy = sigmoid(X * theta);
yy = log(siggy);
ny = log(1 - siggy);

norm = (lambda / (2 * m)) * sum(theta(2:length(theta),:) .^ 2);
reg_sum = (lambda / m) .* theta;
reg_sum(1) = 0;

J = (1 / m) * sum((-y .* yy) - ((1 - y) .* ny)) + norm;
grad = sum(((siggy - y) .* X) ./ m) + reg_sum';


% =============================================================

end
