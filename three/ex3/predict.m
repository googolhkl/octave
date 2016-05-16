function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
% X 는 5000*400
[m,n] = size(X);  % 트레이닝 갯수 
% Theta1은  25 * 401
% Theta2는  10 * 26
num_labels = size(Theta2, 1);   % 구분갯수
num_labels

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

size([ones(m,1) X])
h1 = sigmoid([ones(m,1) X] * Theta1');
size([ones(m,1) h1])
h2 = sigmoid([ones(m,1) h1] * Theta2');
size(h2)
[~,p] = max(h2,[],2);
p
% =========================================================================


end
