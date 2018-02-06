function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);
% cost function      
% X_theta_product = X*Theta'; % (X*Theta').*R
% J = (1/2)*sum( ( X_theta_product(R) - Y(R)).^2);
J=0.5*(sum(sum((R.*(X*Theta'-Y)).^2)) + lambda*( sum(sum(Theta.^2)) + sum(sum(X.^2)) ));

% gradients
for i=1:size(R,1)
    % idx=find(R(i,:)==1); % which users rated movie i
    % Theta_temp=Theta(idx,:); % which users' parameters we need
    % Ytemp=Y(i,idx); % which movies by which user
    % X_grad(i,:)=( X(i,:)*Theta_temp' - Ytemp)*Theta_temp + lambda*X(i,:); 
    X_grad(i,:) = ( X(i,:)*Theta(find(R(i,:)==1),:)' - Y(i,find(R(i,:)==1)))*Theta(find(R(i,:)==1),:) + lambda*X(i,:);
    % grad matrix ith row = (ith movie features)*(parameters of users that rated ith movie) - ratings for ith movie
end

for i=1:size(R,2)
    idx=find(R(:,i)==1); 
    X_temp=X(idx,:); 
    Ytemp=Y(idx,i); % which movies by which user
    Theta_grad(i,:)=( Theta(i,:)*X_temp' - Ytemp')*X_temp + lambda*Theta(i,:); 
end

% J = (1/(2*m))*sum( (X*theta - y).^2 ) + 0.5*(lambda/m)*sum(theta(2:length(theta)).^2);
% grad = (1/m)*sum( (X*theta - y).*X) + [0 (lambda/m)*theta(2:length(theta))'];


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

grad = [X_grad(:); Theta_grad(:)];

end