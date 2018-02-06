function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
A = sigmoid(Theta1*transpose([ones(m,1) X]));
output_layer = sigmoid( Theta2*([ones(1,m); A]) );
[p output_layer_predictions] = max( output_layer, [], 1);
p=p'; output_layer_predictions=output_layer_predictions';
output_layer = output_layer';

for c = 1:num_labels
y_t = y==c;
  J_k_sum(c,:) = (1/m)*sum( -y_t.*log(output_layer(:,c)) - (1-y_t).*log(1 - output_layer(:,c) ) );
end

J =  sum( J_k_sum) + (lambda/(2*m))*( sum(sum(Theta1(:,2:end).^2,1)) + sum(sum(Theta2(:,2:end).^2,1)) ) ;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for ex_counter = 1:m

  a_1=[1 X(ex_counter,:)];
  z_2=Theta1*transpose(a_1);
  a_2=[1 sigmoid(z_2)'];
  z_3=a_2*transpose(Theta2);
  a_3=sigmoid(z_3);
  
 [a b] = max(a_3); % sigmoid([1 sigmoid(Theta1*transpose([1 X(ex_counter,:)]))']*transpose(Theta2)),[],2
  
  output_node_binary=zeros(1,num_labels);
  output_node_binary(y(ex_counter))=1;
  
  delta_3 = (a_3 - output_node_binary)';
  delta_2 = (transpose(Theta2(:,2:length(Theta2)) )*delta_3).*sigmoidGradient(z_2);
  
  Theta2_grad = Theta2_grad + (delta_3*a_2);
  Theta1_grad = Theta1_grad + (delta_2*a_1);
  
%a1 = [ones(m,1) X];%5000*401;
%z2=a1*Theta1'; %5000*25;
%a2=[ones(m,1) sigmoid(z2)];% 5000*26
%z3=a2*Theta2';% 5000*10;
%a3=sigmoid(z3);% 5000*10;
%
% sigma3=a3-Y; % 5000*10;
% sigma2=(sigma3*Theta2(:,2:end)).*sigmoidGradient(z2);% 5000*25
%
% delta2=zeros(size(Theta2));% 10*26
% delta2=delta2+sigma3'*a2;
%
% delta1=zeros(size(Theta1)); %25*401
% delta1=delta1+sigma2'*a1;
  
end

Theta1_grad=Theta1_grad/m + (lambda/m)*[ zeros(size(Theta1,1),1) Theta1(:,2:end) ];
Theta2_grad=Theta2_grad/m + (lambda/m)*[ zeros(size(Theta2,1),1) Theta2(:,2:end) ];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad and Theta2_grad from Part 2.
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end