% A naive implementation of Householder's 1958 algorithm for nulling
% all but one vector entry
% Ref: Householder, Alston S. "Unitary triangularization of a nonsymmetric 
% matrix." Journal of the ACM (JACM) 5.4 (1958): 339-342.
%
% Reza Sameni 3/1/2019
% email: reza.sameni@gmail.com

clc;
clear;
close all;

% an arbitrary square matrix:

% A = [6 5 1; 5 22 3 ; 1 3 7]
% A = [9 5 1; 5 8 3 ; 1 3 7]
% N = 3;

A = [6 5 -1 4; 5 22 3 2; -1 3 37 5 ; 4 2 5 10];
N = 4;

col = 2;

I = eye(N);
v = I(:, col);
a = A(:, col);
alpha = sqrt(a'*a); % the first square rute
mu = sqrt(2*alpha*(alpha - v'*a)); % the second square rute
u = (a - alpha*v)/mu; % the reciprocation
H = (I - 2*(u*u'));
d = det(H);
B = H'*A;

d
A
B

