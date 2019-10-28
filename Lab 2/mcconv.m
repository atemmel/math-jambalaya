function [N, mu, err] = mcconv(D, N)
% MCCONV. Function solving a multidimensional integral using 
%         Monte Carlo. The function calls the function mcint. 
% Calling sequence: [N, val, err] = mcconv(D, N);
% Input: D -  The domain. Each row in the matrix D represents the range 
%             of the corresponding variable. For example, a 3-dimensional
%             integral solved on the unit cube is given by             
%             D = [0 1; 0 1; 0 1];  
%             Default is a 6-dimensional integral in the range -5 to 5 in 
%             each dimension
%        N -  A vector with the numbers of random numbers. For example, if 
%             N = [100; 1000; 2000] three integral values will be computed,
%             based on 100, 1000 and 2000 random numbers respectively.
%             The default is N = [100; 500; 1000; 5000; 10000; 1e5];
% Output: val - the resulting integral value, a vector where val(i) 
%               is the integral computed in N(i) random numbers 
%         err - the error in the result (the standard deviation)
%         N   - The number of random numbers used (same as the input value
%               N)
%    

close;
% Number of replications to base the error estimation on
M = 50;

% Default number of pseudorandom numbers
if (nargin < 2)
   N = [100; 500; 1000; 5000; 10000; 1e5];
end

% Default 6D and range [-5 5] in each dimension
if (nargin == 0) | (isempty(D) )
    D = [-5 5;-5 5;-5 5;-5 5;-5 5]; % 6D
end

N = N(:);
% The dimension and range
[rows, columns] = size(D);
if columns ~= 2
    error('There must be 2 numbers (lower and upper limit) in each row in D');
end



dim = numel(D)/2;
disp(['A ',num2str(dim),'-dimensional integral is solved'])
disp(['for random numbers: ' num2str(N')]);

if dim == 1
    fplot(@fnorm, [D(1) D(2)]);
end
if dim == 2
    NN = 100;
    x = linspace(D(1,1), D(1,2), NN);
    y = linspace(D(2,1), D(2,2), NN);
    [X,Y] = meshgrid(x,y);
    Z = fnorm([X(:)';Y(:)']);;
    Z = reshape(Z,NN,NN);
    surf(X,Y,Z);
    shading interp; 
end
drawnow;
j=1;
mu = zeros(length(N),1);
err = mu;
for i=1:length(N)
    [mu(j),err(j)]=mcint(@fnorm, D, N(i), M);
    j=j+1;
end
end


% -----------------------------------------
% Interna funktioner

function y = fnorm(x)
% FNORM. function defining the density of the d-dim standard normal 
% distribution.
% Calling sequence: y = fnorm(x)
% Input: x - (d,N) matrix. Each column of x is a point   
%            where the function is evaluated.  


   [d,N]=size(x);  
   C = eye(d);
   y = exp(-sum(x.*(C*x),1));
end
   

function [val,err] = mcint(f,D,N,M) 
% MCINT.  Function that solves an integral using Monte Carlo.
%
% Calling sequence: [val,err] = mcint(@fun,D,N,M)    
% Input:  fun - function defining the integrand. 
%               fun will be evaluated in dim*N points, where dim is the
%               dimension of the problem. 
%         D   - the domain. Each row in the matrix D represents the range 
%               of the corresponding variable. For example, a 3-dimensional
%               integral solved on the unit cube would be given by             
%               D = [0 1; 0 1; 0 1];
%         N   - the number of points in each realization.
%         M   - the number of repetitions used for error estimation. 
%               (Recommendation, M = 30+).
%               Total number of points used is thus M*N.  
%
% Output:  val - the resulting integral value
%          err - the error in the result (the standard deviation)

    V   = cumprod(D(:,2)-D(:,1));
    dim = numel(V);
    V   = V(end);
    
    r=zeros(dim,N);
    
    for j=1:M
        r(:,:) = repmat(D(:,1),1,N)+rand(dim,N).*repmat(D(:,2)-D(:,1),1,N);
        I(j)=V*mean(f(r));
    end
    
    val = mean(I);
    err = std(I);
    
end  
   