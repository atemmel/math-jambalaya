format long;

f = @(x)log(x);

% Deluppgift A
[val, err] = mcint(f, [1 exp(1)], 10, 1000)


% Deluppgift B
[val, n ] = trapz(f, 1, exp(1), (exp(1)-1) / 10)

% Deluppgift C
res = integral(f, 1, exp(1));

% Deluppgift D
% För högre dimensioner är monte carlo bättre, då dess prestanda ej
% förändras om man lägger till dimensioner. I det här fallet lämpar sig
% trapetsmetoden bättre då dimensionens storlek är måttligt liten.

% Deluppgift A
function [val, err] = mcint(f, D, N, M)
    V = cumprod(D(:,2)-D(:,1));
    dim = numel(V);
    V = V(end);
    
    r = zeros(dim, N);
    
    for j = 1:M
       r(:,:) = repmat(D(:,1),1,N)+rand(dim, N).*repmat(D(:,2)-D(:,1),1,N);
       I(j) = V*mean(f(r));
    end
    
    val = mean(I);
    err = std(I);
end

% Deluppgift B
function [val, n] = trapz(f, a, b, h) 
    val = f(a) / 2 + f(b) / 2;
    n = 1;
    for i = a + h : h : b - h
        val = val + f(i);
        n = n + 1;
    end
    val = val * h;
end