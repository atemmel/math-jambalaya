format long
% a)
%[N, mu, err] = mcconv([-5, 5], 1000000);
%[N, mu, err] = mcconv([-5, 5; -5, 5]);

% b)
% 
[N1, mu1, err1] = mcconv([-5, 5], [1e4, 1e5, 5e5, 1e6]);
a1 = polyfit(log(N1), log(err1) , 1);
[N2, mu2, err2] = mcconv([-5, 5; -5, 5], [1e4, 1e5, 5e5, 1e6]);
a2 = polyfit(log(N2), log(err2) , 1);

% c)
% N = n^d, där n = 5, d = antal dimensioner och N = totala antalet noder
