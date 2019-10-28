% a)
%{
n = [1e3, 1e4, 1e5];
f = @(n)floor(1 + 6*rand(1, n));
y_1 = f(n(1));
y_2 = f(n(2));
y_3 = f(n(3));

mean(y_1)
mean(y_2)
mean(y_3)
%}

% b)
%tarning_upprepa(10, [1e3, 1e4, 1e5]);

% c)
%tarning_upprepa(200, 1e5);

% d)
%tarning_upprepa(200, 1e5);

% e)

p = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]';
F = cumsum(p);
F_inv = 1-F;
n = 10000;
u = zeros(1, n);
for i = 1:n
    u(i) = rand();
end
s = sum(u < F_inv) + 1;
hist(s);



