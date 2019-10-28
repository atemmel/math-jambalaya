format long
syms x

f = @(x)((sqrt(x - 5)^5) + 2 * cos(pi * sqrt(x) ) ) / (sqrt(x + 4 * log(x - pi) ) - 1);
h = [ 0.04, 0.02, 0.01 ];

result1 = zeros(3,1);
result2 = 0;

f_prim = diff(f(x));
f_biss = diff(f_prim);

x = 7;

for i = 1:3 
    % D3
   result1(i) = (f(x - h(i)) - 2 * f(x)  + f(x + h(i))) / h(i)^2;
end

f_biss_x_0 = subs(f_biss);
result2 = eval(f_biss_x_0);

fprintf("%5s\n", "D3")
disp(result1)

fprintf("%13s\n", "f_biss_x_0")
disp(result2)