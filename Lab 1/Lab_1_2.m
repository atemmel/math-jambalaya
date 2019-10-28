format long
syms x

f = @(x)((x-5)^2.5+2*cos(pi*sqrt(x)))/(sqrt(x+4*log(x-pi))-1) ; % anger f(x) 
f_prim = diff(f(x));
x = 7;

f_prim_x_0 = subs(f_prim)
result = eval(f_prim_x_0)