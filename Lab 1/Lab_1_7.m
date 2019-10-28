format long

F = @(x)(-(cos(x)+(x/5)));
F_prim = @(x)(sin(x) - 1/5);

tol = 0.5e-5;

x = 3;
d = F(x) / F_prim(x);
n = 0;

while abs(d) >= tol
    d = F(x) / F_prim(x);
    xnext = x - d;
    xnextnext = xnext - (xnext - x)*F(xnext) / (F(xnext) - F(x));
    x = xnextnext;
    n = n + 1;
end

fprintf("n iterations: %d\n", n)
fprintf("x approx: %f\n", x)