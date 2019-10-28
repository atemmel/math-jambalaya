F = @(x)(-(cos(x)+(x/5)))
F_prim = @(x)(sin(x) - 1/5);

x = [-2*pi:0.1:2*pi]
y = F(x);
y_prim = F_prim(x);
plot(x, y);
hold
plot(x, y_prim);
grid
shg