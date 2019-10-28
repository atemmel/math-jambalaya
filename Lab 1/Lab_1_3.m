format long

f = @(x)((sqrt(x - 5)^5) + 2 * cos(pi * sqrt(x) ) ) / (sqrt(x + 4 * log(x - pi) ) - 1);
x = 7
h = 0.02
k1 = @(x)2*(2*f(x + (h / 2)) - f(x + h) - f(x) ) / (h^2);

(f(x + (h/2)) - f((x)) ) / (h / 2) + (k1(x) * h / 2)