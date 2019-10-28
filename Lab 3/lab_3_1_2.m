format long;

    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

tol = 0.5e-4;
s  = [0, 0, 1];
%s = [   0.416924055521385   0.416495072187980   0.166580872290635];
s0 = s;
s = s * p;
n = 1;

while abs(s - s0) > tol
    s0 = s;
    s = s*p;
    n = n + 1;
end

disp(s)
disp(n)