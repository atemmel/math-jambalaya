format long;

    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

s = [1, 0, 0];
n = 0;
while s ~= s*p
    s = s*p;
    n = n + 1;
end

disp(n)

s = [0, 1, 0];

n = 0;
while s ~= s*p
    s = s*p;
    n = n + 1;
end

disp(n)

s = [0, 0, 1];

n = 0;
while s ~= s*p
    s = s*p;
    n = n + 1;
end

disp(n)