    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

n = 1e5;
u = zeros(3, n);
u(1:3, 1) = [rand, rand, rand];
u(1:3, 1) = u(1:3, 1)./sum(u(1:3, 1));
u(1:3, 1) = u(1:3, 1)' * p;
for i = 2:n
    u(1:3, i) = u(1:3, i - 1)' * p;
end

s = sum(u')
s = s./sum(s)
%sum(s)