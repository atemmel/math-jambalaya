    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

F = cumsum(p');

n = 1e6;
res = zeros(1, size(F,1));
current = 1;

for i = [1:n]
   res(current) = res(current) + 1;
    
   rnd = rand;
   for j = [1:size(F,1)]
       if rnd < F(j, current)
           current = j;
           break
       end
   end
end

res = res./sum(res)