    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

F = cumsum(p');

n = 1e6;
res = [0, 0, 0]';
bull = 1;
bear = 2;
rec  = 3;
current = 1;

for i = [1:n]
   res(current) = res(current) + 1;
    
   rnd = rand;
   if rnd < F(1, current)
       current = bull;
   elseif rnd < F(2, current)
       current = bear;
   else 
       current = rec;
   end
end

res = res./sum(res)