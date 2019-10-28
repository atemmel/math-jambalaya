%format long;

p = [
    % 0    1    2    3    4
      0.5, 0.5, 0  , 0  , 0  ; % 0
      0.2, 0.3, 0.5, 0  , 0  ; % 1
      0  , 0.4, 0.1, 0.5, 0  ; % 2
      0  , 0  , 0.4, 0.1, 0.5; % 3
      0  , 0  , 0  , 0.4, 0.6  % 4
];

s = [1, 0, 0, 0, 0];

s = iteratemarkov(p, s);
sum([0 1 2 3 4] .* s)

s
r = montemarkov(p)
sum(r)

%{
while s ~= s*p
    s = s*p;
end
%}

% Deluppgift C
function s = iteratemarkov(p, s) 
    for i = [1:5e3]
        s = s*p;
    end
end

% Deluppgift D
function res = montemarkov(p)
    F = cumsum(p');

    n = 1e6;
    res = zeros(1, size(F,1));
    current = 1;

    for i = 1:n
       res(current) = res(current) + 1;

       rnd = rand;
       for j = 1:size(F,1)
           if rnd < F(j, current)
               current = j;
               break
           end
       end
    end

    res = res./sum(res);
end