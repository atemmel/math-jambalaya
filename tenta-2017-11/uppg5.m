format long
% lambda = 0.0008
% mu = 0.01

%{
p = [0.992, 0.008, 0;
     0.01, 0.982, 0.008;
     0, 0.01, 0.99];
%}
%s = [1,0,0];

p = [
    % 0    1    2    3    4
      0.5, 0.5, 0  , 0  , 0  ; % 0
      0.2, 0.3, 0.5, 0  , 0  ; % 1
      0  , 0.4, 0.1, 0.5, 0  ; % 2
      0  , 0  , 0.4, 0.1, 0.5; % 3
      0  , 0  , 0  , 0.4, 0.6  % 4
];


s = iteratemarkov(p, s)

sum(s)

function s = iteratemarkov(p, s) 
    test = true;
    n = 0;
    if s == s*p
        test = false;
    end
    while test
        if s == s*p
            test = false;
        end
        s = s*p;
        n = n + 1;
    end
    disp(n);
end