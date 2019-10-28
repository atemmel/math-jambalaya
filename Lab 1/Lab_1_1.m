format long

f = @(x)((sqrt(x - 5)^5) + 2 * cos(pi * sqrt(x) ) ) / (sqrt(x + 4 * log(x - pi) ) - 1);
h = [ 0.04, 0.02, 0.01 ];
x = 7;
result = zeros(3, 2);

for i = 1:3
    % Framåtdifferens (D1)
    result(i) =            (f(x + h(i)) - f(x) ) / h(i);
    
    % Centraldifferens (D2)
    result(3 + i : end) =  (f(x + h(i)) - f(x - h(i) ) ) / (2 * h(i) );
end

% Grafisk lösning (framåt ligger "före")
fprintf("%5s%20s\n", "D1", "D2")
disp(result)