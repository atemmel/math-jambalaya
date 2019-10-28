n = 365;
s = zeros(n, 1);
for i = [1:n]
    b = blackjacksim(100);
    s(i) = b(end);
end
plot(s)
sum(s)