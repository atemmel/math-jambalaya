format long;
f = @(x,h)(atan(1+x^2));

x0 = 1/2;
h1 = 0.02;
h2 = 0.01;

df1 = forwarddiff(f, x0, h1) % Framåtdifferens med h1
df2 = forwarddiff(f, x0, h2) % Framåtdifferens med h2

dc1 = centraldiff(f, x0, h1) % Centraldifferens med h1
dc2 = centraldiff(f, x0, h2) % Centraldifferens med h2

ref = richextforward(df1, df2);
rec = richextcentral(dc1, dc2);

diff = diffeval(f, x0)

err_df1 = abs(diff - df1);
err_df2 = abs(diff - df2);


function val = forwarddiff(f, x0, h)
    val = (f(x0 + h) - f(x0)) / h;
end

function val = centraldiff(f, x0, h)
    val = (f(x0 + h) - f(x0 - h)) / (2*h);
end

function val = richextforward(df1, df2)
    val = 2*df2 - df1;
end

function val = richextcentral(dc1, dc2)
    val = (4*dc1-dc2)/3;
end

function val = diffeval(f, x0)
    syms x;
    fprim = diff(f(x));
    x = x0;
    subs(fprim);
    val = eval(fprim);
end