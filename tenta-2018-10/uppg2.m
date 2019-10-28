format long;

f = @(x)exp(x)-2*x.^2;

% Deluppgift A
x = linspace(1,3);
y = f(x);
plot(x, y);

% Deluppgift B
intervallhalvering(f, 2, 3, 0.01)

% Deluppgift C
newtonraphson(f, 1, 0.5e-7);
newtonraphson(f, 3, 0.5e-7);

% Deluppgift D

% Ifall vi tänker igenom algoritmen för halveringsmetoden 
% så håller den på så länge 𝑓(𝑎) ∙ 𝑓(𝑏) < 0, vilket betyder 
% att vi vill kolla 𝑓(1) ∙ 𝑓(3), vilket är större än noll 
% som innebär att det finns antingen ingen eller ett jämt 
% antal lösningar på problemet. Ifall vi skulle kringgå 
% detta så skulle man aldrig få ett exakt svar med 
% halveringsmetoden, då det naturliga talet har ett oändligt 
% antal decimaler och svaret då antagligen också skulle 
% behöva det. I Matlab skulle man kanske kunna få 14 decimalers
% precision, men det är inget jämfört med totalen. Däremot 
% skulle det ju gå att lösa ekvationen på ett annat sätt, 
% men intervallhalveringsmetoden är inte den rekommenderade 
% metoden för ett exakt svar.

% Deluppgift B
function val = intervallhalvering(f, a, b, delta)
    m = (a + b) / 2;
    while(abs(a - b) > delta)
        if(f(a) * f(m) < 0)
            b = m;
        else
            a = m;
        end
        m = (a + b) / 2;
    end
    val = m;
end

% Deluppgift C
function [xn, n] = newtonraphson(f, x0, tol)
    syms x;
    fprim = diff(f(x) );
    
    xn = x0;
    d = f(xn) / eval(subs(fprim, xn));
    n = 0;

    while (abs(d) >= tol)
        d = f(xn) / eval(subs(fprim, xn));
        xn = xn - d;
        n = n + 1;
    end

    %fprintf("n iterations: %d\n", n)
    %fprintf("x approx: %f\n", xn)
end