
%L�s med valfri metod i kursen (utom kurvritning).
%{
% a)
%a) En familj skall anl�gga en simbass�ng f�r barnen med vattendjupet x m,
%bredden (5x+1) m och l�ngden (15x+1) m.
%Bass�ngen har formen av ett r�tblock.
%Ber�kna vattendjupet om bass�ngen skall rymma 400 m3 vatten.

% definiera funktionen
f=@(x) 75*x.^3+20*x.^2+x-400;
% definiera funktionens derivata
df=@(x) 225*x.^2+40.*x+1;

% plotta
figure;
x=0:0.01:3;
plot(x,f(x))
grid
% vi ser att rimligt startv�rde kan vara x=1
% k�r Newton-Raphson
x=1; % start punkt
p=f(x);
n_iter=0;
while abs(p)>0.001
    x=x-f(x)/df(x);
    p=f(x);
    n_iter=n_iter+1;
end

% skriv ut svaret
x
%}
%Ber�kna roten ur 3 med valfri metod i kursen.
%{


% Newton-Raphson igen:
f=@(x) x^2-3;
df=@(x) 2*x;
% Newton-Raphson:
x=2; % starting point
p=f(x);
n_iter=0;
while abs(p)>0.001
    x=x-f(x)/df(x);
    p=f(x);
    n_iter=n_iter+1;
end
x
sqrt(3)
%}
%Centralformeln
%{
x=0.5;
h=0.2;
format long

% a)
% centralformeln:
d1=(atan(x+h)-atan(x-h))/(2*h)

% b)
% halvera h
d2=(atan(x+h/2)-atan(x-h/2))/(2*h/2)
% felet i centralformeln �r O(h^2)
% vi har alltsa d~=d1+k*h^2 och d~=d2+k*(h/2)^2,
% d�r d �r exakt v�rde. L�s m.a.p. k:
k=(d1-d2)/(h^2/4-h^2);
% extrapolerat v?rde:
fp_extra = d1+k*h^2
% exakt:
fp_exact=1/(1+x^2)
%}
%Fram�tdifferenser
%{
format long
f = @(x,h)(atan(1+x^2)); %Funktion
x0 = 1/2; %Derivatan n�ra v�rde x = 1/2

h1=0.02; %Stegl�ngd
h2=0.01; %Stegl�ngd

df1 = (f(x0+h1)-f(x0))/(h1) %fram�tdifferens med stegl�ngd h1 = 0.02 utan extrapolation
df2 = (f(x0+h2)-f(x0))/(h2) %fram�tdifferens med stegl�ngd h2 = 0.01 utan extrapolation

REC = ((2*df2)-df1) % Richardsson extrapolation

f_prime = ((2*x0)/((x0^2 + 1)^2 + 1)); %exakta v�rdet vid x0

edc1 = abs(f_prime-df1); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h1
edc2 = abs(f_prime-df2); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h2
eREC = abs(f_prime-REC); %Error hur l�ngt ifr�n man var riktiga v�rdet med Richardsson extrapolation
Resultat = [df1 df2 REC]
Error = [edc1 edc2 eREC]
%}
%Sammansatt Trapetsregel f�r integral
%{
resultat = [];
fx = @(x)(exp(x).*sin(2*x));
[resultat, antal] = trapets(fx,0,(pi/2));
DeltaT = resultat(2) - resultat(1);
DeltaTHalv = resultat(2) -  resultat(1);
%disp(DeltaT / DeltaTHalv);
RombergExtraPol = resultat(2)  + (resultat(2) - resultat(1)) / 3;
Integral = integral(@log,1,exp(1));

%disp(abs((Integral - resultat)));
%disp(abs((Integral - RombergExtraPol)));

CorrectValue = integral(fx,0,pi/2);

abs(CorrectValue - RombergExtraPol)
function [M,q] = trapets(f,a,b)
    %s�tter f�rsta h
    h = (pi/20);
    for i = 1 : 2
        x = a:h:b;
        y = f(x);
        tmp = size(y);
        q(i) = tmp(end);
        M(i) = h*(sum(y)-y(1)/2-y(end)/2);
        h = h/2;
    end
end
%}
%Centraldifferenser
%{
format long
f = @(x,h)(atan(1+x^2)); %Funktion
x0 = 1/2; %Derivatan n�ra v�rde x = 1/2

h1=0.02; %Stegl�ngd
h2=0.01; %Stegl�ngd

dc1 = (f(x0+h1)-f(x0-h1))/(2*h1) %centraldifferens med stegl�ngd h1 = 0.02 utan extrapolation
dc2 = (f(x0+h2)-f(x0-h2))/(2*h2) %centraldifferens med stegl�ngd h2 = 0.01 utan extrapolation

REC = (dc2+(dc2-dc1)/3) % Richardsson extrapolation

f_prime = ((2*x0)/((x0^2 + 1)^2 + 1)); %exakta v�rdet vid x0

edc1 = abs(f_prime-dc1); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h1
edc2 = abs(f_prime-dc2); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h2
eREC = abs(f_prime-REC); %Error hur l�ngt ifr�n man var riktiga v�rdet med Richardsson extrapolation
Resultat = [dc1 dc2 REC]
Error = [edc1 edc2 eREC]
%}
%Fram�t, Central,Bakdiff + Extrapol p� allt 
%{
f = @(x)(atan(2*x+1)); %Funktion
x0 = (sqrt(3)-1)/2; %Derivatan n�ra v�rde x = (sqrt(3)-1)/2
f_deriv = @(x)(1/(2*x^2 + 2*x + 1));%derivatan
DerivExakt = f_deriv(0.5);

h1=0.02; %Stegl�ngd
h2=0.01; %Stegl�ngd

fd = @(x,h)((f(x+h)-f(x))/(h)); %framm�tdifferansen med stegl�ngd h utan extrapolation
cd = @(x,h)((f(x+h)-f(x-h))/(2*h)); %centraldifferens med stegl�ngd h utan extrapolation
bd = @(x,h)((f(x)-f(x-h))/(h)); %bakatdifferens med stegl�ngd h utan extrapolation

x = 0.5; %Derivatan n�ra v�rde x = -ln(2)

FramDiffH1 = fd(x,h1); %approximerar derivatan med hjalp av frammat steglangd h1
CentralDiffH1 = cd(x,h1); %approximerar derivatan med hjalp av central steglangd h1
BakDiffH1 = bd(x,h1); %approximerar derivatan med hjalp av bakat steglangd h1 



FramDiffH2 = fd(x,h2); %approximerar derivatan med hjalp av frammat steglangd h2
CentralDiffH2 = cd(x,h2); %approximerar derivatan med hjalp av central steglangd h2
BakDiffH2 = bd(x,h2); %approximerar derivatan med hjalp av bakat steglangd h1 

%K = GRAD
K = 2;
FramDiffExtrapol = FramDiffH2 + (FramDiffH2 - FramDiffH1) / (2^(K-1)-1); %Extrapol p� Framdiff
BakDiffExtrapol = BakDiffH2 + (BakDiffH2 - BakDiffH1) / (2^(K-1)-1); %Extrapol p� Bakdiff
CentralDiffExtrapol = CentralDiffH2 + (CentralDiffH2 - CentralDiffH1) / (4^(K-1)-1); %Extrapol p� Centraldiff

FramDiff1ErrorH1 = abs(DerivExakt - FramDiffH1); 
FramDiff1ErrorH2 = abs(DerivExakt - FramDiffH2);
CentralDiff1ErrorH1 = abs(DerivExakt - CentralDiffH1);
CentralDiff1ErrorH2 = abs(DerivExakt - CentralDiffH2);
BakDiff1ErrorH1 = abs(DerivExakt - BakDiffH1);
BakDiff1ErrorH2 = abs(DerivExakt - BakDiffH2);

ExtrapolErrorFram = abs(DerivExakt - FramDiffExtrapol);
ExtrapolErrorCent = abs(DerivExakt - CentralDiffExtrapol);
ExtrapolErrorBak = abs(DerivExakt - BakDiffExtrapol);

Framskillnad = FramDiff1ErrorH2 / FramDiff1ErrorH1;
CentralSkilland = CentralDiff1ErrorH2 / CentralDiff1ErrorH1;
BakSkilland = BakDiff1ErrorH2 / BakDiff1ErrorH1;
%}
%TrappetsMetoden
%{
function M = trapets(f,a,b,n)
h =(b-a)/n;
x = a:h:b;
y = f(x);
M = h*(sum(y)-y(1)/2-y(end)/2);
end
%}
%Exakta svaret p� fPrime(1/2)
%{
format rat
f_prime = ((2*(1/2))/(((1/2)^2 + 1)^2 + 1)) %exakta v�rdet vid x0 = 16/41

%}
%Intervallhalveringsmetoden f�r att hitta f(x)=0 i [2,3]
%{
format long
y = @(x) (exp(x)-2*x^2);
a = 2; %start p� intervall
b = 3; %slutet p� intervall
fplot(y,[a,b])
e = 0.01;
tmp = intervallhalvering(a,b,e)

function q = intervallhalvering (a,b,e)
    while (funktionen(a)*funktionen(b) < 0 && b-a >= e)
        c = (a+b)/2;
        if ( funktionen(a)*funktionen(c) < 0)
            b = c;
        else
          a = c;
        end
    end
    q = a
end

function y = funktionen(x)

 y = exp(x)-2*x^2;

end


%}
%Newton Rhapson f�r att hitta f(x)=0 i [1,3]
%{
function [ x ] = NewRap( x0 )
    x = x0 - funktionIUppgiften(x0) / funktionIUppgiften_prim(x0); %Skapar x0
    while abs(x0 - x) >= 10^-7 %stannar n�r differancen �r mindre 10^-7 x0 = x;
    x0 = x;
    x = x0 - funktionIUppgiften(x0) / funktionIUppgiften_prim(x0); %Assign the new x value using Newton Rhapsons method
    end
end

%Funktion f�r att kolla funktionsv�rdet
function y = funktionIUppgiften( x )
%Returns the y-value of the inputted x-value
y = exp(x) - 2*x^2;
end

function y = funktionIUppgiften_prim( x )
%Returns the y-value of the inputted x-value
y = exp(x) - 4*x;

end
%}
%Monte Carlo f�r att hitta l�sning till integral med gr�nser
%{
fx = @(x)log(x);
Resultat = mcint(fx,[1 exp(1)], 10, 1000)

function [val,err] = mcint(f,D,N,M)
%MCINT. Function that solves an integral using MonteCarlo
    V = cumprod(D(:,2)-D(:,1));
    dim = numel(V);
    V = V(end);
    
    r= zeros(dim,N);
    
    for j = 1:M
        r(:,:) = repmat(D(:,1),1,N) + rand(dim,N).*repmat(D(:,2)-D(:,1),1,N);
        I(j) = V*mean(f(r));
    end
    
    val = mean(I);
    err = std(I);

end


%}
%Trapetsregeln med stegl�ng h functionen f.
%{
reslutat = trapets(@log,1,exp(1),10)
function M = trapets(f,a,b,n)
%h =(b-a)/n;
h = (exp(1)-1)/10;
x = a:h:b;
y = f(x);
M = h*(sum(y)-y(1)/2-y(end)/2);
end
%}
% L�t Matlab L�sa integralen
%{
func = @(x) log(x)
value=integral(func,1,exp(1))

%}
%Best�m samtliga l�sningar till ekavationen f
%{
f = @(x)(x.^2-5.*x-5-cos(x)); %funktionen f
fp = @(x)(2.*x-5+sin(x)); %derivatan av f

x = -2; %startgissning f�r x

tol = 1e-7; %noggrannheten i l�sningen

n = 0; %antal iterationer som initialt ?r 0

while abs(f(x)/fp(x))>tol
     
      x = x-f(x)/fp(x); %nytt x-v�rde
      n = n+1; %�kar antal iterationer
      
end

x %l�sningen till ekvationen, d.v.s. f(x) ~ 0
n %antal iterationer 


%}
%Hitta maximum till funktionen med Monte Carlo KOLLA P� DENNA INTE KLAR
%{
x = -3:0.1:3;
y = -3:0.1:3;
f = zeros(length(y),length(x));

for i=1:length(x)
    for j=1:length(y)
        f(i,j) = -(x(i).^2+y(j).^2+2*x(i)+3*y(j)+(sin(x(i)+y(j))).^2 + 2);
    end
end

figure; 
surf(x,y,f); 
axis tight

%F�rs�ker hitta minimum:
xy_max = 3; %x,y-max
xy_min = -3; %x,y-min

N = 100; %antal slumptal

f_max = -1/eps; %startv�rde f�r f_max (litet)
x_max = 0; %startv�rde f�r x_max
y_max = 0; %startv�rde f�r y_max

for i=1:N
    x = xy_min+(xy_max-xy_min)*rand(1,1);
    y = xy_min+(xy_max-xy_min)*rand(1,1);
    f = -(x^2+y^2+2*x+3*y+(sin(x+y))^2 + 2);
    if f>f_max
        x_max = x;
        y_max = y;
        f_max = f;
    end
end
hold on;

plot3(x_max,y_max,f_max,'r*','markersize',15)
hold off

f_max
x_max
y_max
%}
%{
function [coef] = PolyReg(x,y,order)

    matrix = zeros(order+1,order+1);
    right = zeros(order+1,1);
    powers = zeros(2*order+1,1);
    
    for i = 0:(2*order)
        powers(i+1) = sum(x.^i); 
    end
    
    for i = 0:order
        right(i+1) = sum(y.*(x.^i)); 
    end
    
    for i = 0:order
        matrix(i+1,:) = powers((i+1):(i+1+order)); 
    end
    
    coef = flip(matrix\right);

end

%}
%SekantMetod
%{
%SekantMetoden anv�nder inte derv, som ibland kan vara sv�r att hitta
fun = @(x)exp(x)-x^2;
x1 = -10;
x2 = 0;
f1 = fun(x1);
dx = 1;
iterSek = 0;

while abs(dx) > 1e-10 && iterSek < 10
  f2 = fun(x2);
  dx = - f2*(x2 - x1)/(f2 - f1);
  x1 = x2;
  f1 = f2;
  x2 = x2 + dx;
  iterSek = iterSek + 1;
end
disp(x1);
iterSek
%}
%Ber�kna omr�dets volym med Monte Carlo INTE KLAR
N = 1000:1000:50000; %olika antal slumptal

for n=1:length(N)
    
    N_in = 0; %antal punkter innanf�r omr�det (initialt)
    
    for i=1:N(n)
        x = rand(1,1); %slumpat v�rde f�r x
        y = rand(1,1); %slumpat v�rde f�r y
        z = rand(1,1); %slumpat v�rde f�r z
        
        if (x^2+sin(y))<=z && (x+exp(y)-z)<=1 %unders�ker om punkten (x,y,z) �r innanf�r
            N_in = N_in+1; %�kar antal v�rden innanf�r
        end
    end
    V_bes(n) = N_in/N(n); %alla v�rden p� volymerna i en array    
    
end

figure; 

plot(N, V_bes,'.-'); 

xlabel('N')
ylabel('V?rden f?r volymerna')


%Vi s�ger att sista v�rdet f�r V_bes ?r en bra approximation att anv�nda d�
%vi ska kolla p� felet:

err_V = abs(V_bes(1:end-1)-V_bes(end)); %absoluta felet


plot(N(1:end-1), err_V,'.-'); 

xlabel('N')
ylabel('error V�rden f�r volymerna')

plot(log(N(1:end-1)), log(err_V),'.-'); 

xlabel('N')
ylabel('V�rden f�r volymerna')

%Hur felet beror av N kan man se fr�n en r�t linje

%Uppgift 5 20171001 
%{
%lambda = 0.008
% mu = 0.01
lambda = 0.08;
mu = 0.01;

%{
P = [1-lambda Lambda 0;
    mu 1-lambda-mu lambda;
    0 mu 1-mu];
%}

P = [0.992 0.008 0;
    0.01 0.982 0.008;
    0 0.01 0.99]; %Array med sannolikheter
s = [1 0 0];
n = 0;
test = true;
if s == s*P
    test = false;
end
while test
    if s == s*P
        test = false;
    end
    s = s*P;
    n = n+1;
end
M = sum ([1 2 3] .* s);
mean(M)% Viktat medelv�rde - medelantal kunder

monteCarloPaMarkovKedja(P)

function values = monteCarloPaMarkovKedja(P)

    F = cumsum(P'); % Delar platsernas sannolikhet att besk�kas j�mnt.
    N = 1e6;
    s = zeros(1, size(F,1));
    CURPOS = 1;

    for i  = [1:N]
        s(CURPOS) = s(CURPOS) + 1;
        DICE = rand; 
        for j = [1:size(F,1)]
            CURPOS;
            if DICE < F(j,CURPOS)
                CURPOS = j;
                break
            end
        end

    end

values = s./sum(s);
end
%}
%Uppgift 5 181003
%{
%Lambda = 0.5
%Mu = 0.2
%{
P = [Lambda Lambda 0 0 0;
    Mu 1-Lambda-mu lambda 0 0;
    0 2*lambda 1-2*mu-lambda lambda 0;
    0 0 2*lambda 1-2*mu-lambda lambda;
    0 0 0 2*mu 1-2*mu]; 
%}
P = [0.5 0.5 0 0 0;
    0.2 0.3 0.5 0 0;
    0 0.4 0.1 0.5 0;
    0 0 0.4 0.1 0.5;
    0 0 0 0.4 0.6]; %Array med sannolikheter
s0 = 1; %startv�rde
T = [0.0000001 0.0000001 0.0000001 0.0000001 0.0000001]; %Tolerans
rest = 80;
i=0;
s = [1.0 0.0 0.0 0.0 0.0];
n = 0;
test = true;
if s == s*P
    test = false;
end
while test
    if s == s*P
        test = false;
    end
    s = s*P;
    n = n+1;
end
M = sum ([1 2 3 4 5] .* s);
mean(M)% Viktat medelv�rde - medelantal kunder

monteCarloPaMarkovKedja(P)
function values = monteCarloPaMarkovKedja(P)

    F = cumsum(P'); % Delar platsernas sannolikhet att besk�kas j�mnt.
    N = 1e9;
    s = zeros(1, size(F,1));
    CURPOS = 1;

    for i  = [1:N]
        s(CURPOS) = s(CURPOS) + 1;
        DICE = rand; 
        for j = [1:size(F,1)]
            CURPOS;
            if DICE < F(j,CURPOS)
                CURPOS = j;
                break
            end
        end

    end

values = s./sum(s);
end
%}

