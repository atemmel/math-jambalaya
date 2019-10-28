    %Bull Bear Recession
p = [0.8,  0.15, 0.05;
     0.1,  0.75, 0.15;
     0.25, 0.25, 0.5
];

i = [1, 0, 0;
     0, 1, 0;
     0, 0, 1
];

rhs = null(p' - i); %Nullvektorn till p
Tt = rhs./sum(rhs)  %Normalisera nullvektorn