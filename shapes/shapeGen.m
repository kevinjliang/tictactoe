%% Script to generate X and O shapes in all 9 locations for tic-tac-toe
%  Kevin Liang

%% Generate X shape
X = zeros(39);

for i = 1:39
    for j = 1:39
        if abs(i-j) < 3
            if(i<4  || i>36 || j<4 || j>36)
                continue
            end
            X(i,j) = 1;
        end
        if abs(40-(i+j)) < 3
            if(i<4  || i>36 || j<4 || j>36)
                continue
            end
            X(i,j) = 1;
        end
    end
end

figure(1), imagesc(X), axis square

% Place X in each of the 9 locations and save a corresponding txt file of
% values
for i=1:3
    for j=1:3
        full = [zeros(39,43*(i-1)), X, zeros(39,43*(3-i))];
        full = [zeros(43*(j-1),125); full; zeros(43*(3-j),125)];
        eval(sprintf('dlmwrite(''X%d.txt'',full,''delimiter'','' '')',(j-1)*3+i));
        eval(sprintf('X%d = full;',(j-1)*3+i));
    end     
end

%% Generate O shape
O = zeros(39);

for i = 1:39
    for j = 1:39
        r = (i-20)^2 + (j-20)^2;
        if r < 17.05^2 && r>14.05^2
            O(i,j) = 1;
        end
    end
end

figure(2), imagesc(O), axis square

% Place O in each of the 9 locations and save a corresponding txt file of
% values
for i=1:3
    for j=1:3
        full = [zeros(39,43*(i-1)), O, zeros(39,43*(3-i))];
        full = [zeros(43*(j-1),125); full; zeros(43*(3-j),125)];
        eval(sprintf('dlmwrite(''O%d.txt'',full,''delimiter'','' '')',(j-1)*3+i));
        eval(sprintf('O%d = full;',(j-1)*3+i));
    end     
end