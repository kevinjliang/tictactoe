%% Script to generate X and O shapes in all 9 locations for tic-tac-toe
%  Kevin Liang

%% Generate X shape
X = zeros(20);

for i = 1:20
    for j = 1:20
        if abs(i-j) < 2
            if(i<2  || i>19 || j<2 || j>19)
                continue
            end
            X(i,j) = 1;
        end
        if abs(21-(i+j)) < 2
            if(i<2  || i>19 || j<2 || j>19)
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
        full = [zeros(20,22*(i-1)), X, zeros(20,22*(3-i))];
        full = [zeros(22*(j-1),64); full; zeros(22*(3-j),64)];
        eval(sprintf('dlmwrite(''X%d.txt'',full,''delimiter'','' '')',(j-1)*3+i));
        eval(sprintf('X%d = full;',(j-1)*3+i));
    end     
end

%% Generate O shape
O = zeros(20);

for i = 1:20
    for j = 1:20
        r = (i-10.5)^2 + (j-10.5)^2;
        if r < 8.55^2 && r>6.55^2
            O(i,j) = 1;
        end
    end
end

figure(2), imagesc(O), axis square

% Place O in each of the 9 locations and save a corresponding txt file of
% values
for i=1:3
    for j=1:3
        full = [zeros(20,22*(i-1)), O, zeros(20,22*(3-i))];
        full = [zeros(22*(j-1),64); full; zeros(22*(3-j),64)];
        eval(sprintf('dlmwrite(''O%d.txt'',full,''delimiter'','' '')',(j-1)*3+i));
        eval(sprintf('O%d = full;',(j-1)*3+i));
    end     
end