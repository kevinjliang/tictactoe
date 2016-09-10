%% Tic-tac-toe Results
%  10 trials, 25000 games, 250 games/update
%  (3,3) CNNs, ((8,16),(16,16),(16,16)) nFilters, 30 fully connected, 9 out
%  Default ADAM optimizer, epsilon = 0, gamma = 0.95
%  AIX: 0.5, AIO: 0.6

avgRecords_small = [4 3 4 3 5 5 5 5 5 5 6 6 5 7 5 7 8 9 10 11 15 14 15 18 19 17 17 16 16 22 19 24 26 27 25 26 29 33 33 33 29 32 32 31 30 36 39 42 43 47 45 43 41 45 44 44 45 49 51 51 55 57 55 54 57 58 52 56 57 55 57 63 60 61 57 65 61 57 59 64 67 68 68 73 66 75 71 74 72 80 74 75 74 73 74 77 77 77 77 81;
2 4 4 3 5 4 4 4 4 6 6 7 8 10 13 13 16 17 20 23 29 35 40 39 46 55 61 67 71 72 81 80 79 83 84 84 83 89 89 89 94 97 97 98 101 95 85 83 84 85 88 94 96 94 96 93 97 91 82 87 89 86 92 91 91 90 93 96 91 94 100 90 91 96 101 97 103 99 102 95 98 98 99 91 101 97 94 92 92 90 98 95 99 101 104 103 103 100 99 96;
34 32 32 32 33 33 35 36 38 42 37 45 45 44 51 53 60 62 68 74 74 79 84 88 89 93 97 95 100 100 100 96 96 102 104 105 110 102 101 101 102 98 99 100 99 98 106 107 104 100 99 94 94 96 91 97 92 94 98 95 90 92 89 92 89 87 90 86 89 88 82 84 86 80 81 79 76 82 80 82 77 76 75 76 75 72 77 76 76 72 71 72 70 67 65 63 65 68 68 68;
208 209 208 210 206 206 204 203 201 195 199 190 190 187 179 175 165 160 149 141 130 120 109 103 93 84 73 70 61 55 48 48 47 37 35 34 26 25 25 25 22 21 20 19 19 19 18 16 16 16 16 17 18 14 18 14 14 14 17 15 14 14 12 11 11 12 13 9 10 11 10 12 11 11 9 8 9 10 8 7 7 7 6 8 6 5 7 6 8 6 5 7 5 7 5 5 3 3 4 3]./250;

figure(4), clf, hold on
gamesPlayed = 0:250:24750;
plot(gamesPlayed,avgRecords_small(1,:),'LineWidth',4)
plot(gamesPlayed,avgRecords_small(2,:),'LineWidth',4)
plot(gamesPlayed,avgRecords_small(3,:),'LineWidth',4)
plot(gamesPlayed,avgRecords_small(4,:),'LineWidth',4)
xlabel('Games Played')
ylabel('Game % (out of 250)')
title('DeepAI playing Tic-tac-toe (averaged over 10 trials)')
legend('Wins','Draws','Losses','Broken')
axis([0,25000,0,1])