%% Load the file and collect columns.
clear; close all; clc;

nonSaddleD = readmatrix("../cmake-build-release-3d/6/non_saddle_sinusoid.csv", "Range", "DE2");
poshk=nonSaddleD(:,1); posihk=nonSaddleD(:,2); posh2kg=nonSaddleD(:,3); posih2kg=nonSaddleD(:,4);
clear nonSaddleD;

saddleD = readmatrix("../cmake-build-release-3d/6/saddle_sinusoid.csv", "Range", "DE2");
neghk=saddleD(:,1); negihk=saddleD(:,2); negh2kg=saddleD(:,3); negih2kg=saddleD(:,4);
clear nonSaddleD;

%% Showing samples with ih2kg>0 and ih2kg<=0.

f1 = figure;
plot3(neghk, negih2kg, abs(neghk - negihk),"."); 
xlabel("hk"); ylabel("ih2kg"); zlabel("|hk-ihk|");
grid on;
hold on;
flippedsign = neghk.*negihk<0;
plot3(neghk(flippedsign), negih2kg(flippedsign), abs(neghk(flippedsign)-negihk(flippedsign)),"o");
title("Negative ih2kg samples");
axis square;

f2 = figure;
plot3(poshk, posih2kg, abs(poshk - posihk),"."); 
xlabel("hk"); ylabel("ih2kg"); zlabel("|hk-ihk|");
grid on;
hold on;
flippedsign = poshk.*posihk<0;
plot3(poshk(flippedsign), posih2kg(flippedsign), abs(poshk(flippedsign)-posihk(flippedsign)),"o");
title("Positive ih2kg samples");
axis square;