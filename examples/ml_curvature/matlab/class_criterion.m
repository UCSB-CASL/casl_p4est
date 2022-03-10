%% Load the file and collect columns.
clear; close all; clc;
D = readmatrix("../cmake-build-release-3d/6/sinusoid.csv", "Range", "DE2");
hk=D(:,1); ihk=D(:,2); h2kg=D(:,3); ih2kg=D(:,4);
clear D;

%% Showing samples with ih2kg>0 and ih2kg<=0.
allpos = ih2kg > 0;

poshk = hk(allpos);
posihk = ihk(allpos);
posh2kg = h2kg(allpos);
posih2kg = ih2kg(allpos);

neghk = hk(~allpos);
negihk = ihk(~allpos);
negh2kg = h2kg(~allpos);
negih2kg = ih2kg(~allpos);

f1 = figure(1);
plot3(neghk, negih2kg, abs(neghk - negihk),"."); 
xlabel("hk"); ylabel("ih2kg"); zlabel("|hk-ihk|");
grid on;
hold on;
flippedsign = neghk.*negihk<0;
plot3(neghk(flippedsign), negih2kg(flippedsign), abs(neghk(flippedsign)-negihk(flippedsign)),"o");
title("Negative ih2kg samples");

f2 = figure(2);
plot3(poshk, posih2kg, abs(poshk - posihk),"."); 
xlabel("hk"); ylabel("ih2kg"); zlabel("|hk-ihk|");
grid on;
hold on;
flippedsign = poshk.*posihk<0;
plot3(poshk(flippedsign), posih2kg(flippedsign), abs(poshk(flippedsign)-posihk(flippedsign)),"o");
title("Positive ih2kg samples");