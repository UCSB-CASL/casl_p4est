% Checking a balltree example in 2D.
points = [   2,   3;	2,   4;		3,   3;		3,   5;		3,   6;		3.5,   5;	4.5,   5; 
		     4,   6;	5, 5.5;		6,   5;		7,   6;		7,   4;		  6,   7;	  5, 7.5; 
			 6, 8.5;  6.5,   8;		7,   9;	  7.5,   8;		9,   7;		 10,   7;  10.5,   6; 
		    11,   5;   11,   7;  11.5,   7;	   12, 7.5;	   11,   8;		 13,   6;    13,   8; 
			14,   7;   14, 7.5;  13.5, 8.5;	   14,   9;	   16,   8];
		
c = [8.227, 6.484848];		% Root: children noted with l for left, r for right.
rad = 7.919026;
c_l = [4.88889, 5.86111];
rad_l = 4.065911;
c_r = [12.233333, 7.233333];
rad_r = 3.843898;
c_l_l = [3.333333, 4.722222];
rad_l_l = 2.178033;
c_l_r = [6.444444, 7];
rad_l_r = 3.051007;
c_l_l_l = [3.833333, 5.416667];
rad_l_l_l = 1.169639;
c_l_l_r = [2.333333, 3.333333];
rad_l_l_r = 0.745356;
c_l_l_l_l = [3.166666, 5.333333];
rad_l_l_l_l = 0.687184;
c_l_l_l_r = [4.5, 5.5];
rad_l_l_l_r = 0.707107;
c_l_r_l = [6.333333, 8];
rad_l_r_l = 1.424001;
c_l_r_r = [6.666667, 5];
rad_l_r_r = 1.054093;
c_l_r_l_l = [7, 8.333333];
rad_l_r_l_l = 0.666667;
c_l_r_l_r = [5.666667, 7.666667];
rad_l_r_l_r = 0.897527;

c_r_l = [10.75, 6.8125];
rad_r_l = 1.829660;
c_r_r = [13.928571, 7.714286];
rad_r_r = 2.091040;
c_r_l_l = [11.3, 6.9];
rad_r_l_l = 1.923538;
c_r_l_r = [9.833333, 6.666667];
rad_r_l_r = 0.942809;
c_r_r_l = [15, 8.5];
rad_r_r_l = 1.118034;
c_r_r_r = [13.5, 7.4];
rad_r_r_r = 1.486607;

figure; hold on;
t = linspace(0, 2*pi, 100);
plot( points(:,1), points(:,2), ".", "markerfacecolor", "b" );
plot( c(1) + rad*cos(t), c(2) + rad*sin(t), "-" );
plot( c_l(1) + rad_l*cos(t), c_l(2) + rad_l*sin(t), "-" );
plot( c_r(1) + rad_r*cos(t), c_r(2) + rad_r*sin(t), "-" );
plot( c_l_l(1) + rad_l_l*cos(t), c_l_l(2) + rad_l_l*sin(t) );
plot( c_l_r(1) + rad_l_r*cos(t), c_l_r(2) + rad_l_r*sin(t) );
plot( c_l_l_l(1) + rad_l_l_l*cos(t), c_l_l_l(2) + rad_l_l_l*sin(t) );
plot( c_l_l_r(1) + rad_l_l_r*cos(t), c_l_l_r(2) + rad_l_l_r*sin(t) );
plot( c_l_l_l_l(1) + rad_l_l_l_l*cos(t), c_l_l_l_l(2) + rad_l_l_l_l*sin(t), "-" );
plot( c_l_l_l_r(1) + rad_l_l_l_r*cos(t), c_l_l_l_r(2) + rad_l_l_l_r*sin(t), "-" );
plot( c_l_r_l(1) + rad_l_r_l*cos(t), c_l_r_l(2) + rad_l_r_l*sin(t), "-" );
plot( c_l_r_r(1) + rad_l_r_r*cos(t), c_l_r_r(2) + rad_l_r_r*sin(t), "-" );
plot( c_l_r_l_l(1) + rad_l_r_l_l*cos(t), c_l_r_l_l(2) + rad_l_r_l_l*sin(t), "-" );
plot( c_l_r_l_r(1) + rad_l_r_l_r*cos(t), c_l_r_l_r(2) + rad_l_r_l_r*sin(t), "-" );

plot( c_r_l(1) + rad_r_l*cos(t), c_r_l(2) + rad_r_l*sin(t), "-" );
plot( c_r_r(1) + rad_r_r*cos(t), c_r_r(2) + rad_r_r*sin(t), "-" );
plot( c_r_l_l(1) + rad_r_l_l*cos(t), c_r_l_l(2) + rad_r_l_l*sin(t), "-" );
plot( c_r_l_r(1) + rad_r_l_r*cos(t), c_r_l_r(2) + rad_r_l_r*sin(t), "-" );
plot( c_r_r_l(1) + rad_r_r_l*cos(t), c_r_r_l(2) + rad_r_r_l*sin(t), "-" );
plot( c_r_r_r(1) + rad_r_r_r*cos(t), c_r_r_r(2) + rad_r_r_r*sin(t), "-" );

xticks( linspace(0, 17, 18) );
xlim( [0, 17] );
yticks( linspace(-2, 15, 18) );
ylim( [-2, 15] );
xlabel( "x" );
ylabel( "y" );
grid on;
axis equal;
legend( "Data", "Root node", "L", "R", "LL", "LR", "LLL", "LLR", "LLLL", "LLLR", "LRL", "LRR", "LRLL", "LRLR", ...
		"RL", "RR", "RLL", "RLR", "RRL", "RRR" );