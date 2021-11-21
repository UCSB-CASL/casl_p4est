% Visualizing triangles from C++ discretized Monge patch.
C = readmatrix( "../cmake-build-debug-2d/triangles.csv", "Range", 2 );

figure;
hold on;
for i = 1:length(C)
	plot3([C(i,1), C(i,4), C(i,7), C(i,1)], ...		% Closed triangle.
		  [C(i,2), C(i,5), C(i,8), C(i,2)], ...
		  [C(i,3), C(i,6), C(i,9), C(i,3)], "-");
end

% The normal vectors.
n4 = [0.25, -0.25, 0.25];	% Normal vector in IV quadrant.
p4 = [0.25, -0.25, 0.5];	% Plane in IV quadrant goes through this point.

% Query points.
q0 = [0.4, -0.2, 0.6];
plot3(q0(1), q0(2), q0(3), "b.");
r0 = q0 - dot(q0 - p4, n4) * n4 / dot(n4, n4);
plot3(r0(1), r0(2), r0(3), "b*");
plot3([r0(1),q0(1)], [r0(2),q0(2)], [r0(3),q0(3)], "b-" );

q1 = [-0.2, -0.4, 0.8];
plot3(q1(1), q1(2), q1(3), "m.");
r1 = [-2/30, -8/30, 2/3];
plot3(r1(1), r1(2), r1(3), "m*");
plot3([r1(1),q1(1)], [r1(2),q1(2)], [r1(3),q1(3)], "m-" );

q2 = [0.5, 0.5, 1];
plot3(q2(1), q2(2), q2(3), "r.");
r2 = [5/30, 5/30, 2/3];
plot3(r2(1), r2(2), r2(3), "r*");
plot3([r2(1),q2(1)], [r2(2),q2(2)], [r2(3),q2(3)], "r-" );

q3 = [-0.125, 0.125, 1];
plot3(q3(1), q3(2), q3(3), "g.");
r3 = [-125/3000, 125/3000, 275/300];
plot3(r3(1), r3(2), r3(3), "g*");
plot3([r3(1),q3(1)], [r3(2),q3(2)], [r3(3),q3(3)], "g-" );

axis equal; grid on;
xlabel("x"); ylabel("y"); zlabel("z");
xlim([-1,1]); ylim([-1,1]); zlim([-0.5,1.5]);
hold off;
rotate3d on;