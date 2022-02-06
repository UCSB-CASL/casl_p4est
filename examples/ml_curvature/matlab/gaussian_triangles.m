% Visualizing triangles in a paraboloid.
C = readmatrix( "../cmake-build-debug-3d/gaussian_triangles.csv", "Range", 2 );

figure;
hold on;
for i = 1:length(C)
	plot3([C(i,1), C(i,4), C(i,7), C(i,1)], ...		% Closed triangle.
		  [C(i,2), C(i,5), C(i,8), C(i,2)], ...
		  [C(i,3), C(i,6), C(i,9), C(i,3)], "-");
end
hold off;
axis equal;
xlabel("x"); ylabel("y"); zlabel("Q(x,y)");
rotate3d on;
grid on;