%%%% Reflections about planes in the 2x2x2 cube centered at the origin,
%%%% which is what we want to use for augmentation in 3D curvature
%%%% computation.
clear; clc;

% Let's generate the coordinates for the 27 points, in the range [-1,1]^3.
baseColors = ["#0072BD", "#EDB120", "#A2142F"];
coords = zeros(3,27);	% One point per column.
colors = strings(1,27);	% Each point gets a color depending on its x level.
idx = 1;
for x = -1:1:1			% x-coord changes the least.
	color = baseColors(x+2);	% Three color levels depending on x value.
	for y = -1:1:1		% y-coord changes faster than x.
		for z = -1:1:1	% z-coord changes the fastest.
			coords(:,idx) = [x,y,z];
			colors(idx) = color;
			idx = idx + 1;
		end
	end
end

%%%% Original data packet alongside the gradient at the center point.
%%%%

v = [-3/4, -1/4, -2/4]';
vxy = [v(1:2); 0];
drawCube( coords, colors, "Original", v, vxy );

%%%% Reorienting data packet so that the gradient points in the direction
%%%% of the first quadrant in the (fix-axis) local coordinate system.

theta = pi;		% Rotating about the z axis puts v's projection onto xy-plane on first octant.
Rz = [cos(theta), -sin(theta), 0;
	  sin(theta),  cos(theta), 0;
	           0,           0, 1];
coords = Rz * coords;
v = Rz * v;
vxy = Rz * vxy;
vxz = [v(1); 0; v(3)];
drawCube( coords, colors, "Rotated about z-axis", v, vxy );
quiver3( 0, 0, 0, vxz(1), vxz(2), vxz(3), "color", "#7E2F8E", "MaxHeadSize", 0.5, "AutoScale", 0, "LineWidth", 2 );

psi = -pi/2;	% Rotating about the y axis puts v's projection onto xz plane on first octant.
Ry = [ cos(psi), 0, sin(psi);
	          0, 1,        0;
	  -sin(psi), 0, cos(psi)];
coords = Ry * coords;
v = Ry * v;
vxy = Ry * vxy;
vxz = Ry * vxz;
drawCube( coords, colors, "Rotated about y-axis", v, vxy, vxz );

% Reflections about the plane with normal n=[-0.5, -0.5, 1], are no good -- put points outside canonical cube.
% Reflections about the plane with normal n=[1,-1,0] do work as expected.
n = normalize([1, -1, 0], "norm")';
for idx = 1:27
	coords(:,idx) = coords(:,idx) - 2*n*dot(n,coords(:,idx));
end
v2 = v - 2*n*dot(n,v);
vxy = vxy - 2*n*dot(n,vxy);
vxz = vxz - 2*n*dot(n,vxz);
fig = drawCube( coords, colors, "Reflected", v2 );
mArrow3([0, 0, 0], v, "stemWidth", 0.012, "color", "#EDB120", "FaceAlpha", 0.15);	% Previous v's location.
plot3([v(1), v2(1)], [v(2), v2(2)], [v(3), v2(3)], ":", "color", "#ccc", "LineWidth", 2);

patch([1,1,-1,-1], [1,1,-1,-1], [-1,1,1,-1], [230/255, 234/255, 235/255], ...
	  "EdgeColor", "none", "FaceAlpha", 0.1 );
% print( fig, "figure", "-dpdf" );
