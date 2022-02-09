function [f] = drawCube(coords, colors, titleMsg, v, vxy, vxz)
	% Let's generate the line segments in the grid.
	linesXZ = zeros(18,3);	% 9 lines perpendicular to x and z axes.
	linesXY = zeros(18,3);	% 9 lines perpendicular to x and y axes.
	idx = 0;
	for x = -1:1:1
		for z = -1:1:1
			startIdx = 2*idx + 1;	% Line starting point.
			linesXZ(startIdx, :) = [x, -1, z];
			endIdx = startIdx + 1;	% Line ending point.
			linesXZ(endIdx, :) = [x, +1, z];

			linesXY(startIdx, :) = [x, z, -1];
			linesXY(endIdx, :) = [x, z, +1];
			idx = idx + 1;
		end
	end
	linesY = zeros(18,3);	% 9 lines perpendicular to y axis.
	idx = 0;
	for y = -1:1:1
		for z = -1:1:1
			startIdx = 2*idx + 1;	% Line starting point.
			linesY(startIdx, :) = [-1, y, z];
			endIdx = startIdx + 1;	% Line ending point.
			linesY(endIdx, :) = [+1, y, z];
			idx = idx + 1;
		end
	end

	f = figure("Renderer", "Painters", 'units', 'inch');
	set( f, 'PaperSize', [5 5] );
	set( f, 'PaperPositionMode', 'manual' );
	set( f, 'PaperPosition', [0 0 5 5] );
	hold on;
	for idx = 0:8
		lidx = idx*2 + 1;
		plot3([linesXZ(lidx,1), linesXZ(lidx+1,1)], ...	% Lines perpendincular to x and z.
			  [linesXZ(lidx,2), linesXZ(lidx+1,2)], ...
			  [linesXZ(lidx,3), linesXZ(lidx+1,3)], "-", "color", "#abb");
		plot3([linesXY(lidx,1), linesXY(lidx+1,1)], ...	% Lines perpendincular to x and y.
			  [linesXY(lidx,2), linesXY(lidx+1,2)], ...
			  [linesXY(lidx,3), linesXY(lidx+1,3)], "-", "color", "#abb");
		plot3([linesY(lidx,1), linesY(lidx+1,1)], ...	% Lines perpendicular to y.
			  [linesY(lidx,2), linesY(lidx+1,2)], ...
			  [linesY(lidx,3), linesY(lidx+1,3)], "-", "color", "#abb");
	end

	for idx = 1:27	% Plotting the points.
		plot3( coords(1,idx), coords(2,idx), coords(3,idx), "o", ...
			   "MarkerFaceColor", colors(idx), "color", colors(idx), "MarkerSize", 7 );
		text( coords(1,idx)+0.04, coords(2,idx)+0.04, coords(3,idx), sprintf("%2d", idx-1), "color", colors(idx) );
	end
	xlabel( "x" );
	ylabel( "y" );
	zlabel( "z" );
	axis equal;
	xlim([-1.5, 1.5]);
	ylim([-1.5, 1.5]);
	zlim([-1.5, 1.5]);
	rotate3d on;
	
	title(titleMsg);
	
	if exist( "v", "var" )
		mArrow3([0, 0, 0], v, "stemWidth", 0.012, "color", "#EDB120");	% The gradient at the center point.
		light('Position',[1.5 -1.5 1.5],'Style','local');
	end
	
	% Emphasizing the first octant.
	octantColor = [0.3010 0.7450 0.9330];
	patch([0,1,1,0], [0,0,1,1], [0,0,0,0], octantColor, ...
	  "EdgeColor", "none", "FaceAlpha", 0.05 );
	patch([0,1,1,0], [0,0,0,0], [0,0,1,1], octantColor, ...
	  "EdgeColor", "none", "FaceAlpha", 0.05 );
	patch([0,0,0,0], [0,1,1,0], [0,0,1,1], octantColor, ...
	  "EdgeColor", "none", "FaceAlpha", 0.05 );
	
	% The (possibly transformed) projection onto xy plane.
	if exist( "vxy", "var" )
		quiver3( 0, 0, 0, vxy(1), vxy(2), vxy(3), "color", "#77AC30", "MaxHeadSize", 0.5, "AutoScale", 0, "LineWidth", 1.5 );
	end
	
	% The (possibly transformed) projection onto xz plane.
	if exist( "vxz", "var" )
		quiver3( 0, 0, 0, vxz(1), vxz(2), vxz(3), "color", "#7E2F8E", "MaxHeadSize", 0.5, "AutoScale", 0, "LineWidth", 1.5 );
	end
	
	view(30,20);
end

