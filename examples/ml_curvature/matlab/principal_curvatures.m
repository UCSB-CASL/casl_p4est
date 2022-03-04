clear variables;

m = 50;

xmin = -4; xmax = 4;
ymin = -4; ymax = 4;
zmin = -4; zmax = 4;

rotationAngle = pi / 3;

a = 1; b = 1; c = 1;

x = linspace(xmin,xmax,m); dx = x(2)-x(1);
y = linspace(ymin,ymax,m); dy = y(2)-y(1);
z = linspace(zmin,zmax,m); dz = z(2)-z(1);

phi = zeros(m,m,m);
phi_x  = zeros(m,m,m); phi_y  = zeros(m,m,m); phi_z  = zeros(m,m,m);
phi_xx = zeros(m,m,m); phi_yy = zeros(m,m,m); phi_zz = zeros(m,m,m);
phi_xy = zeros(m,m,m); phi_xz = zeros(m,m,m); phi_yz = zeros(m,m,m);
kappaM = zeros(m,m,m); kappaG = zeros(m,m,m);
kappa1 = zeros(m,m,m); kappa2 = zeros(m,m,m);
kappaMCheck = zeros(m,m,m);
kappaGCheck = zeros(m,m,m);

% Define a saddle:
for i = 1:m
    for j = 1:m
        for k = 1:m
            xi = cos(rotationAngle)*x(i) - sin(rotationAngle)*y(j);
            yj = sin(rotationAngle)*x(i) + cos(rotationAngle)*y(j);
            zk = z(k);
            % Sphere:
%             phi(i,j,k) = xi*xi + yj*yj + zk*zk - (2)^2;
            % Saddle:
            phi(i,j,k) = xi*xi/a/a - yj*yj/b/b + zk*zk/c/c - 1;
        end
    end
end


% Compute first-order derivatives for the interior points:
for i = 2:m-1
    for j = 2:m-1
        for k = 2:m-1
            phi_x(i,j,k) = ( phi(i+1,j,k) - phi(i-1,j,k) )/2/dx;
            phi_y(i,j,k) = ( phi(i,j+1,k) - phi(i,j-1,k) )/2/dy;
            phi_z(i,j,k) = ( phi(i,j,k+1) - phi(i,j,k-1) )/2/dz;
        end
    end
end

% Compute second-order derivatives for the interior points:
for i = 3:m-2
    for j = 3:m-2
        for k = 3:m-2
            phi_xx(i,j,k) = ( phi_x(i+1,j,k) - phi_x(i-1,j,k) )/2/dx;
            phi_yy(i,j,k) = ( phi_y(i,j+1,k) - phi_y(i,j-1,k) )/2/dy;
            phi_zz(i,j,k) = ( phi_z(i,j,k+1) - phi_z(i,j,k-1) )/2/dz;

            phi_xy(i,j,k) = ( phi_x(i,j+1,k) - phi_x(i,j-1,k) )/2/dy;
            phi_xz(i,j,k) = ( phi_x(i,j,k+1) - phi_x(i,j,k-1) )/2/dz;
            phi_yz(i,j,k) = ( phi_y(i,j,k+1) - phi_y(i,j,k-1) )/2/dz;
        end
    end
end

% Compute curvatures for the interior points:
for i = 3:m-2
    for j = 3:m-2
        for k = 3:m-2
            phix  = phi_x (i,j,k);  phiy  = phi_y (i,j,k);  phiz  = phi_z (i,j,k);
            phixx = phi_xx(i,j,k);  phiyy = phi_yy(i,j,k);  phizz = phi_zz(i,j,k);
            phixy = phi_xy(i,j,k);  phixz = phi_xz(i,j,k);  phiyz = phi_yz(i,j,k);

            % Compute the mean curvature kappaM:
            num = (phiyy + phizz)*phix^2 ...
                + (phixx + phizz)*phiy^2 ...
                + (phixx + phiyy)*phiz^2 ...
                - 2*phix*phiy*phixy ...
                - 2*phix*phiz*phixz ...
                - 2*phiy*phiz*phiyz;
            
            den = (phix^2 + phiy^2 + phiz^2)^1.5;
            
            kappaM(i,j,k) = 0.5* num / den;

            % Compute the Gauss curvature kappaG:
            num = phix^2*(phiyy*phizz - phiyz^2) ...
                + phiy^2*(phixx*phizz - phixz^2) ...
                + phiz^2*(phixx*phiyy - phixy^2) ...
                + 2*( ...
                phix*phiy*(phixz*phiyz-phixy*phizz) + ...
                phiy*phiz*(phixy*phixz-phiyz*phixx) + ...
                phix*phiz*(phixy*phiyz-phixz*phiyy) );

            den = (phix^2 + phiy^2 + phiz^2)^2;
            
            kappaG(i,j,k) = num / den;

            % Compute the principal curvatures kappa1 and kappa2:
            kappa1(i,j,k) = kappaM(i,j,k) + sqrt( kappaM(i,j,k)*kappaM(i,j,k) - kappaG(i,j,k) );
            kappa2(i,j,k) = kappaM(i,j,k) - sqrt( kappaM(i,j,k)*kappaM(i,j,k) - kappaG(i,j,k) );
            kappaMCheck(i,j,k) = 0.5*( kappa1(i,j,k) + kappa2(i,j,k) );
            kappaGCheck(i,j,k) =       kappa1(i,j,k) * kappa2(i,j,k);
        end
    end
end

% Plot kappaM on top of zero-level set:
figure(1);
isosurface(x,y,z,phi,0,kappaM); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('KappaM'); colorbar;

% Plot kappaG on top of zero-level set:
figure(2);
isosurface(x,y,z,phi,0,kappaG); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('KappaG'); colorbar;

% Plot kappa1 on top of zero-level set:
figure(3);
isosurface(x,y,z,phi,0,kappa1); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('Kappa1'); colorbar;

% Plot kappa2 on top of zero-level set:
figure(4);
isosurface(x,y,z,phi,0,kappa2); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('Kappa2'); colorbar;

% Plot kappaMCheck on top of zero-level set:
figure(5);
isosurface(x,y,z,phi,0,kappaMCheck); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('KappaMCheck'); colorbar;

% Plot kappaGCheck on top of zero-level set:
figure(6);
isosurface(x,y,z,phi,0,kappaGCheck); 
axis([xmin + 2*dx xmax-2*dx, ymin + 2*dy ymax-2*dy, zmin + 2*dz zmax-2*dz]);
title('KappaGCheck'); colorbar;