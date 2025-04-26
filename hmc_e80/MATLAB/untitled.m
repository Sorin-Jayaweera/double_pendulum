L =1;
spacing = 0.05;
x = 0:spacing:L;
y = 0:spacing:L;
z = 0:spacing:L;
[X,Y,Z] = meshgrid(x,y,z);
nx = 2;
ny = 1;
nz = 1;

%%
p = cos(nx*pi*X/L).*sin(ny*pi*Y/L).*sin(nz*pi*Z/L);
q = sin(nx*pi*X/L).*cos(ny*pi*Y/L).*sin(nz*pi*Z/L);
r = sin(nx*pi*X/L).*sin(ny*pi*Y/L).*cos(nz*pi*Z/L);


%%

figure;
quiver3(X,Y,Z,p,q,r)
title("Electric field inside a Hollow Box")
subtitle(sprintf("nx %d ny %d nz %d", nx,ny,nz))
xlabel("X")
ylabel("Y")
zlabel("Z")

