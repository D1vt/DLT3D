pkg load communications
REAL_OBJECT=csvread("realworld.csv");
x=rand(1,50);   %take a random X
y=rand(1,50);   %random Y
m=0.00001+randn;      %take a random constant
z=m*ones(50);  %fill Z with that constant
%surf(x, y, z)
xsize = numel(x);
zsize = numel(z(:,1));
for i=1:6
indexreal = randi([1, numel(REAL_OBJECT(:,1))], 1, 1);
indexx = randi([1, xsize], 1, 1);
%indexx = randperm(xsize);
indexz =randi([1, zsize], 1, 1);
    p(i,1)= x(indexx(1: 1))/z(indexz(1: 1));    %convert from homogeneous
    p(i,2)=y(indexx(1: 1))/z(indexz(1: 1)); %convert from homogeneous
%p is the u,v matrix
X(i)=REAL_OBJECT(indexreal,1);
Y(i)=REAL_OBJECT(indexreal,2);
Z(i)=REAL_OBJECT(indexreal,3);  %Six points real world
 Noise(i,:)= awgn(p(i,:),randn); %randomnoiseadded to each x&y pair
 %6 random points u,v from my plane & 6 from real world
 if (i>1) 
   if(sqrt(Noise(i,1)*Noise(i,1)+Noise(i,2)*Noise(i,2))>maximum)
   maximum= sqrt(Noise(i,1)*Noise(i,1)+Noise(i,2)*Noise(i,2));
 endif
 if(sqrt(p(i,1)*p(i,1)+p(i,2)*p(i,2))>maximum2)
 maximum2= sqrt(p(i,1)*p(i,1)+p(i,2)*p(i,2));
 endif
else
 maximum= sqrt(Noise(i,1)*Noise(i,1)+Noise(i,2)*Noise(i,2));
 maximum2= sqrt(p(i,1)*p(i,1)+p(i,2)*p(i,2));
 endif
endfor 
%to normalize find max distance (sqrt(x^2+y^2))
Norm_Noise(:,:)= Noise(:,:)/maximum; %normalize after adding noise  
pnorm(:,:)=p(:,:)/maximum2;
scatter(Norm_Noise(:,1),Norm_Noise(:,2),'g','*')
title('Combine Normalized&Non-Normalized Points')
hold on
scatter(Noise(:,1),Noise(:,2), 'r')
scatter(p(:,1),p(:,2), 'b')
scatter(pnorm(:,1),pnorm(:,2),'y','*')
hold off

%pnorm -> without noise, yellow & Norm_Noise -> with noise ,green
%now DLT to find M matrix 
l=1;
for j=1:2*i
  if(mod(j,2)==0)   %if even 0,0,0,0,xi,yi,zi,1,-vixi,-viyi,-vizi,-vi
 for k=1:4
   A(j,k)=0;
   B(j,k)=0;
 endfor
  A(j,5)=X(l);
  A(j,6)=Y(l);
  A(j,7)=Z(l);
  A(j,8)=1;
  A(j,9)=-Norm_Noise(l,2)*X(l);
  A(j,10)=-Norm_Noise(l,2)*Y(l);
  A(j,11)=-Norm_Noise(l,2)*Z(l);
  A(j,12)=-Norm_Noise(l,2);
  B(j,5)=X(l);
  B(j,6)=Y(l);
  B(j,7)=Z(l);
  B(j,8)=1;
  B(j,9)=-pnorm(l,2)*X(l);
  B(j,10)=-pnorm(l,2)*Y(l);
  B(j,11)=-pnorm(l,2)*Z(l);
  B(j,12)=-pnorm(l,2);
 l=l+1;
else            %if odd xi,yi,zi,1,,0,0,0,0,-uixi,-uiyi,-uizi,-ui
   for k=5:8
    A(j,k)=0; 
   B(j,k)=0; 
  endfor
  A(j,1)=X(l);
  B(j,1)=X(l);
  A(j,2)=Y(l);
  A(j,3)=Z(l);
  A(j,4)=1;
  B(j,2)=Y(l);
  B(j,3)=Z(l);
  B(j,4)=1;
  A(j,9)=-Norm_Noise(l,1)*X(l);
  A(j,10)=-Norm_Noise(l,1)*Y(l);;
  A(j,11)=-Norm_Noise(l,1)*Z(l);
  A(j,12)=-Norm_Noise(l,1);
  B(j,9)=-pnorm(l,1)*X(l);
  B(j,10)=-pnorm(l,1)*Y(l);;
  B(j,11)=-pnorm(l,1)*Z(l);
  B(j,12)=-pnorm(l,1);
endif
endfor
%now we can create code for SVD/ready function  %ATA=(A')*(A);[V,D] = eig(ATA)
[U,S,V] = svd(A);
[U1,S1,V1]=svd(B);
Mnoise=V(:,12);
%M is our final vector, with R|t parameters , now the goal is to find the R|t
Mnorm=V1(:,12);
%error between M with noise and M without noise
errormean=1/12*sum(abs(Mnoise-Mnorm));
Mnoise=reshape(Mnoise,[],3)';  %Make 3*4 as we want 

%K is well known [a b c
%                 0 d e
%                 0 0 f]
  K(1,1)=input("Give me random a: ")
  K(1,2)=input("Give me random b: ")
  K(1,3)=input("Give me random c: ")
  K(2,2)=input("Give me random d: ")
  K(2,3)=input("Give me random e: ")
  K(3,3)=input("Give me random f: ")
T(3)=Mnoise(3,4)/K(3,3);  %fTz=m34
T(2)=(Mnoise(2,4)-K(2,3)*T(3))/K(2,2); %dTy+eTz=m24
T(1)=(Mnoise(1,4)-K(1,2)*T(2)-K(1,3)*T(3))/K(1,1); %aTx+bTy+cTz=m14

for i=1:3
  R(3,i)=Mnoise(3,i)/K(3,3);  %fr3i=m3i
  R(2,i)=(Mnoise(2,i)- R(3,i)*K(2,3))/K(2,2);  %dr2i+er3i=m2,i
 R(1,i)= (Mnoise(1,i)-K(1,2)*R(2,i)-K(1,3)*R(3,i))/K(1,1); %ar1i+br2i+cr3i=m1i
endfor
