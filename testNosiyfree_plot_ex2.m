clc,clear all
[~,sucess]=Noisefree(5,21,5,100,2); %图一
figure(1)
plot(sucess(:,1),'b-')
hold on 
plot(sucess(:,2),'g-.')
hold on 
plot(sucess(:,3),'r*')
hold on 
plot(sucess(:,4),'ks')
hold on 
LEG = legend('L1ADMM',  'DCA','ADMM','ADMMweight', 'location', 'NorthEast');


alpha1 = Noisefree(5,1,5,1,7499); %图2 ksta,num,F,numiter,不提前终止
alpha2 = Noisefree(5,1,20,1,4800); %图2 可以提前终止
figure(2)
plot(alpha1,'k-')
hold on
plot(alpha2,'b-.')
LEG = legend('F=5',  'F=20', 'location', 'NorthEast');

%图3
[~,sucess2] = Noisefree(15,20,5,100,2); 
figure(3)
plot(sucess2(:,1),'b-')
hold on 
plot(sucess2(:,2),'g-.')
hold on 
plot(sucess2(:,3),'r*')
hold on 
plot(sucess2(:,4),'k+')
hold on 
LEG = legend('L1ADMM',  'DCA','ADMM','ADMMweight', 'location', 'NorthEast');

%图4
[~,sucess3] = Noisefree(15,20,20,100,2);%图4
figure(4)
plot(sucess3(:,1),'b-')
hold on 
plot(sucess3(:,2),'g-.')
hold on 
plot(sucess3(:,3),'r*')
hold on 
plot(sucess3(:,4),'k+')
hold on 
LEG = legend('L1ADMM',  'DCA','ADMM','ADMMweight', 'location', 'NorthEast');