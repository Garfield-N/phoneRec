close all;
clear all;
%% variable
n = 100; %number of data
%screen size
width=1080;
height=1920;


%% load data
resampleFlag = false;

[rightX,rightY]=load_data('./SimpleRight',resampleFlag);
[leftX,leftY]=load_data('./SimpleLeft',resampleFlag);

fprintf('Load data done\n');

%% preprocessing
fprintf('preprocessing done\n');

%% NN

nIn = 6;
nOut =2;
nHidden=5;
lRate=0.03;
epoch=200000;
nData=160;
nTest=40;

w1=rand(nHidden, nIn);
b1=zeros(nHidden,1);
%h1=zeros(nHidden,nData);
w2=rand(nOut,nHidden);
b2=zeros(nOut,1);
h2=zeros(nOut,nData);
out=zeros(nOut,nData);
in=[];
for i=1:80
    in=[in [mean(leftX{i,1}); mean(leftY{i,1});max(leftX{i,1}); max(leftY{i,1});min(leftX{i,1}); min(leftY{i,1})]];
    out(1,i)=1;
end
for i=1:80
    in=[in [mean(rightX{i,1}); mean(rightY{i,1});max(leftX{i,1}); max(leftY{i,1});min(leftX{i,1}); min(leftY{i,1})]];
    out(2,i+80)=1;
end
test=[];
t_out=[];
for i=1:20
   test=[test [mean(leftX{80+i}); mean(leftY{80+i,1});max(leftX{80+i}); max(leftY{80+i,1});min(leftX{80+i}); min(leftY{80+i,1})]];
   t_out=[t_out [1;0]];
   test=[test [mean(rightX{80+i});mean(rightY{80+i,1});max(rightX{80+i});max(rightY{80+i,1});min(rightX{80+i});min(rightY{80+i,1})]];
   t_out=[t_out [0;1]];
end
%256*200
%50*256
in=in;
acc_log=[];
err_log=[];
for i= 1:epoch
    %ff
    h1=1./(1+exp(-(w1*in+repmat(b1,[1,nData]))));
    h2=exp((w2*h1+repmat(b2,[1,nData])));
    h2=h2./repmat(sum(h2,1),[2,1]);
    error=sum(sum((out-h2).^2))/nData;
    %bpp
    err2=out-h2;
    grad_w2=-err2*h1';
    grad_b2=-sum(err2,2);
    w2=w2-lRate*grad_w2/nData;
    b2=b2-lRate*grad_b2/nData;
    
    err1=w2'*err2.*(1-h1).*h1;
    grad_w1=-err1*in';
    grad_b1=-sum(err1,2);
    w1=w1-lRate*grad_w1/nData;
    b1=b1-lRate*grad_b1/nData;
    
    
    h1=1./(1+exp(-(w1*test+repmat(b1,[1,nTest]))));
    h2=exp((w2*h1+repmat(b2,[1,nTest])));
    h2=h2./repmat(sum(h2,1),[2,1]);
    [m, ind]=max(h2);
    [m1, ind1]=max(t_out);
    acc=(40-sum(abs(ind-ind1)))/40*100;
    
    if mod(i,1000) == 0
        fprintf('Error %d\nAccuracy %d\n',error,acc);
        acc_log=[acc_log acc];
        err_log=[err_log error];
    end
    
end
figure;
plot(acc_log);
xlabel('number of epoch *1000');
ylabel('accuracy');
figure;
plot(err_log);
xlabel('number of epoch *1000');
ylabel('error');














