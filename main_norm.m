close all;
clear all;
%% variable
n = 100; %number of data
%screen size
width=1080;
height=1920;


%% load data
resampleFlag = false;

[rightX,rightY]=load_data('./CircleRight',resampleFlag);
[leftX,leftY]=load_data('./CircleLeft',resampleFlag);

fprintf('Load data done\n');

%% preprocessing
NFFT = 128;
%resample
init = 50;
rs = 10;
%f=linspace(0,1,NFFT/2+1);
%plot(f,abs(Y(1:NFFT/2+1)));

for i = 1:n
    temp = leftX{i,1};
    %temp=fft(temp,NFFT);
    %resample
    temp=myresample(temp,rs,init);
    leftX{i,1}=abs(temp);
    
    temp = leftY{i,1};
    %temp=fft(temp,NFFT);
    %resample
    temp=myresample(temp,rs,init);
    leftY{i,1}=(abs(temp));
    %figure;
    %plot(abs(temp));
    
    temp = rightX{i,1};
    %temp=fft(temp,NFFT);
    temp=myresample(temp,rs,init);
    rightX{i,1}=abs(temp);
    
    temp = rightY{i,1};
    %temp=fft(temp,NFFT);
    temp=myresample(temp,rs,init);
    rightY{i,1}=abs(temp);
    %figure;
    %plot(abs(temp));
    
    
end
fprintf('preprocessing done\n');

%% NN

nIn = 2*rs;
nOut =2;
nHidden=5;
lRate=0.01;
epoch=500000;
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
    in=[in [leftX{i,1}'; leftY{i,1}']];
    out(1,i)=1;
end
for i=1:80
    in=[in [rightX{i,1}'; rightY{i,1}']];
    out(2,i+80)=1;
end
test=[];
t_out=[];
for i=1:20
   test=[test [leftX{80+i}';leftY{80+i,1}']];
   t_out=[t_out [1;0]];
   test=[test [rightX{80+i}';rightY{80+i,1}']];
   t_out=[t_out [0;1]];
end

%normalization
in_mean=mean(in,2);
in_std=std(in,1,2);
in = (in - repmat(in_mean,[1 nData]))./repmat(in_std, [1 nData]);
in=in/5;
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














