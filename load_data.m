function [X_out,Y_out,Pressure_out]=load_data(data_dir,resampleFlag)
    n=100;
    X_out=cell(n,1);
    Y_out=cell(n,1);
    Pressure_out=1;
    width=1080;
    height=1920;
    for i = 1:n
        fileX=strcat(data_dir,'/X');
        fileX=strcat(fileX,int2str(i));
        fileX=strcat(fileX,'.txt');
        fileY=strcat(data_dir,'/Y');
        fileY=strcat(fileY,int2str(i));
        fileY=strcat(fileY,'.txt');
       
        X=csvread(fileX);
        X=X(1:length(X)-1)./width;
    
        Y=csvread(fileY);
        Y=Y(1:length(Y)-1)./height;
        if resampleFlag==true
            X=myresample(X,50,size(X,2));
            Y=myresample(Y,50,size(Y,2));
        end
        X_out{i,1}=X;
        Y_out{i,1}=Y;
        %figure;
        %scatter(X,Y);
    end

end
