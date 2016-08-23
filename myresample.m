function out = myresample(in,new,old)
    out=zeros(1,new);
    for i = 1:new
        temp = i*old/new;
        if temp<1
            out(1,i)=in(1,1);
        elseif temp>old
            out(1,i)=in(1,old);
        else
            out(1,i)=in(1,floor(temp))+(temp-floor(temp))*(in(1,ceil(temp))-in(1,floor(temp)));
        end
    end

end