clear all 
close all
x=0:0.01:5*pi
a=1%5*sin(x)
b=3*sin(10*x)
c=sin(100*x)
a=a+b+c
for i=30:100
    [p,s,mu] = polyfit((1:numel(a))',a,i);
    f_y = polyval(p,(1:numel(a))',[],mu);
    hold off
    plot(a)
    hold on
    plot(f_y)
    drawnow
    pause
    plot(f_y-a')
    drawnow
    pause
end