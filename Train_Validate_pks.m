clc
clear all
close all

p=35;
p_ad=37;
v1=[];
v2=[];
e1=[];
e2=[];
AV=load("Resumen_a.mat").historia;
NMAL=load("Resumen_n.mat").historia;
%NMAL=NMAL(randperm(size(NMAL,2)))
ls=160;%limite superior para probar
puntos0=zeros((2*p)+p_ad,80);

Train_loss_fn=[];%loss falsos normales en entrenamiento
Train_loss_fb=[];%loss falsos bloqueos en entrenamiento
Val_loss_fn=[];%loss falsos normales en validacion
Val_loss_fb=[];%loss falsos bloqueos en validacion
%Cargando data de entrenamiento
for n=1:p
    puntos0(n,1:size(NMAL(n).pks,2))=NMAL(n).pks;
    puntos0(n,size(NMAL(n).pks,2):end)=NMAL(n).pks(end);
    disp(strcat("TN",NMAL(n).nombre))
end
for n=1:p+p_ad
    puntos0(n+p,1:size(AV(n).pks,2))=AV(n).pks;
    puntos0(n+p,size(AV(n).pks,2)+1:end)=AV(n).pks(end);
        disp(strcat("TA",AV(n).nombre))
end

%Cargando data de validacion
for n=p+1:size(NMAL,2)
    puntos_valida_n(n-p,1:size(NMAL(n).pks,2))=NMAL(n).pks;
    puntos_valida_n(n-p,size(NMAL(n).pks,2):ls+1)=NMAL(n).pks(end);%puntos_valida_n(n-p,size(NMAL(n).pks,2));
    disp(strcat("VN",NMAL(n).nombre))
end
for n=p+p_ad+1:size(AV,2)
    puntos_valida_a(n-(p+p_ad),1:size(AV(n).pks,2))=AV(n).pks;
    puntos_valida_a(n-(p+p_ad),size(AV(n).pks,2)+1:ls+1)=AV(n).pks(end);%puntos_valida_a(n-p,size(AV(n).pks,2));
    disp(strcat("VA",AV(n).nombre))
end
puntos_val=[puntos_valida_n;puntos_valida_a];

%Creando vector target de validacion
Y_val=[ones(size(puntos_valida_n,1),1);2*ones(size(puntos_valida_a,1),1)];
%Y_val=[ones(1,p) 2*ones(1,p)];

for o=26%2:ls%3:200
    of=o;
    nbin=of;
    bins=linspace(0,1,nbin);
    puntos=puntos0;

    dp=diff(puntos,1,2);
    dp=dp./max(dp')';
    bins=linspace(0,max(max(dp)),nbin);
    hc=histc(dp',bins)';
    hc=hc(1:end,2:end);
    bins=bins(2:end);
    hc=hc./max(hc')';
    %Ahora puntos va a llevar la data del conteo del histograma
    puntos=hc;
    cn=mean(puntos(1:p,1:nbin-1)',2)';
    ca=mean(puntos(p+1:p+1+p_ad,1:nbin-1)',2)';

    figure()
    subplot(2,2,[2,4])
    imshow(hc)
    xlabel("Bins")
    ylabel("n-ésima señal")
    title("Hist-count del dataset de entrenamiento")
    set(gcf, 'Position', get(0, 'Screensize'));
    subplot(2,2,1)
    bar(bins,cn)
    xlabel("Bins")
    ylabel("Conteo Normalizado")
    title("Histograma promedio datos de entrenamiento Normales")


    subplot(2,2,3)
    bar(bins,ca)
    xlabel("Bins")
    ylabel("Conteo Normalizado")
    title("Histograma promedio datos de entrenamiento con bloqueo")
    drawnow 
    hold off
    saveas(gcf,strcat("./Histo",num2str(o),".jpg"));
    %Entrenamiento kmeans
    [idx,C] = kmeans(puntos,2,Start=[cn;ca]);

    %Entrenamiento de SVM (Support Vector Machine)
    Y=[ones(1,p) 2*ones(1,p+p_ad)];
    SVMModel = fitcsvm(puntos,int2str(Y'),'KernelFunction','polynomial');
    save svm.mat SVMModel
    %pause(3.5)

    %Plotting scores de entrenamiento
    figure(3)
    disp(puntos)
    [Y_pred_svm,score] = predict(SVMModel,puntos);
    subplot(231)
    [fnt1,fbt1]=plot_confmat(Y,idx,"Kmeans",50,50);
    subplot(232)
    [fnt2,fbt2]=plot_confmat(int2str(Y'),Y_pred_svm,strcat("SVM ",int2str(nbin)),50,50);
    Train_loss_fn=[Train_loss_fn,[fnt1;fnt2]];
    Train_loss_fb=[Train_loss_fb,[fbt1;fbt2]];
    %Procesamiento de validacion
    dp_val=diff(puntos_val,1,2);
    dp_val=dp_val./max(dp_val')';
    bins=linspace(0,max(max(dp_val)),nbin);
    hc_val=histc(dp_val',bins)';
    hc_val=hc_val(1:end,2:end);
    bins=bins(2:end);
    hc_val=hc_val./max(hc_val')';
    %Ploteo de validacion
    subplot(234)
    [fnv1,fbv1]=plot_confmat_val_KM(C,hc_val,Y_val,"Kmeans\_Validacion",90,90);
    subplot(235)
    [fnv2,fbv2]=plot_confmat_val_SVM(SVMModel,hc_val,Y_val,"SVM\_Validacion",90,90);
    Val_loss_fn=[Val_loss_fn,[fnv1;fnv2]];
    Val_loss_fb=[Val_loss_fb,[fbv1;fbv2]];
    saveas(gcf,strcat("./Confusion",num2str(o),".jpg"));
    close all
end
figure(4)
plot_loses(Train_loss_fn,Train_loss_fb,Val_loss_fn,Val_loss_fb,["Kmeans","SVM"]);
function [fn,fb]=plot_confmat(Y,Y_pred,Metodo,mfp,mfn)
%============================================================================================================
%Y:Clases reales
%Y_pred:Clases predichas por el algoritmo
%Metodo: Algoritmo usado
%mfp: Maximo falsos positivos
%mfp: Maximo falsos negativos
%============================================================================================================
c=confusionmat(Y,Y_pred);
if (c(1,2)<mfp) & (c(2,1)<mfn)
    confusionchart(c,["Normal","Bloqueo AV"],"Normalization","row-normalized",XLabel="Prediccion del algoritmo",Ylabel="Clase a la que pertenece");
    title(Metodo)
end
fn=c(1,2)/sum(c(1,:));
fb=c(2,1)/sum(c(2,:));
end

function plot_hist_points(bins,hc,p)
%============================================================================================================
%bins:vector de bins del histograma
%hc:hist count o conteo para cada bin del histograma
%p:numero de vectores de entrenamiento
%============================================================================================================
scatter(0,0,60,"r","filled");
hold on
scatter(0,0,20,"b","filled");
for n=1:p
    scatter(bins,hc(n,:),60,"r","filled");
end
for n=p+1:2*p
    scatter(bins,hc(n,:),20,"b","filled");
end
xlabel("Bins");
ylabel("Frecuencia normalizada al maximo");
legend(["Normal","Bloqueo AV"]);

end

function [fn,fb]=plot_confmat_val_KM(C,v_points,Y_val,Metodo,mfp,mfn)
%============================================================================================================
%C:Matriz de centroides
%v_points:vector de puntos para validacion
%Y_val:Etiquetas de validacion
%============================================================================================================
[~,idx_val]=pdist2(C,v_points,'euclidean','Smallest',1);
c=confusionmat(Y_val,idx_val);
if (c(1,2)<mfp) & (c(2,1)<mfn)
    confusionchart(c,["Normal","Bloqueo AV"],"Normalization","row-normalized",XLabel="Prediccion del algoritmo",Ylabel="Clase a la que pertenece");
    title(Metodo)
end
fn=c(1,2)/sum(c(1,:));
fb=c(2,1)/sum(c(2,:));
end
function [fn,fb]=plot_confmat_val_SVM(svmModel,v_points,Y_val,Metodo,mfp,mfn)
%============================================================================================================
%C:Matriz de centroides
%v_points:vector de puntos para validacion
%Y_val:Etiquetas de validacion
%============================================================================================================
[Y_pred_svm_val,score]=predict(svmModel,v_points);
c=confusionmat(int2str(Y_val),Y_pred_svm_val);
if (c(1,2)<mfp) & (c(2,1)<mfn)
    confusionchart(c,["Normal","Bloqueo AV"],"Normalization","row-normalized",XLabel="Prediccion del algoritmo",Ylabel="Clase a la que pertenece");
    title(Metodo)
end

fn=c(1,2)/sum(c(1,:));
fb=c(2,1)/sum(c(2,:));
end
function plot_loses(tfn,tfb,vfn,vfb,modelos)
fig=figure(4);
%============================================================================================================
%============================================================================================================
for m=1:size(modelos,2)
    subplot(2,2,1)
    hold on
    title("Totales en entrenamiento")
    plot(tfb(m,:)+tfn(m,:),'DisplayName',strcat('error aboluto ',modelos(m)))
    legend
    subplot(2,2,2)
    hold on
    title("Desgloce en entrenamiento")
    plot(tfn(m,:),'DisplayName',strcat('% Falsos Normales ',modelos(m)))
    plot(tfb(m,:),'DisplayName',strcat('% Falsos Bloqueos ',modelos(m)))
    legend
    subplot(2,2,3)
    hold on
    title("Totales en validacion")
    plot(vfb(m,:)+vfn(m,:),'DisplayName',strcat('error aboluto ',modelos(m)))
    legend
    subplot(2,2,4)
    hold on
    title("Desgloce en validacion")
    plot(vfn(m,:),'DisplayName',strcat('% Falsos Normales ',modelos(m)))
    plot(vfb(m,:),'DisplayName',strcat('% Falsos Bloqueos ',modelos(m)))
    legend
end

han=axes(fig,'visible','off');
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Error normalizado por clase');
xlabel(han,'Numero de bins');
end