clc
close all
clear all

imagen = struct("nombre","v","tr",zeros(4,1),"ecg",[],"pks",[],"pks_ext",[]);%"descriptor",[],"frec",[],"LPC",[],

directorio='./Normal/';
listado_imagenes=dir(strcat(directorio,'*.jpg'));%dir(strcat(strcat('./',directorio),'/*.jpg'));
listado_imagenes={listado_imagenes.name};
%               __   ___                  __   ___  __        ___       __
% |  |\/|  /\  / _` |__  |\ |     /\     /__` |__  /  ` |  | |__  |\ | /  ` |  /\
% |  |  | /~~\ \__> |___ | \|    /~~\    .__/ |___ \__, \__/ |___ | \| \__, | /~~\

%Metodo para no repetir todo el proceso
nuevo_inicio=0;
if nuevo_inicio==1
    historia=[];
    for n=1:size(listado_imagenes,2)
        historia=[historia imagen];
    end
else
    historico="Concentrado_Normal.mat";
    historico_cargado=load(historico);
    historia=historico_cargado.historia;
end

for n=1:size(listado_imagenes,2)
    nombre_imagen=strcat(directorio, char(listado_imagenes(n)));
    historia(n).nombre=char(listado_imagenes(n));
    color=imread(nombre_imagen);
    %Estandarizando altura de las imagenes
    color=imresize(color,[1600,NaN]);
    %Maxima distancia del tono seleccionado
    umbral=120;
    estima=tanh(mean((mean(mean(color)))-240)/10)*15 +45;
    distancias=sqrt(sum(double(color-estima).^2,3));

    nuea=(distancias>=umbral)*255;

    nsd=zeros(size(nuea));
    nsd(:,:,1)=nuea;
    nsd(:,:,2)=nuea;
    nsd(:,:,3)=nuea;
    nueva=nsd;

    %Definiendo las estructuras para las operaciones morfologicas
    se1 = strel('line',1,2);
    se2 = offsetstrel('ball',2,6);
    se3 = strel('disk',1);
    se4 = strel('line',4,10);
    se5 = strel('diamond',3);
    se6 = strel('octagon',3);
    se7 = strel('disk',3);

    Nueva=nueva;
    IM=imcomplement(Nueva);

    IM2 = imdilate(IM,se1);

    IM3 = imdilate(IM2,se3);

    IM4 = imclose(IM3,se1);

    IM5=imcomplement(IM4);

    Eg=IM5;


    I = rgb2gray(Eg);
    BW =I;% imbinarize(I);

    if nuevo_inicio
        Tira=imcrop(BW);
    else
        [Tira,coords_tira]=imcrop(BW,historia(n).tr);
        historia(n).tr=coords_tira;
    end

    if  sum(sum(Tira==0))/numel(Tira)>.11
        Tira=medfilt2(medfilt2(Tira,[6,6]),[3,3])>.9;
    else
        Tira=medfilt2(Tira,[6,6])>.9;
    end
    Tira=Tira(2:end-2,2:end-2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%% Mapeo del espacio de imagenes al espacio de serie temporal im2tmp %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %encontrar la curva
    S=size(Tira);
    Fs=2820/11;%FS aproximada
    Ts=1/Fs
    for b=1:S(2)
        pixels_0=find(Tira(:,b)==0);
        m_t=mean(pixels_0);
        y=median(pixels_0(abs(pixels_0-m_t)<.05*S(1)));
        x=b*Ts*1j;
        s_k(b)=x+y;
    end
    %quitar nulos al inicio
    for b=1:S(2)
        if ~isnan(s_k(b))
            s_k(1)=s_k(b);
            break
        end
    end
    %quitar nulos al final
    for b=S(2):-1:1
        if ~isnan(s_k(b))
            s_k(end)=s_k(b);
            break
        end
    end
    %Interpolando
    V=s_k;
    X = ~isnan(V);
    Y = cumsum(X-diff([1,X])/2);
    Z = interp1(1:nnz(X),V(X),Y);
    s_k=Z;
    ims=imag(s_k);
    imr=real(s_k);

    %Quitando la tendencia%
    [p,s,mu] = polyfit((1:numel(imr))',imr,6);
    f_y = polyval(p,(1:numel(imr))',[],mu);
    ECG_data = imr - f_y';        % Detrend data
    s_k=ECG_data+(1j*ims);
    Z=s_k;%ECG Vectorizado
    historia(n).ecg=s_k;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Mapeo de dominio temporal a encontrar picos %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Aplicando filtro de media movil y downsampling
    dsf=3;
    s_k=downsample(s_k,dsf);
    s_k=movmean(real(s_k),3)+(1j*imag(s_k));
    s_k=real(s_k);


        atmp=s_k;%(s_k+max(s_k))/(.8*2*max(s_k));
    atmp(atmp<0)=0;
    [a,l]=findpeaks(atmp.^6,'MinPeakDistance',25,'MinPeakHeight',.8*max(atmp));
    mind=mean(diff(l));
        rango_d=max(s_k)-min(s_k);
    s_k=s_k./(.5*(rango_d));
    rango_d=max(s_k)-min(s_k);
    mind=.8*mind;
    smoothECG = sgolayfilt(s_k,7,21);

    [a,l,w,p]=findpeaks(smoothECG,'MinPeakProminence',.1*max(s_k),'MinPeakDistance',mind/9,'Annotate','extents','WidthReference','halfprom');
    historia(n).pks = l;

    [rh,locs_Rwave] = findpeaks((s_k),'MinPeakHeight',0.2,...
        'MinPeakDistance',mind);
    ECG_inverted = -s_k;
    [sh,locs_Swave]=findpeaks(ECG_inverted,'MinPeakHeight',0.2,...
        'MinPeakDistance',mind);
    
    [mh,min_locs] = findpeaks(-smoothECG,'MinPeakProminence',.05*rango_d);

    %Peaks between
    locs_Qwave = min_locs(smoothECG(min_locs)>.9*max(-sh) & smoothECG(min_locs)<.9*max(smoothECG));%.3*(max(s_k)-min(s_k)-1));
    % figure
    l=sort([locs_Rwave,locs_Qwave]);
    
    % historia(n).pks = l;


    % subplot(212)
    % plot(s_k)
    % hold on
    % plot(smoothECG);
    % plot(locs_Qwave,smoothECG(locs_Qwave),'rs','MarkerFaceColor','g')
    % plot(locs_Rwave,s_k(locs_Rwave),'rv','MarkerFaceColor','r')
    % plot(locs_Swave,s_k(locs_Swave),'rs','MarkerFaceColor','b')
    % grid on
    % title('Complejo QRS')
    % xlabel('Muestras')
    % ylabel('Voltage (u)')
    % drawnow
    % hold off
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%% Mapeo de dominio temporal a Fourier %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fs=2820/11
    L=size(Z,2)
    f = Fs*(0:(L/2))/L;
    Y = fft(Z);
    P2 = Y/L;
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    app.four=Pq;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%% Mapeo del espacio de imagenes a descriptores de Fourier %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

end
save Resumen_n.mat historia