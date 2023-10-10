# MIMO-OFDM
《MIMO-OFDM系统原理、应用及仿真》李莉（实例代码）
余乃双非学渣一枚，注册本站不足一月，2023年10月10日早上9时许，余研究本书，站内寻电子版代码，不得，登陆机械工业出版社官网搜索，搜得，欲下载，提示注册后下载，余无语但理解，注册毕，欲下载，提示本书资源仅供站内验证过教师资格的账号下载，验证教师资格需上传工作单位及证明材料，余大怒，开始对机工社全体职工进行辱骂，辱骂之余，忽灵光一现，身处杭州，也许要拜拜Jack Ma？立刻打开淘宝搜索，果然，复搜几次，豁然开朗，最终花费3.99钱得PPT与代码，感叹之余再次忆起余至爱之人维维豆奶的一句话，“这，就是中国！”......余懒但来自废都，生性仗义，故上传全部代码与诸君分享，然代码并不是m文件，乃一个word文档，鄙人只是代码的搬运工，若有格式疏漏或代码错误等问题，本人概不负责，请辱骂机工社全体职工，谢谢。



——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


2.5 仿真实例
实例2-1 瑞利分布与莱斯分布
功能：绘制瑞利分布曲线与莱斯分布曲线
程序名称：Example2_1.m
程序代码：
clear, clf
N=200000; %产生200000个信道系数供统计使用
level=30; %统计区间被划分的分数。
K_dB=[-40 0 15];%莱斯因子为-40dB、0dB、15dB
gss=['k-*'; 'k-o'; 'k-+';'k-^'];%绘制曲线的颜色、线形与标志符号

% 瑞利模型
Rayleigh_ch=Ray_model(N);%调用Ray_model子程序，产生瑞利分布幅度系数
[temp,x]=hist(abs(Rayleigh_ch(1,:)),level);%统计数据分布
plot(x,temp,gss(1,:))
hold on

%莱斯模型
for i=1:length(K_dB);%对不同莱斯因子进行信道模型仿真
Rician_ch(i,:) = Ric_model(K_dB(i),N);%调用Ric_model产生莱斯分布幅度系数
[temp x] = hist(abs(Rician_ch(i,:)),level);%统计数据分布
plot(x,temp,gss(i+1,:))
end
xlabel('x'), ylabel('Occurrence')
legend('Rayleigh','Rician, K=-40dB','Rician, K=0dB','Rician, K=15dB')

%瑞利信道模型子程序，子程序程序名称：Ray_model.m
function H=Ray_model(L)
% 输入参数 L: 仿真信道个数，为N=200000
% 输出参数 H： 返回瑞利信道矩阵
H = (randn(1,L)+j*randn(1,L))/sqrt(2);
%产生实部为高斯分布、虚部为高斯分布、包络为瑞利分布的信道系数。实部功率为1/2，虚部功率为1/2，因
%此该行指令返回单位功率的或称归一化功率的瑞利信道幅度系数。

   %莱斯信道模型子程序，子程序程序名称：Ric_model.m
function H=Ric_model(K_dB,L)
% 输入参数 : K_dB 为莱斯因子，L为仿真信道个数
% 输出参数H: 返回莱斯信道矩阵
K = 10^(K_dB/10);%将dB值描述的莱斯因子转换为幅度值
H = sqrt(K/(K+1)) + sqrt(1/(K+1))*Ray_model(L);
%产生莱斯信道幅度系数。莱斯信道模型中包含视距通信，收发之间有直通路径。

程序仿真结果见图2-9。

实例2-2 两径信道与指数信道模型	
功能：产生一个两径信道和一个指数衰减的多径信道。
程序名称：Example2_2
程序代码：
    clear, clf
 scale=1e-9;      % 纳秒量级
   Ts=10*scale;     % 抽样时间间隔为10ns，在这个程序中这个量也为指数信道路径间隔
   t_rms=30*scale; % RMS 时延扩展为30ns
   num_ch=10000;    % 仿真信道个数

% 两径信道模型
% 产生并绘制了理想的两径信道模型和瑞利分布两径信道模型。
   pow_2=[0.5 0.5]; delay_2=[0 t_rms*2]/scale;
   %给出理想两径信道功率均为0.5，延时为0和60ns
   H_2 = [Ray_model(num_ch); Ray_model(num_ch)].'*diag(sqrt(pow_2));
   %产生瑞利两径信道幅度系数。通过调用子程序Ray_model产生归一化功率的瑞利两径信道幅度系数。
   avg_pow_h_2 = mean(H_2.*conj(H_2));
   %计算瑞利分布两径信道每一径的平均功率。在这里可以看到上一条语句中diag(sqrt(pow_2))的作用。%当通过对幅度系数进行 运算计算功率时，sqrt(pow_2)可以使每一径的功率为pow_2，即每一径的功率为0.5。
   subplot(121)
   stem(delay_2,pow_2,'ko'), hold on, stem(delay_2,avg_pow_h_2,'k.');
   xlabel('Delay[ns]'), ylabel('Channel Power[linear]');
   title('2-ray Model');
   legend('Ideal','Simulation'); axis([-10 140 0 0.7]);

% 指数信道模型
%产生并绘制理想的指数信道模型和瑞利分布的指数信道模型。
pow_e=exp_PDP(t_rms,Ts); %通过调用exp_PDP子程序，计算理想指数信道每一径上的功率。
  delay_e=[0:length(pow_e)-1]*Ts/scale;%计算指数信道每一径的延时，单位为ns
  for i=1:length(pow_e)
  H_e(:,i)=Ray_model(num_ch).'*sqrt(pow_e(i));
  end
  %计算瑞利分布的指数信道幅度系数。通过调用Ray_model产生归一化功率的瑞利分布幅度系数，%sqrt(pow_e(i))的作用类似于diag(sqrt(pow_2))。
  avg_pow_h_e = mean(H_e.*conj(H_e));%计算瑞利分布指数信道的平均功率。
  %由于sqrt(pow_e(i))的存在，瑞利分布指数信道每一径的平均功率也为pow_e(i)，即与理想指数信道
%每一径功率相同。
  subplot(122)
  stem(delay_e,pow_e,'ko'), hold on, stem(delay_e,avg_pow_h_e,'k.');
  xlabel('Delay[ns]'), ylabel('Channel Power[linear]');
  title('Exponential Model'); axis([-10 140 0 0.7])
  legend('Ideal','Simulation')

% 瑞利信道模型子程序，子程序程序名称：Ray_model.m
  function H=Ray_model(L)
  H = (randn(1,L)+j*randn(1,L))/sqrt(2);

%指数信道PDP子程序，子程序名称：exp_PDP.m
  function PDP=exp_PDP(tau_d,Ts,A_dB,norm_flag)
% 输入参数:
% tau_d : RMS 延时扩展，单位为s
% Ts : 抽样时间间隔，在这里也为指数信道路径间隔，单位为s
% A_dB : 最小不可忽略径[dB]
% norm_flag : 标准化标志
% 输出参数:
% PDP : 输出指数信道PDP矢量
  if nargin<4, norm_flag=1; end % 判断子程序调用参数个数，小于4，则norm_flag=1。
  if nargin<3, A_dB=-20; end % 判断子程序调用参数个数，小于4，则A_dB=-20。
  %由于主程序中调用该子程序时，只有两个参数，所以上两条语句实际是幅值norm_flag=1和A_dB=-20。
sigma_tau=tau_d; 
A=10^(A_dB/10);
  lmax=ceil(-tau_d*log(A)/Ts); % 计算最大路径序号，参见式(2-34)。
  %以下参见式(2-36) 
  if norm_flag
  p0=((1-exp(-(lmax+1)*Ts/sigma_tau))/(1-exp(-Ts/sigma_tau)))/30; 
  else p0=1/sigma_tau; %计算式(2-37)中的P0
  end
  
% 指数信道PDP
  l=0:lmax; 
  PDP = p0*exp(-l*Ts/sigma_tau); % 参见式(2-37)

程序仿真结果如图2-17所示。图2-17（a）为理想两径信道和瑞利两径信道的PDP曲线，图2-17（b）为离散指数信道和瑞利指数信道的PDP曲线。
 
                               （a）                          （b）
图2-17 两径与指数信道模型

实例2-3 IEEE802.11信道PDP与频谱分布
功能：实现IEEE802.11信道仿真，画出IEEE802.11信道的PDP曲线与频谱图。
程序名称：Example2_3.m
程序代码：
clear, clf
scale=1e-9; % 纳秒量级
Ts=50*scale; % 抽样时间间隔，50ns
t_rms=25*scale; % RMS 实验扩展，25ns
num_ch=10000; % 信道数
N=128; % FFT长度
PDP=IEEE802_11_model(t_rms,Ts);调用IEEE802_11_model子程序，计算IEEE802.11信道的PDP。
for k=1:length(PDP)
h(:,k) = Ray_model(num_ch).*sqrt(PDP(k));
avg_pow_h(k)= mean(h(:,k).*conj(h(:,k)));
end
H=fft(h(1,:),N);
subplot(121)
stem([0:length(PDP)-1],PDP,'ko'), hold on,
stem([0:length(PDP)-1],avg_pow_h,'k.');
xlabel('channel tap index, p'), ylabel('Average Channel Power[linear]');
title('IEEE 802.11 Model, \sigma_\tau=25ns, T_S=50ns');
legend('Ideal','Simulation'); axis([-1 7 0 1]);
subplot(122)
plot([-N/2+1:N/2]/N/Ts/1e6,10*log10(H.*conj(H)),'k-');
xlabel('Frequency[MHz]'), ylabel('Channel power[dB]')
title('Frequency response, \sigma_\tau=25ns, T_S=50ns')

% IEEE 802.11 信道模型 PDP 产生子程序，子程序名称：IEEE802_11_model.m
function PDP=IEEE802_11_model(sigma_t,Ts)
% 输入参数:
% sigma_t : RMS 延时扩展
% Ts : 抽样时间间隔
% 输出参数:
% PDP : IEEE 802.11 信道PDP矩阵
lmax = ceil(10*sigma_t/Ts); % 计算最大路径序号，参见式(2-38)
sigma02=(1-exp(-Ts/sigma_t))/(1-exp(-(lmax+1)*Ts/sigma_t)); % 参见式(2-41)
l=0:lmax; PDP = sigma02*exp(-l*Ts/sigma_t); % 参见式(2-40)

仿真结果参见图2-11。

实例2-4 滤波白噪声模型
功能：产生滤波白噪声信道模型，画出信道幅度系数及其包络的概率密度函数与相位的概率密度函数。
程序名称：Example2_4.m
程序代码：
clear, clf
fm=100;%最大多普勒频率
 scale=1e-6; % 微秒量级 
ts_mu=50; ts=ts_mu*scale; fs=1/ts; % 抽样时间与抽样频率
Nd=1e6; % 抽样个数
% 获得复信道系数
[h,Nfft,Nifft,doppler_coeff] = FWGN_model(fm,fs,Nd);
subplot(211)
 plot([1:Nd]*ts,10*log10(abs(h)))%画信道幅度系数
str=sprintf('Clarke/Gan Model, f_m=%d[Hz], T_s=%d[us]',fm,ts_mu);
title(str), axis([0 0.5 -30 5])
subplot(223)
hist(abs(h),50)%画信道包络概率密度函数，包络一维概率密度函数呈瑞利分布。
 subplot(224)
 hist(angle(h),50)%画信道相位概率密度函数，相位一维概率密度函数呈均匀分布。

% 滤波白噪声信道子程序 (Clarke/Gan模型)，子程序名称：FWGN_model.m
function [h,Nfft,Nifft,doppler_coeff]=FWGN_model(fm,fs,N)
% 输入参数: fm为最大多普勒频率，fs为抽样频率，N为抽样个数
% 输出参数: h为复信道系数
Nfft = 2^nextpow2(2*fm/fs*N);
Nifft = ceil(Nfft*fs/(2*fm));
% 产生独立的复高斯随机过程，参见图2-12
GI = randn(1,Nfft); GQ = randn(1,Nfft);
% 求实信号的FFT，以获得Hermitian对称
CGI = fft(GI); CGQ = fft(GQ);%将高斯随机过程转换到频域。
% 多普勒谱产生 ，仿真多普勒滤波器
doppler_coeff = Doppler_spectrum(fm,Nfft);
% 将转换到频域的高斯过程加入到多普勒滤波器，频域内乘积。
f_CGI = CGI.*sqrt(doppler_coeff); f_CGQ = CGQ.*sqrt(doppler_coeff);
% 补零，使多普勒滤波器输出数据长度为Nifft，以备求ifft。
Filtered_CGI=[f_CGI(1:Nfft/2) zeros(1,Nifft-Nfft) f_CGI(Nfft/2+1:Nfft)];
Filtered_CGQ=[f_CGQ(1:Nfft/2) zeros(1,Nifft-Nfft) f_CGQ(Nfft/2+1:Nfft)];
%求ifft，将多普勒滤波器输出频域信号转换到时域
hI = ifft(Filtered_CGI); hQ= ifft(Filtered_CGQ);
% 计算实部的平方加虚部的平方开平方，即呈瑞利分布的包络
rayEnvelope = sqrt(abs(hI).^2 + abs(hQ).^2);
% 计算包络的均方根值
rayRMS = sqrt(mean(rayEnvelope(1:N).*rayEnvelope(1:N)));
%图2-12最终输出的Clarke/Gan模型信道冲激响应。
h = complex(real(hI(1:N)),-real(hQ(1:N)))/rayRMS;

%多普勒谱子程序，子程序名称：Doppler_spectrum.m
%对经典多普勒谱的仿真，参见式（2-25）。
function y=Doppler_spectrum(fd,Nfft)
%输入参数： fd为最大多普勒频移，Nfft为频域样值点个数。
%输出参数： y返回多普勒谱
df = 2*fd/Nfft; % 计算频率间隔
% 计算f=0时多普勒谱
f(1) = 0; y(1) = 1.5/(pi*fd);
% 计算其他频率多普勒谱。计算了多普勒谱从第2个样值到Nfft/2个样值以及第Nfft/2+2个样值到第Nfft个%样值。
for i = 2:Nfft/2,
f(i)=(i-1)*df; % 根据频率序号及频率间隔计算频率
y([i Nfft-i+2]) = 1.5/(pi*fd*sqrt(1-(f(i)/fd)^2));
%计算式（2-25），注意randn产生的高斯噪声方差为1。
end
% 计算多普勒谱的Nfft/2+1点样值。用四点样值构成多项式，之后求出多项式在Nfft/2+1点的样值。这个多%重相当于内插。
nFitPoints=3 ; kk=[Nfft/2-nFitPoints:Nfft/2];%四点样值序号为Nfft/2-3到Nfft/2
polyFreq = polyfit(f(kk),y(kk),nFitPoints);%形成多项式
y((Nfft/2)+1) = polyval(polyFreq,f(Nfft/2)+df);%从多项式求Nfft/2点值。

仿真结果见图2-13。


3.7 仿真实例
实例3-1 OFDM信号的产生与解调
功能：
（1）通过对OFDM信号各个子载波赋共轭对称的数据产生一个实OFDM符号；
（2）给OFDM符号加循环前缀与循环后缀；
（3）给OFDM符号加窗。在程序中加入的是升余弦窗，可以通过改变升余弦窗的滚降系数观察加入不同升余弦窗，对OFDM信号频谱的影响；
（4）信道采用加性高斯白噪声信道。可以通过改变信噪比改变信道环境，从而在接收端通过误码率或星座图观察信道对OFDM信号传输的影响；
（5）去除循环前缀与循环后缀，对OFDM信号进行解调。
程序名称：Example3_1.m
程序代码：
clear all;
close all;
carrier_count=200;
%这个程序中OFDM子载波个数为512，其中400即carrier_count*2为数据符号，其余赋0值。
symbols_per_carrier=20;%每个子载波上的符号数，在这里即为OFDM符号的个数。
bits_per_symbol=4;%OFDM符号的每个子载波上传输的比特数。4比特通常采用16QAM调制。
IFFT_bin_length=512;%FFT长度，也即一个OFDM符号的子载波的个数。
PrefixRatio=1/4;%循环前缀的比率，即循环前缀与OFDM符号长度的比值，通常在 1/6~1/4之间。
GI=PrefixRatio*IFFT_bin_length ;%保护间隔的长度，这里为128。
beta=1/32;%升余弦窗的滚降系数。
GIP=beta*(IFFT_bin_length+GI);%循环后缀的长度，这里为20
SNR=30; %本程序考虑加性高斯白噪声信道，这里信噪比为30dB。

%===============================OFDM信号产生=============================
baseband_out_length = carrier_count * symbols_per_carrier * bits_per_symbol;
%计算传输数据总的比特数，为200*20*4=16000比特。16000比特的构成为20个OFDM符号，每个OFDM
%符号200个子载波，每个子载波传输4比特信息。
carriers=(1:carrier_count)+(floor(IFFT_bin_length/4) –floor(carrier_count/2));
%计算OFDM符号子载波的序号，carriers中存放的序号是29~228。 
conjugate_carriers = IFFT_bin_length - carriers + 2;
%计算OFDM符号子载波的序号，conjugate_carriers中存放的序号是282~481。
rand( 'twister',0);
baseband_out=round(rand(1,baseband_out_length));
%产生16000比特待传输的二进制比特流。这里存放的是发送的二进制信号与后面解调后的二进制信号比
%较，可以计算误码率。

%16QAM调制并绘制星座图
complex_carrier_matrix=qam16(baseband_out);
%调用子程序qam16进行16QAM调制。将baseband_out中的二进制比特流，每4比特转换为一个16QAM信
%号，即将二进制比特流每4比特转换为-3-3j、-3+3j、3-3j、3+3j、-1-3j、-1+3j、1-3j、1+3j、%-3-j、-3+j、3-j、3+j、-1-j、-1+j、1-j、1+j中的一个。转换后complex_carrier_matrix为%1*4000矩阵。
complex_carrier_matrix=reshape…
(complex_carrier_matrix',carrier_count,symbols_per_carrier)';
%转换complex_carrier_matrix中的数据为carrier_count*symbols_per_carrier矩阵，这里为%20*200矩阵。
figure(1);
plot(complex_carrier_matrix,'*r');% 绘制16QAM星座图
axis([-4, 4, -4, 4]);
title('16QAM调制后星座图');
grid on

%IFFT，即进行OFDM调制。
IFFT_modulation=zeros(symbols_per_carrier,IFFT_bin_length);
%将symbols_per_carrier*IFFT_bin_length矩阵赋0值，这里将20*512矩阵赋0值。这里512是%IFFT的长度，也是OFDM符号子载波的个数。
 IFFT_modulation(:,carriers ) = complex_carrier_matrix ;
%将20*200的complex_carrier_matrix的数据赋给IFFT_modulation的第29~228列，即给512个子%载波中的29~229个子载波赋值。
IFFT_modulation(:,conjugate_carriers ) = conj(complex_carrier_matrix);
%将20*200的complex_carrier_matrix的数据赋给512个子载波中的第282~481个子载波。
%这段程序构造了512个子载波的OFDM符号，并且各个子载波上的数据是共轭对称的。这样做的目的是经过%IFFT后形成的OFDM符号均为实数。另外，在512个子载波中，仅有400个子载波为数据，其余为0值。相%当于补零，补零的目的是通常IFFT的长度应该为2的整数次幂。
signal_after_IFFT=ifft(IFFT_modulation,IFFT_bin_length,2);%IFFT实现OFDM调制。
time_wave_matrix=signal_after_IFFT;%
figure(2);
plot(0:IFFT_bin_length-1,time_wave_matrix(2,:));%画一个OFDM信号的时域表现
axis([0, 512, -0.4, 0.4]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal, One Symbol Period');

%添加循环前缀与循环后缀
XX=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);
%IFFT_bin_length+GI+GIP为OFDM、循环前缀、循环后缀长度之和。
for k=1:symbols_per_carrier;
        for i=1:IFFT_bin_length;
            XX(k,i+GI)=signal_after_IFFT(k,i);
        end
        for i=1:GI;
            XX(k,i)=signal_after_IFFT(k,i+IFFT_bin_length-GI);%添加循环前缀
        end
        for j=1:GIP;
            XX(k,IFFT_bin_length+GI+j)=signal_after_IFFT(k,j);%添加循环后缀
        end
end
time_wave_matrix_cp=XX;%带循环前缀与循环后缀的OFDM符号。
figure(3);
plot(0:length(time_wave_matrix_cp)-1,time_wave_matrix_cp(2,:));
%画带循环前缀与循环后缀的OFDM信号的时域波形 
axis([0, 600, -0.3, 0.3]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal with CP, One Symbol Period');
%OFDM符号加窗
windowed_time_wave_matrix_cp=zeros(1,IFFT_bin_length+GI+GIP);
for i = 1:symbols_per_carrier 
windowed_time_wave_matrix_cp(i,:) =… real(time_wave_matrix_cp(i,:)).*rcoswindow(beta,IFFT_bin_length+GI)';
%调用rcoswindow产生升余弦窗，对带循环前缀与循环后缀的OFDM符号加窗。
end  
figure(4);
plot(0:IFFT_bin_length-1+GI+GIP,windowed_time_wave_matrix_cp(2,:));
%画加窗后的OFDM符号
axis([0, 700, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal Apply a Window , One Symbol Period');

%生成发送信号，并串变换
windowed_Tx_data=zeros(1,symbols_per_carrier*(IFFT_bin_length+GI)+GIP);
%注意并串变换后数据的长度为symbols_per_carrier*(IFFT_bin_length+GI)+GIP，这里考虑了循%环前缀与循环后缀的重叠相加。
windowed_Tx_data(1:IFFT_bin_length+GI+GIP)=windowed_time_wave_matrix_cp(1,:);
%赋第一个加窗带循环前缀后缀的OFDM符号至windowed_Tx_data，即发送串行数据。
for i = 1:symbols_per_carrier-1 ;
windowed_Tx_data((IFFT_bin_length+GI)*i+1:(IFFT_bin_length+GI)*(i+1)+GIP)=…
windowed_time_wave_matrix_cp(i+1,:);%并串变换，循环前缀与循环后缀重叠相加
end
Tx_data_withoutwindow=reshape… (time_wave_matrix_cp',(symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';
%不加窗数据并串变换
Tx_data=reshape(windowed_time_wave_matrix_cp',… (symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';
%加窗数据，但按照循环前缀与循环后缀不重叠相加进行并串变换。此时数据长度为%(symbols_per_carrier)*(IFFT_bin_length+GI+GIP)。
temp_time1 = (symbols_per_carrier)*(IFFT_bin_length+GI+GIP);
%加窗，循环前缀与循环后缀不重叠数据长度，即为发送的总的数据比特数
figure (5)
subplot(2,1,1);
plot(0:temp_time1-1,Tx_data );%画循环前缀与循环后缀不重叠相加OFDM信号的时域波形
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM Time Signal')
temp_time2 =symbols_per_carrier*(IFFT_bin_length+GI)+GIP;
%加窗，循环前缀与循环后缀重叠相加数据长度
subplot(2,1,2);
plot(0:temp_time2-1,windowed_Tx_data);
%画循环前缀与循环后缀重叠相加OFDM信号的时域波形

grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM Time Signal')

%未加窗发送信号频谱
%对发送数据分段，分段计算频谱，取平均作为OFDM信号的频谱
symbols_per_average = ceil(symbols_per_carrier/5);% 
avg_temp_time = (IFFT_bin_length+GI+GIP)*symbols_per_average;% 
averages = floor(temp_time1/avg_temp_time);
%将发送数据分5段，每段数据长度为avg_temp_time 
average_fft(1:avg_temp_time) = 0;% 存放平均后的OFDM信号的谱，先置零。
for a = 0:(averages-1)
        subset_ofdm =Tx_data_withoutwindow… (((a*avg_temp_time)+1):((a+1)*avg_temp_time));%分段
    subset_ofdm_f = abs(fft(subset_ofdm));% 对分段后的OFDM信号计算频谱
    average_fft = average_fft + (subset_ofdm_f/averages);% 取平均
end
average_fft_log = 20*log10(average_fft);求对数平均谱
figure (6)
plot((0:(avg_temp_time-1))/avg_temp_time, average_fft_log)
% 画未加窗OFDM符号对数平均谱
hold on
grid on
axis([0 0.5 -20 max(average_fft_log)])
ylabel('Magnitude (dB)')
xlabel('Normalized Frequency (0.5 = fs/2)')
title('OFDM Signal Spectrum ')

%计算加窗OFDM信号的频谱
%这段程序与上段程序类似，不同之处在与这段程序是对加窗OFDM信号进行分段计算频谱，再取平均。
symbols_per_average = ceil(symbols_per_carrier/5);  
avg_temp_time = (IFFT_bin_length+GI+GIP)*symbols_per_average; 
averages = floor(temp_time1/avg_temp_time);
%将发送数据分5段，每段数据长度为avg_temp_time 
average_fft(1:avg_temp_time) = 0;%存放平均后的OFDM信号的谱，先置零。 
for a = 0:(averages-1)
       subset_ofdm = Tx_data(((a*avg_temp_time)+1):((a+1)*avg_temp_time));%分段  
 subset_ofdm_f = abs(fft(subset_ofdm));% 对分段后的OFDM信号计算频谱
    average_fft = average_fft + (subset_ofdm_f/averages);% 取平均
end
average_fft_log = 20*log10(average_fft);%求对数平均谱
subplot(2,1,2)
plot((0:(avg_temp_time-1))/avg_temp_time, average_fft_log)
% 画加窗OFDM信号对数平均谱
hold on
grid on
axis([0 0.5 -20 max(average_fft_log)])
ylabel('Magnitude (dB)')
xlabel('Normalized Frequency (0.5 = fs/2)')
title('Windowed OFDM Signal Spectrum')

%====================经过加性高斯白噪声信道=======----================== 
Tx_signal_power = var(windowed_Tx_data);% 计算信号功率
linear_SNR=10^(SNR/10);% 转换对数信噪比为线性幅度值
noise_sigma=Tx_signal_power/linear_SNR;%计算噪声功率，也就是方差
noise_scale_factor = sqrt(noise_sigma);% 计算标准差
noise=randn(1,((symbols_per_carrier)*(IFFT_bin_length+GI))…
+GIP)*noise_scale_factor;% 产生功率为noise_scale_factor高斯噪声
Rx_data=windowed_Tx_data +noise
%在发送数据上加噪声，相当于OFDM信号经过加性高斯白噪声信道。

%==========================OFDM信号解调=============================== 
Rx_data_matrix=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);
%存放并行的接收数据
for i=1:symbols_per_carrier;   Rx_data_matrix(i,:)=Rx_data(1,(i-1)*(IFFT_bin_length+GI)…
+1:i*(IFFT_bin_length+GI)+GIP);% 串并变换
end
Rx_data_complex_matrix=Rx_data_matrix(:,GI+1:GI+IFFT_bin_length);
% 去掉循环前缀与循环后缀，取出OFDM符号传输的数据
Y1=fft(Rx_data_complex_matrix,IFFT_bin_length,2);% 求FFT，即OFDM信号解调
Rx_carriers=Y1(:,carriers);
% 取出carriers序号对应的子载波上的发送数据，去掉加入的零及共轭对称部分
Rx_phase =angle(Rx_carriers);% 计算接收信号的相位特性
Rx_mag = abs(Rx_carriers);% 计算接收信号的幅度特性
[M, N]=pol2cart(Rx_phase, Rx_mag);转换极坐标数据为直角坐标数据 
    Rx_complex_carrier_matrix = complex(M, N);两个直角坐标的实数据为构成复数据。
figure(7);
plot(Rx_complex_carrier_matrix,'*r');%画接收信号的星座图 
axis([-4, 4, -4, 4]);
title('SNR=30dB接收数据星座图');
grid on
%16QAM解调
Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',… size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
%将矩阵Rx_complex_carrier_matrix转换为1为的数组
   Rx_decoded_binary_symbols=demoduqam16(Rx_serial_complex_symbols);%进行16QAM解调
baseband_in = Rx_decoded_binary_symbols;%将解调恢复的二进制信号存放在baseband_in
%误码率计算
bit_errors=find(baseband_in ~=baseband_out);
%解调恢复的二进制信号与发送二进制信号比较，查找误码
bit_error_count = size(bit_errors, 2) %计算误码个数
ber=bit_error_count/baseband_out_length%计算误码率

%16QAM调制子程序，子程序名称：qam16.m
%将二进制数目流转换为16QAM信号
function [complex_qam_data]=qam16(bitdata)
%输入参数：bitdata为二进制数码流
%输出参数：complex_qam_data为16QAM复信号
X1=reshape(bitdata,4,length(bitdata)/4)';%将二进制数码流以4比特分段
d=1;
%转换4比特二进制码为十进制码1~16，生成mapping映射表中的索引。
for i=1:length(bitdata)/4;
    for j=1:4
        X1(i,j)=X1(i,j)*(2^(4-j)); 
    end
        source(i,1)=1+sum(X1(i,:)); 
end
%16QAM映射表，该表中存放的是16对，每对两个实数，标识星座位置。
mapping=[-3*d 3*d; -d  3*d;d  3*d;3*d  3*d;-3*d  d; -d  d; d  d;3*d  d;…
 -3*d  -d; -d  -d; d  -d;3*d  -d;-3*d  -3*d;-d  -3*d; d  -3*d;3*d  -3*d];
    for i=1:length(bitdata)/4
        qam_data(i,:)=mapping(source(i),:);%数据映射
    end
complex_qam_data=complex(qam_data(:,1),qam_data(:,2));
%组合为复数形式，形成16QAM信号。

%窗函数子程序，子程序名称：rcoswindow.m
function [rcosw]=rcoswindow(beta, Ts) 
%输入参数：beta为升余弦窗滚降系数，Ts为IFFT长度加循环前缀长度
t=0:(1+beta)*Ts;
rcosw=zeros(1,(1+beta)*Ts);
%计算升余弦窗，共有三部分
for i=1:beta*Ts;
rcosw(i)=0.5+0.5*cos(pi+ t(i)*pi/(beta*Ts));%计算升余弦窗第一部分
end
rcosw(beta*Ts+1:Ts)=1;%升余弦窗第二部分
for j=Ts+1:(1+beta)*Ts+1;
        rcosw(j-1)=0.5+0.5*cos((t(j)-Ts)*pi/(beta*Ts));%计算升余弦窗第三部分
end
rcosw=rcosw';% 转换为列向量

%16QAM信号的解调子程序，子程序名称：demoduqam16.m
%转换16QAM信号为二进制信号
function [demodu_bit_symble]=demoduqam16(Rx_serial_complex_symbols)
%输入参数：Rx_serial_complex_symbols为接收端接收到的复16QAM信号
%输出参数：demodu_bit_symble为二进制数码流
complex_symbols=reshape(Rx_serial_complex_symbols,…
length(Rx_serial_complex_symbols),1);
d=1;
mapping=[-3*d 3*d;-d  3*d;d  3*d;3*d  3*d; -3*d  d; -d  d;d  d;3*d  d;…
-3*d  -d;  -d  -d;  d  -d; 3*d  -d;-3*d  -3*d; -d  -3*d; d  -3*d;  3*d  -3*d];
complex_mapping=complex(mapping(:,1),mapping(:,2));
%将数据映射表转换为16QAM信号，即３组合为复数。
    for i=1:length(Rx_serial_complex_symbols);
         for j=1:16;
             metrics(j)=abs(complex_symbols(i,1)-complex_mapping(j,1));
         end
         [min_metric  decode_symble(i)]= min(metrics) ;
% 将接收数据与标准16QAM信号比，找到差最小的，将其对应恢复成标准的16QAM信号
    end
    decode_bit_symble=de2bi((decode_symble-1)','left-msb');%16QAM转为二进制
demodu_bit_symble=reshape(decode_bit_symble',1,…
length(Rx_serial_complex_symbols)*4);%转换为一行

仿真结果：
(1)发送端发送16QAM信号星座图，见图3-16。
        
图3-16 发送16QAM信号星座图                 图3-17 一个周期的OFDM信号时域波形
（2）OFDM信号时域波形，见图3-17。
（3）带循环前缀与后缀的OFDM符号时域波形，见图3-18。
         
图3-18 带循环前缀与后缀的一个周期的              3-19 加窗的带循环前缀与后缀的一个周
OFDM信号时域波形                             期的OFDM信号时域波形图
（4）加窗的带循环前缀与后缀的OFDM信号时域波形，见图3-19。
（5）循环前缀与循环后缀叠加与不叠加OFDM信号时域波形图，见图3-20。
  
 图3-20  循环前缀与循环后缀不叠加与循环前缀           图3-21  OFDM信号频域与加窗的
      与循环后缀叠加OFDM信号时域波形图                       OFDM信号的频谱
（6）加窗与不加窗的OFDM信号信号的频谱，见图3-21。
由频谱可以看出加窗对带外辐射的抑制。
（7）解调后的16QAM信号星座图，见图3-22、图3-23、图2-24。
        
图3-22  SNR=15dB接收数据星座图              图3-23 SNR=20dB接收数据星座图

           
图3-24  SNR=30dB接收数据星座图                 图3-25 误码率曲线
    （8）误码率曲线仿真，见图3-25。



4.6 仿真实例
实例4_1  发送端未知信道状态信息情况下MIMO系统容量仿真
功能：发送端与接受端天线个数不同的情况下，MIMO系统的容量。
程序名称：Example4_1.m
程序代码：
clear all;
close all;
M = 1000;%循环次数
n_bins = round(M/10);
Nt = [2:2:8];%发送端天线个数。
SNR = [10]; %信噪比，单位为dB
figure(1);
title('发送端未知CSI情况下MIMO系统容量的CCDF', 'FontSize',14)
xlabel('容量（bit/s/Hz）', 'FontSize',12);
ylabel('Pr(容量>=横坐标)', 'FontSize',12);
hold on;
text(1.8,0.8,'1×1', 'FontSize',12);text(3,0.9,'2×2', 'FontSize',12);
text(6,0.8,'2×4', 'FontSize',12);text(8,0.7,'2×6', 'FontSize',12);
text(10.4,0.2,'2×8', 'FontSize',12);text(13,0.1,'4×4', 'FontSize',12);
text(19,0.1,'6×6', 'FontSize',12);text(24,0.1,'8×8', 'FontSize',12);
%画1发1收时，即SISO系统的容量
for n = 1:1,
for m = 1:M,	
     %产生瑞利信道，信道矩阵参见式（2-45）。
            H = (randn(1,1)+1i*randn(1,1))/sqrt(2);
            rho = 10^(SNR/10);%转换信噪比为幅值，而非dB值描述。
            % 计算发送端未知信道状态信息情况下的容量，参见式4-10。
            CU(m,n) = log2(real(det(eye(1)+rho*H*H')));
end
[cdf,c] = hist(CU(:,n),n_bins);%统计容量分布
plot(c,1-(cumsum(cdf))/M)%画M个循环得到的容量的平均值，用CCDF描述。
end
%画收、发天线个数为2×4、2×6、2×8时的容量
for n = 2:length(Nt),
N = Nt(n);
for m = 1:M,
          H = (randn(2,N)+1i*randn(2,N))/sqrt(2);
          rho = 10^(SNR/10); 
          % 计算发送端未知信道状态信息情况下的容量，参见式4-10。
          CU(m,n) = log2(real(det(eye(2)+rho*H*H'/2)));
end
[cdf,c] = hist(CU(:,n),n_bins);
plot(c,1-(cumsum(cdf))/M)
end
%画收、发天线个数为2×2、4×4、6×6、8×8时的容量
for n = 1:length(Nt),
N = Nt(n);
for m = 1:M,
           H = (randn(N,N)+1i*randn(N,N))/sqrt(2);
           rho = 10^(SNR/10);   
            % 计算发送端未知信道状态信息情况下的容量，参见式4-10。
           CU(m,n) = log2(real(det(eye(N)+rho*H*H'/N)));
end
[cdf,c] = hist(CU(:,n),n_bins);
plot(c,1-(cumsum(cdf))/M)
end
   grid on;
仿真结果如4-17所示。
 
图4-17 发送端未知CSI时MIMO系统容量CCDF仿真图
实例4_2  MIMO系统平均容量与中断容量仿真
功能：（1）发送端已知状态信息与发送端未知信道状态信息时，MIMO系统平均容量随信噪比变化曲线仿真；
      （2）发送端已知状态信息与发送端未知信道状态信息时，MIMO系统中断容量随信噪比变化曲线仿真；
（3）发送端已知状态信息与发送端未知信道状态信息时，MIMO系统平均容量随天线数变化曲线仿真；
程序名称：Example4_2.m
程序代码：
clear all;
close all;
clc;
M = 1000;%循环数
Nt = [2:2:8];%发送端天线个数
   SNR = [0:2:20]; %信噪比，单位为dB
figure(1);
title('MIMO系统平均容量  vs SNR','fontsize',18)
xlabel('SNR [dB]','fontsize',18);
ylabel('容量(bps/Hz)','fontsize',18);
grid on
hold on;
figure(2);
title('MIMO系统10%中断容量  vs SNR','fontsize',18)
xlabel('SNR [dB]','fontsize',18);
ylabel('中断容量(bps/Hz)','fontsize',18);
grid on
hold on;
for n = 1:length(Nt),
        N = Nt(n);
        for m = 1:M,
            H = (randn(N,N)+1i*randn(N,N))/sqrt(2);%产生瑞利信道
            for snr_idx = 1:length(SNR),
               rho = 10^(SNR(snr_idx)/10); 
               % 发送端未知信道状态信息情况下，MIMO系统平均容量，参见式（4-10）
               CU(m,snr_idx) = log2(real(det(eye(N)+rho*H*H'/N)));
               % 发送端已知信道状态信息情况下，MIMO系统平均容量，参见式（4-16）
               [gamma,eigs] = pwr_modes(H,rho);%计算式（4-16）中的gamma和奇异值。
               CK(m,snr_idx) = sum(log2(real(1+eigs.*gamma*rho/N)));
            end
        end
        C_unknown(:,n) = mean(CU)';%对M个未知CSI的MIMO系统容量取均值
        C_known(:,n) = mean(CK)';%对M个已知CSI的MIMO系统容量取均值
        %统计中断容量
        for snr_idx = 1:length(SNR),
            [cdf_u,co_u] = hist(CU(:,snr_idx),100);
            cdf_u = cumsum(cdf_u);
            idx_ten_percent = find(abs(cdf_u-100)==min(abs(cdf_u-100)));
            C_unknown_outage(snr_idx,n) = co_u(idx_ten_percent(1));
            [cdf_k,co_k] = hist(CK(:,snr_idx),100);
            cdf_k = cumsum(cdf_k);
            idx_ten_percent = find(abs(cdf_k-100)==min(abs(cdf_k-100)));
            C_known_outage(snr_idx,n) = co_k(idx_ten_percent(1));
        end
        figure(1);
        plot(SNR,C_unknown(:,n),'k-');
        plot(SNR,C_known(:,n),'k-.');
        figure(2);
        plot(SNR,C_unknown_outage(:,n),'k-');
        plot(SNR,C_known_outage(:,n),'k-.');
        if n == 1
            figure(1)
            legend('\fontsize{16}未知CSI',''\fontsize{16}已知CSI',2);
            text(15,10,'2×2', 'FontSize',18);text(15,19,'4×4', 'FontSize',18);
text(15,27,'6×6', 'FontSize',18);text(15,36,'8×8', 'FontSize',18);
            figure(2)
            legend('\fontsize{16}未知CSI','\fontsize{16}已知CSI',2);
            text(15,8,'2×2', 'FontSize',18);text(15,17,'4×4', 'FontSize',18);
text(15,25,'6×6', 'FontSize',18);text(15,34,'8×8', 'FontSize',18);
        end
end
hold off
figure
plot(Nt,C_unknown(6,:),'k-')
hold on
plot(Nt,C_known(6,:),'k-.')
title('MIMO系统信道容量 vs 天线','fontsize',18)
xlabel('天线数','fontsize',18)
ylabel('容量(bps/Hz)','fontsize',18);
legend('\fontsize{16}Î´未知CSI','\fontsize{16}已知CSI',2);
grid on

     %计算gamma和奇异值子程序，子程序名称：pwr_modes.m
function [g,l] = pwr_modes(H,rho)
%输入参数：H为信道矩阵，rho为信噪比。
%输出参数：g为gamma值，l为奇异值eigs。
%这部分程序参看注水算法。
N_tilde = size(H,1);
l_tilde = real(eig(H*H'));
l = l_tilde(find(l_tilde~=0));
N = length(l);
mu = (N_tilde+sum(N_tilde./(rho*l)))/N;
g = mu-N_tilde./(rho*l);
while (length(find(g <= 0)) ~= 0)
        l = l(find(g > 0));
        N = length(l);
        mu = (N_tilde+sum(N_tilde./(rho*l)))/N;
        g = mu-N_tilde./(rho*l);
end
仿真结果如4-18（a）、（b）、（c）所示。
 
（a）MIMO系统平均容量 vs SNR曲线     
 
    
（b）MIMO系统10%中断容量 vs SNR曲线
 
（c）MIMO信道容量 vs天线数曲线
图4-18 实例4_2仿真结果图
实例4_3  相关对MIMO系统容量影响仿真
clear all
close all
clc
M = 5000;
R = [0.2  0.95];相关矩阵
SNR = [0:2:20];
figure;
xlabel('SNR [dB]', 'FontSize',18);
ylabel('容量（bit/s/Hz）', 'FontSize',18);
title('相关信道的容量比较' ,'FontSize',18)
 grid on
hold on;
for l = 1:length(R),
        R_t = eye(2);
        R_r = [1 R(l);R(l) 1]
        for snr_idx = 1:length(SNR),
            snr = 10^(SNR(snr_idx)/10);
            for m = 1:M,
                Hw = (randn(2,2)+1i*randn(2,2))/sqrt(2);
                H = R_r^(.5)*Hw*R_t^(.5);
                C(m,snr_idx)=log2((det(eye(2)+snr*H*H'/2)));
            end
            Capacity(snr_idx,l) = mean(C(:,snr_idx));
        end
   
end
 plot(SNR,Capacity(:,1),'k-',SNR,Capacity(:,3),'kx-');
for l = 1:length(R),
        R_t = eye(2);
        R_r = [1 R(l);R(l) 1]
        for snr_idx = 1:length(SNR),
            snr = 10^(SNR(snr_idx)/10);
            for m = 1:M,
                Hw = (randn(2,2)+1i*randn(2,2))/sqrt(2);
                H = R_r^(.5)*Hw*R_t^(.5);
                [gamma,eigs] = pwr_modes(H,snr);
                C(m,snr_idx) = sum(log2(real(1+eigs.*gamma*snr/2)));
            end
            Capacity(snr_idx,l) = mean(C(:,snr_idx));
        end
end
 plot(SNR,Capacity(:,1),'k-.',SNR,Capacity(:,3),'kd-');
legend('R=0 CSI未知','R=0.95 CSI未知','R=0 CSI已知','R=0.95 CSI已知',2);

仿真结果如图4-19所示。



5.5 仿真实例
实例5_1  基于特殊训练序列的信道估计与误码率仿真
功能：（1）MIMO-OFDM系统在多径信道下的LS估计；
     （2）给出估计的最小均方误差（MSE）曲线；
     （3）给出LS估计下系统的误码率（SER）曲线。
信道：在本程序中采用的信道是多径信道，信道的冲激响应为
                           （5-75）
其中 为离散多普勒频移；
 为最大多普勒频移；
 为离散多普勒相位；
 为独立随机变量。
L为多径信道的径数；M为调和系数； 与 如表5-2所示。
表5-2 多径信道参数
路径序号
 
时间延迟
 ( )
路径能量
 

1	0	1.0
2	50	0.6095
3	100	0.4945
4	150	0.3940
5	200	0.2371
6	250	0.1900
7	300	0.1159
8	350	0.0699
9	400	0.0462

2发2收MIMO-OFDM系统LS估计原理：
若收、发端信号用式（5-76）表示
                                        （5-76）
其中 是第j个天线上接收的信号且N是OFDM符号子载波的数量；
 且 ， 是以 的元素为对角线的对角阵； 为第i个发送天线上发生的块状导频信号， 。
 且 ， 为2发2收MIMO-OFDM系统信道的频域特性；
 为每个接收天线上的加性高斯白噪声矩阵；
又信道时域特性与频域特性之间的关系为
                                                                （5-77）
其中 为信道的时域冲激响应， 为信道时域冲激响应长度，并且
 
 为 的前L列构成的矩阵。
将（5-77）式代入（5-76）式有
                                                    （5-78）其中   ， 。
对（5-78）进行LS准则运算，在 满秩的情况下，可得信道时域冲激响应的估计值
                                                （5-79）
程序名称：Example5_1.m
程序代码：
%OFDM系统参数设置
clc;
clear all;
NFFT = 64;             % FFT 长度
G = 0;                  % 保护间隔长度
M_ary =4;              % 进制数
P_A = sqrt(2);        % 导频符号的幅度
D_t = 4;               % 时域内导频序列的间隔，即一个训练序列与另一训练序列的间隔
t_a = 50*10^(-9);       % HiperLAN/2中的抽样间隔
%蒙特卡洛信道参数设置symbol_duration = NFFT * t_a;   %一个OFDM信号的时间
number_of_summations = 40;       % 蒙特卡洛方法中的调和系数
f_dmax = 50.0;                     % 最大多普勒频移
load h_decimation.am -ascii;%数据文件，存多径信道的 与 参数。
h11_initial = h_decimation;     %h11：发送天线1到接收天线1信道系数
h12_initial = h_decimation;	    %h12：发送天线1到接收天线2信道系数
h21_initial = h_decimation;     %h21：发送天线2到接收天线1信道系数
h22_initial = h_decimation;     %h22：发送天线2到接收天线2信道系数
N_P  = length(h_decimation);
NofOFDMSymbol = 1000;        %OFDM符号个数，包括导频OFDM符号与数据OFDM符号
No_Of_OFDM_Data_Symbol = NofOFDMSymbol-ceil(NofOFDMSymbol/D_t);
                                     %数据OFDM符号个数
length_data = (No_Of_OFDM_Data_Symbol) * NFFT;  
                            % 总的数据长度，例如：若采用4QAM则是4进制数据的长度
Number_Relz = 100;%循环数
ser_relz = [];%存每个循环的误码率
for number_of_relialization= 1: Number_Relz;  %循环开始 
%产生信道中的随机系数
u11 = rand(N_P,number_of_summations); 
u12 = rand(N_P,number_of_summations);  
u21 = rand(N_P,number_of_summations);  
u22 = rand(N_P,number_of_summations);  
% 发送的二进制比特流
source_data1 = randint(length_data,2);
source_data2 = randint(length_data,2);
% 转换为四进制信号
symbols1 = bi2de(source_data1);  
symbols2 = bi2de(source_data2);  
% 进行4QAM映射
QASK_Symbol1 = dmodce(symbols1,1,1,'qask',M_ary);%存放第1个天线发送的串行数据
QASK_Symbol2 = dmodce(symbols2,1,1,'qask',M_ary);%存放第2个天线发送的串行数据
% 天线1IFFT前的信号，为IFFT点的并行数据，即将串行数据转换为并行数据
Data_Pattern1 = []; % 为IFFT准备输入信号
m = 0;
for i=0:No_Of_OFDM_Data_Symbol-1;
            QASK_tem = [];
            for n=1:NFFT;
                 QASK_tem = [QASK_tem,QASK_Symbol1(i*NFFT+n)];
            end;
            Data_Pattern1 = [Data_Pattern1;QASK_tem];
            clear QASK_tem;
end;
% 天线2IFFT前的信号，为IFFT点的并行数据，即将串行数据转换为并行数据
Data_Pattern2 = [];  
m = 0;
for i=0:No_Of_OFDM_Data_Symbol-1;
            QASK_tem = [];
            for n=1:NFFT;
                QASK_tem = [QASK_tem,QASK_Symbol2(i*NFFT+n)];
            end;
            Data_Pattern2 = [Data_Pattern2;QASK_tem];
            clear QASK_tem;
end;
% 天线1的导频信号 
PP_A1 = []; 
for m = 0:NFFT-1; 
        PP_A1 = [PP_A1,P_A*exp(j*D_f*pi*(m)^2/NFFT)];  
end;
% 天线2的导频信号
PP_A2 = [];
for m = 0:NFFT-1;
            PP_A2 = [PP_A2,P_A*exp(j*D_f*pi*(m+NFFT/2)^2/NFFT)];
end;
% FFT 矩阵
F = [];
for k=0:NFFT-1
            W_tem = [];
            for n = 0:NFFT-1;
                W_tem = [W_tem,exp(-j*2*pi*n*k/NFFT)];
            end;
            F = [F;W_tem];
end;
%LS估计系数  
PP = [diag(PP_A1)*F(:,1:N_P),diag(PP_A2)*F(:,1:N_P)];%参看式（5-78）中的 。
Q = inv(PP'*PP);%参看式（5-79）中的 。
% 天线1发送信号。将导频插入并行OFDM数据，一个块状导频OFDM符号后3个数据OFDM符号
TS1_BeforeIFFT = Insert_PilotSymbol(PP_A1,Data_Pattern1,D_t,NofOFDMSymbol,NFFT);
% 天线2发送信号。将导频插入并行OFDM数据，一个块状导频OFDM符号后3个数据OFDM符号
TS2_BeforeIFFT = Insert_PilotSymbol(PP_A2,Data_Pattern2,D_t,NofOFDMSymbol,NFFT);
ser_without_isic = [];
snr_min =0;%最小信噪比
snr_max =50;%最大信噪比
step = 5;%信噪比步长
for snr = snr_min:step:snr_max; %信噪比循环
%对第1个发送天线
rs11_frame = [];
%存放并行的经过多径与加性高斯白噪声信道的第1个发送天线到第1个接收天线的每个OFDM符号
rs12_frame = [];
%存放并行的经过多径与加性高斯白噪声信道的第1个发送天线到第2个接收天线的每个OFDM符号
initial_time=0;                 % 初始时间
for i=0:NofOFDMSymbol-1;	%对所有OFDM符号（包括导频OFDM符号和数据OFDM符号）
% 产生第1个天线上的OFDM符号。通过调用OFDM_Modulator进行OFDM调制 
OFDM_signal_tem = OFDM_Modulator(TS1_BeforeIFFT(i+1,:),NFFT,G);
%产生随时间变化的第1个发送天线到第1个接收天线之间的信道冲激响应  
        [h11, t] = MCM_channel_model(u11, initial_time, number_of_summations,…
symbol_duration,f_dmax, h11_initial);%对应每一个OFDM符号产生多径信道h11
     %产生随时间变化的第1个发送天线到第2个接收天线之间的信道冲激响应  
[h12, t] = MCM_channel_model(u12, initial_time, number_of_summations,…
symbol_duration,f_dmax, h12_initial);%对应每一个OFDM符号产生多径信道h12
         initial_time = t;%每一个OFDM符号的起始时间
         %第1个接收天线接收到的来自第1个发送天线的信号。发送信号经多径信道到达接收端。
         rs11 = conv(OFDM_signal_tem, h11);
         %第2个接收天线接收到的来自第1个发送天线的信号。发送信号经多径信道到达接收端。
         rs12 = conv(OFDM_signal_tem, h12);
         rs11 = awgn(rs11,snr,'measured','dB');%考虑加性高斯白噪声
         rs12 = awgn(rs12,snr,'measured','dB');%考虑加性高斯白噪声
         rs11_frame = [rs11_frame; rs11];%将每个循环的接收信号并行存储
         rs12_frame = [rs12_frame; rs12];%将每个循环的接收信号并行存储
         clear OFDM_signal_tem;
end;
%对第2个发送天线
rs21_frame = [];
%存放并行的经过多径与加性高斯白噪声信道的第2个发送天线到第1个接收天线的每个OFDM符号
rs22_frame = [];
%存放并行的经过多径与加性高斯白噪声信道的第2个发送天线到第2个接收天线的每个OFDM符号
initial_time=0;              % 初始时间
for i=0:NofOFDMSymbol-1;%这个循环可参考天线1
       OFDM_signal_tem = OFDM_Modulator(TS2_BeforeIFFT(i+1,:),NFFT,G);
       [h21, t] = MCM_channel_model(u21, initial_time, number_of_summations,…
symbol_duration,f_dmax, h21_initial);%对应每一个OFDM符号产生多径信道h21
       [h22, t] = MCM_channel_model(u22, initial_time, number_of_summations,…
symbol_duration,f_dmax, h22_initial);%对应每一个OFDM符号产生多径信道h22
       initial_time = t;
       rs21 = conv(OFDM_signal_tem, h21);
       rs22 = conv(OFDM_signal_tem, h22);
       rs21 = awgn(rs21,snr,'measured','dB');
       rs22 = awgn(rs22,snr,'measured','dB');
       rs21_frame = [rs21_frame; rs21];
       rs22_frame = [rs22_frame; rs22];
       clear OFDM_signal_tem;
end;
% 接收天线1：OFDM信号解调及信道估计
estimated_h11_21 = [];
Received_PP= []; % 存放接收的导频
Receiver_Data = []; % 存放接收的信号
SignalPostFFT1 = [];
SignalPostFFT2 = [];
d1 = []; % 第一个天线接收的信号
d2 = []; % 第二个天线接收的信号
data_symbol_1 = [];
data_symbol_2 = [];
for i=1:NofOFDMSymbol;
       if (N_P > G+1) & (i>1)
          % 如果不是第一个信号并且多径信道的长度（多径的个数）大于保护间隔，则需考虑ISI
          previous_symbol11 = rs11_frame(i-1,:);% 第一个发送天线到第一个接收天线的信号        
          previous_symbol21 = rs21_frame(i-1,:);% 第二个发送天线到第一个接收天线的信号 
          previous_symbol12 = rs12_frame(i-1,:);% 第一个发送天线到第二个接收天线的信号
          previous_symbol22 = rs22_frame(i-1,:);% 第二个发送天线到第二个接收天线的信号   
          ISI_term11 = previous_symbol11(NFFT+2*G+1:NFFT+G+N_P-1); 
                           % 截取NFFT+2G+1: NFFT+G+N_P-1 来构造ISI用
         ISI_11[ISI_term11,zeros(1,length(previous_symbol11)-length(ISI_term11))];                  
          ISI_term21 = previous_symbol21(NFFT+2*G+1:NFFT+G+N_P-1); 
        ISI_21=[ISI_term21,zeros(1,length(previous_symbol21)-length(ISI_term21))];  
          ISI_term12 = previous_symbol12(NFFT+2*G+1:NFFT+G+N_P-1); 
                          % 截取NFFT+2G+1: NFFT+G+N_P-1 来构造ISI用
        ISI_12=[ISI_term12,zeros(1,length(previous_symbol12)-length(ISI_term12))];                  
          ISI_term22 = previous_symbol22(NFFT+2*G+1:NFFT+G+N_P-1); 
        ISI_22=[ISI_term22,zeros(1,length(previous_symbol22)-length(ISI_term22))];
          rs1_i = rs11_frame(i,:) + rs21_frame(i,:) + ISI_11 +  ISI_21;  
          rs2_i = rs12_frame(i,:) + rs22_frame(i,:) + ISI_12 +  ISI_22;
                        % 将ISI添加到OFDM符号
      else %否则就不需要考虑码间干扰，此时在每个接收天线上收到是两个发送信号的叠加
          rs1_i = rs11_frame(i,:) + rs21_frame(i,:);
          rs2_i = rs12_frame(i,:) + rs22_frame(i,:);
   end;  
   if (mod(i-1,D_t)==0)%判断是导频OFDM符号还是数据OFDM符号；在D_t的整数倍上的OFDM符号为导
%频符号，在导频OFDM符号上进行信道估计。
        Demodulated_Pilot1 = OFDM_Demodulator(rs1_i,NFFT,NFFT,G);
        SignalPostFFT1 = [SignalPostFFT1; Demodulated_Pilot1];
        Demodulated_Pilot2 = OFDM_Demodulator(rs2_i,NFFT,NFFT,G);
        SignalPostFFT2 = [SignalPostFFT2; Demodulated_Pilot2];
        Demodulated_P1 = [];
        Demodulated_P2 = [];
        for i = 1:NFFT;
            Demodulated_P1 = [Demodulated_P1; Demodulated_Pilot1(i)];
            Demodulated_P2 = [Demodulated_P2; Demodulated_Pilot2(i)];
        end;
        estimated_h11_21_i = Q * PP' * Demodulated_P1;%h11、h21信道估计，参考式（5-79）。
        he11_21_i = estimated_h11_21_i;
        he11_i = he11_21_i(1:length(estimated_h11_21_i)/2);
%he11_21_i中前一半为h11
        H11_i = fft([he11_i;zeros(NFFT-N_P,1)]);
        he21_i = he11_21_i(length(he11_21_i)/2+1:length(he11_21_i));
%he11_21_i中后一半为h21
        H21_i = fft([he21_i;zeros(NFFT-N_P,1)]);
        estimated_h12_22_i = Q * PP' * Demodulated_P2;%h12、h22信道估计，参考式（5-79）。
        he12_22_i = estimated_h12_22_i;
        he12_i = he12_22_i(1:length(he12_22_i)/2);
        %he12_22_i中前一半为h12
        H12_i = fft([he12_i;zeros(NFFT-N_P,1)]);
        he22_i = he12_22_i(length(he12_22_i)/2+1:length(he12_22_i));
        %he12_22_i中后一半为h22
        H22_i = fft([he22_i;zeros(NFFT-N_P,1)]);        
   else%不在D_t的整数倍上的OFDM符号为数据符号。在数据OFDM符号上借助导频OFDM符号估计出的信道
%恢复发送端数据。
        % OFDM 符号解调。这部分是在前面信道估计后，对数据OFDM符号进行解调
        Demodulated_signal1_i = OFDM_Demodulator(rs1_i,NFFT,NFFT,G);  
        SignalPostFFT1 = [SignalPostFFT1; Demodulated_signal1_i];
        Demodulated_signal2_i = OFDM_Demodulator(rs2_i,NFFT,NFFT,G); 
        SignalPostFFT2 = [SignalPostFFT2; Demodulated_signal2_i];
        d1_i = [];
        d2_i = [];
        for k = 1:NFFT;
            H_k = [H11_i(k),H21_i(k); H12_i(k),H22_i(k)];
            y = [Demodulated_signal1_i(k);Demodulated_signal2_i(k)];
            x = inv(H_k) * y;%计算发送信号
            d1_i = [d1_i,x(1)];
            d2_i = [d2_i,x(2)];
        end;
        d1 = [d1; d1_i];
        d2 = [d2; d2_i];
        demodulated_symbol_1i = ddemodce(d1_i,1,1,'qask',M_ary);%星座反映射
        demodulated_symbol_2i = ddemodce(d2_i,1,1,'qask',M_ary);%星座反映射
        data_symbol_1 = [data_symbol_1, demodulated_symbol_1i];
        data_symbol_2 = [data_symbol_2, demodulated_symbol_2i];
    end;
end;
    data_symbol_1 = data_symbol_1';
data_symbol_2 = data_symbol_2';
%计算误码率
    [number1_without_isic, ratio1_without_isic] = symerr(symbols1,data_symbol_1);
    ser_without_isic = [ser_without_isic, ratio1_without_isic];
end;
ser_relz = [ser_relz;ser_without_isic];
end;
ser = sum(ser_relz)/Number_Relz;%计算平均误码率
snr = snr_min:step:snr_max;
semilogy(snr, ser,'b*');%画误码率曲线
ylabel('SER');
xlabel('SNR');
data = [snr; ser];
save ser_conventional_method.am data -ascii;
%存储误码率数据。可利用该指令存储各种情绪下的数据，以备比对。

%插入导频子程序，子程序名称：Insert_PilotSymbol.m
功能：将导频OFDM信号插入到数据OFDM符号中。在调用该子程序之前，程序中有750个数据OFDM符号，250个导频OFDM符号，分别存放在两个矩阵中，插入导频子程序是将两个矩阵合并为一个1000个OFDM符号的矩阵。导频插入方式是在一个导频OFDM符号后有3个数据OFDM符号。
function [IPS]=Insert_PilotSymbol…
(PP_A,QASK_Symbol,Pilot_Distance,NofOFDMSymbol,NFFT)
%输入参数：PP_A为导频序列；QASK_Symbol为数据序列；Pilot_Distance为导频间隔；NofOFDMSymbol
%为整体OFDM符号个数，包括导频OFDM符号数与数据OFDM符号数；NFFT为FFT长度。
%输出参数：返回一个1000个OFDM符号矩阵，该矩阵每1个导频OFDM信号后都有3个数据OFDM符号。
TS2_BeforeIFFT = []; % Transmitted Signal before IFFT
m = 0;
for i=0:NofOFDMSymbol-1;
    QASK_tem = [];
    if (mod(i, Pilot_Distance)==0)
        TS2_BeforeIFFT = [TS2_BeforeIFFT; PP_A];
        m=m+1;
    else
    TS2_BeforeIFFT = [TS2_BeforeIFFT; QASK_Symbol(i-m+1,:)];
    end;
    clear QASK_tem;
end;
IPS = TS2_BeforeIFFT;

%OFDM调制子程序，子程序名称：OFDM_Modulator.m
功能：利用ifft进行OfdM调制，并且加入循环前缀做保护间隔。
function [y] = OFDM_Modulator(data,NFFT,G);
%输入参数：data为数据符号，是发送端要发送的4QAM或16QAM数据；NFFT为FFT长度；G为保护间隔长度
%输出参数：y返回插入保护间隔的OFDM符号。当G为0时，仅返回OFDM符号本身。
chnr = length(data);
N = NFFT;
x = [data,zeros(1,NFFT - chnr)]; %若数据长度不为2的整数次幂，则补零
a = ifft(x); % fft
y = [a(NFFT-G+1:NFFT),a]; % 插入保护间隔

%多径信道子程序，子程序名称：MCM_channel_model.m
功能：这个程序围绕式（5-75）展开，要产生一个多径信道。
function [h,t_next] = MCM_channel_model(u, initial_time, number_of_summations,… symbol_duration, f_dmax, channel_coefficients);
%输入参数：u为式（5-75）中的独立随机变量；initial_time为对应每一个OFDM符号的信道起始时间；%number_of_summations为信道的调和系数即为M；symbol_duration为一个OFDM符号时间；f_dmax为
%最大多普勒频移；channel_coefficients为信道系数矩阵，包括 与 。
t = initial_time;
Channel_Length = length(channel_coefficients);
h_vector = [];
for k=1:Channel_Length;
    u_k = u(k,:); %一个随机变量
    phi = 2 * pi * u_k; % 产生随机相位
    f_d = f_dmax * sin(2*pi*u_k); % 产生多普勒频移
    h_tem= channel_coefficients(k)* 1/(sqrt(number_of_summations))… *sum(exp(j*phi).*exp(j*2*pi*f_d*t));
    h_vector = [h_vector,  h_tem];
end;
h = h_vector;
t_next = initial_time + symbol_duration; %产生对应下一个OFDM符号的信道起始时间
	
%OFDM解调子程序，子程序名称：OFDM_Demodulator.m
功能：利用fft对接收的OFDM进行解调。在解调前先去掉保护间隔。
function [y] = OFDM_Demodulator(data,chnr,NFFT,G);
%输入参数：data为接收端OfDM符号；
%输出参数：y返回径fft恢复的4QAM或16QAM数据。
x_remove_guard_interval = [data(G+1:NFFT+G)]; % 如有保护间隔，则去掉保护间隔
x = fft(x_remove_guard_interval);
y = x(1:chnr); %若补零，则去掉所补零值。

仿真结果如图5-16所示。图中仿真了基于块状导频、多径信道、2发2收MIMO-OFDM系统、LS信道估计下，没有加入保护间隔、保护间隔为5和保护间隔为9时的误码率曲线。信道多径数为9。
 
图5-16 误码率比较图



6.5 仿真实例
实例6-1 选择性映射（SLM）PAPR减小方法仿真
功能：采用SLM映射方法减小OFDM符号PAPR。
程序名称：Example6_1.m
程序代码：
clear all; clc; close all;
N = 128;  %FFT长度，即OFDM信号子载波个数；                                                                  
M = 5;     %M-1为SLM选择支路数                                                                
%Base_MOD_Set  = [1 -1 j -j];%星座映射为4PSK
Base_MOD_Set  = [1+j -1+j 1-j -1-j 3+j -3+j 3-j -3-j 1+3j -1+3j 1-3j …
-1-3j 3+3j -3+3j 3-3j -3-3j];%星座映射为16QAM
Phase_Set = [1 -1 j -j];%4种随机相位选择
%Phase_Set = [1 -1];%2种随机相位选择
MAX_SYMBOLS  = 1e5;%循环数，参数MAX_SYMBOLS个OFDM符号，求PAPR供统计用
PAPR_Orignal = zeros(1,MAX_SYMBOLS);%存每个循环中的原OFDM符号PAPR
PAPR_SLM     = zeros(1,MAX_SYMBOLS);%存每个循环中的SLM后的OFDM符号的PAPR
X     = zeros(M,N);%存放星座映射后的发送数据，可以理解为频域信号
Index = zeros(M,N);
%存放在Base_MOD_Set和Phase_Set中进行随机进行星座映射和产生随机相位的索引
for nSymbol=1:MAX_SYMBOLS
        Index(1,:)   = randint(1,N,length(Base_MOD_Set))+1;%产生星座映射索引
        Index(2:M,:) = randint(M-1,N,length(Phase_Set))+1;%产生随机相位索引
        X(1,:) = Base_MOD_Set(Index(1,:)); % 进行星座映射
        Phase_Rot = Phase_Set(Index(2:M,:));%产生随机相位序列
X(2:M,:) = repmat(X(1,:),M-1,1).*Phase_Rot; 
% 进行星座映射后的数据与随机相位序列矢量点乘。X中的第1行是原数据，第2至M行是经过相位旋转的数据
        x = ifft(X,[],2);   
 % 进行OFDM调制，得到时域信号。第1行为原OFDM符号，第2至M行是SLM中的各个支路OFDM符号，在其
%中挑选PAPR最小的发送。
        Signal_Power = abs(x.^2);%计算OFDM信号的功率
        Peak_Power   = max(Signal_Power,[],2);%计算功率的最大值
        Mean_Power   = mean(Signal_Power,2);%计算平均功率
    PAPR_temp = 10*log10(Peak_Power./Mean_Power);%计算PAPR
        PAPR_Orignal(nSymbol) = PAPR_temp(1);%从第1行提取原OFDM信号的PAPR
        PAPR_SLM(nSymbol)     = min(PAPR_temp(2:M));
%各个支路OFDM信号的PAPR中选择选择最小值，发送端对应发送PAPR最小的OFDM符号
end
%[cdf1, PAPR1] = ecdf(PAPR_Orignal);%统计原OFDM符号PAPR的概率密度函数
[OrignalOFDM PAPR1] = hist(PAPR_Orignal,[1:0.1:18]);
cdf1=cumsum(OrignalOFDM)/MAX_SYMBOLS;
%统计原OFDM符号PAPR的概率密度函数。可以两种统计方法中选择一种。
%[cdf2, PAPR2] = ecdf(PAPR_SLM);%统计SLM-OFDM符号PAPR的概率密度函数
[SLMOFDM PAPR2] = hist(PAPR_SLM,[1:0.1:18]);
cdf2=cumsum(SLMOFDM)/MAX_SYMBOLS;
%统计SLM-OFDM符号PAPR的概率密度函数。可以两种统计方法中选择一种。
semilogy(PAPR1,1-cdf1,'-b',PAPR2,1-cdf2,'-r')
%画原OFDM符号与SLM-OFDM符号PAPR的CCDF。
legend('Orignal','SLM')
title('选择支路数为4')
xlabel('PAPR0 [dB]');
ylabel('CCDF (Pr[PAPR>PAPR0])');
grid on
data = [PAPR2; 1-cdf2];
save PAPR_SLM16.am data -ascii;%存各种状态下的仿真结果，以备比较。

   仿真结果如图6-12所示。
实例6-2 部分传输序列（PTS）PAPR减小方法仿真
    功能：利用PTS方法减小OFDM系统PAPR仿真。
程序名称：Example6_2.m
程序代码：
clear all; clc; close all;
N = 128; %子载波个数或IFFT长度                                                                  
V = 4;    %PTS方法分组数，可选择2组、4组、8组，若16组需相应修改程序                                                                  
MAX_SYMBOLS  = 1e4;%循环数，也就是OFDM符号个数
M_ary =4;%星座映射进制数
Phase_Num=4;%旋转相位的相位个数
%产生旋转相位矩阵
for i=1:Phase_Num
       Phase_Set=exp(j*2*pi/Phase_Num);
end
   Phase =[];%存旋转相位所有可能组合
    %分两组时，旋转相位所有可能组合。若旋转相位数为4，则共有42个组合。当V=2时使用。
%for b1=1:length(Phase_Set)
    %    for b2=1:length(Phase_Set)
    %       Phase =[Phase ; [Phase_Set(b1)  Phase_Set(b2)  ]]; 
    %   end
%end
%分四组时，旋转相位所有可能组合。若旋转相位数为4，则共有44个组合。当V=4时使用。
for b1=1:length(Phase_Set)
for b2=1:length(Phase_Set)
for b3=1:length(Phase_Set)
 for b4=1:length(Phase_Set)
Phase =[Phase ; [Phase_Set(b1)  Phase_Set(b2)  Phase_Set(b3) Phase_Set(b4)]]; % end
end
 end
end  
end
%分八组时，旋转相位所有可能组合。若旋转相位数为4，则共有48个组合。当V=8时使用。
% for b1=1:length(Phase_Set)
% for b2=1:length(Phase_Set)
% for b3=1:length(Phase_Set)
% for b4=1:length(Phase_Set)
% for b5=1:length(Phase_Set)
% for b6=1:length(Phase_Set)
% for b7=1:length(Phase_Set)
% for b8=1:length(Phase_Set)
% Phase =[Phase ; [Phase_Set(b1)  Phase_Set(b2)  Phase_Set(b3) Phase_Set(b4)… Phase_Set(b5)  Phase_Set(b6)  Phase_Set(b7) Phase_Set(b8)]]; 
% end
% end
% end
% end
% end
% end
% end
% end
%MAX_SYMBOLS个OFDM符号频域数据产生
length_data=N*MAX_SYMBOLS ;%发送的总的数据符号数，每个数据符号可以是4PSK或16QAM
Base_MOD_Set = []; %存放星座映射后的数据
source_data = randint(length_data,log2(M_ary));%随机产生log2(M_ary)二进制数
symbols = bi2de(source_data);%将log2(M_ary)二进制数合并成多进制数
MOD_Symbol = dmodce(symbols,1,1,'qask',M_ary);%进行星座映射
%将N*MAX_SYMBOLS个串行数据转换为MAX_SYMBOLS个N点并行数据，作为MAX_SYMBOLS个OFDM符号的频%域数据。
m = 0;
for i=0:MAX_SYMBOLS-1;
      MOD_tem = [];
      for n=1:N;
          MOD_tem = [MOD_tem,MOD_Symbol(i*N+n)];
      end;
      Base_MOD_Set = [Base_MOD_Set;MOD_tem];
      clear MOD_tem;
end;
%进行原OFDM符号与经PTS后的OFDM符号PAPR的CCDF统计
Choose_Len = length(Phase_Set)^V;    %旋转相位所有组合数 
PAPR_Orignal = zeros(1,MAX_SYMBOLS);%存放原OFDM信号所有循环的PAPR
PAPR_PTS     = zeros(1,MAX_SYMBOLS);%存放原PTS-OFDM信号所有循环的PAPR
for nSymbol=1:MAX_SYMBOLS  %循环开始
    %对原始OFDM符号
        X = Base_MOD_Set(nSymbol,:);                                               
        x = ifft(X,[],2);                 %OFDM调制                                      
        Signal_Power = abs(x.^2);　　 　%计算功率
        Peak_Power   = max(Signal_Power,[],2);%计算峰值功率
        Mean_Power   = mean(Signal_Power,2);  %计算平均功率
        PAPR_Orignal(nSymbol) = 10*log10(Peak_Power./Mean_Power);%计算PAPR
        % 对经过PTS后的OFDM符号 
        A = zeros(V,N);
        %  交织分组方式 
        for v=1:V
            A(v,v:V:N) = X(v:V:N);
        end
%顺序分组方式
%for v=1:V
%A(v,(1+(v-1)*N/V):(N/V+(v-1)*N/V)) = X((1+(v-1)*N/V):(N/V+(v-1)*N/V));
%end
%随机分组方式 
%Index= randperm(N); 
%for v=1:V 
%   A(v,Index(v:V:N)) = X(Index(v:V:N));
%end
a = ifft(A,[],2);% OFDM调制 
       min_value = 10;
       %分组后的数据进行OfDM调制后与旋转相位相乘，按全举法搜索最优旋转相位组合
       for n=1:Choose_Len
       temp_phase = Phase(n,:).';
       temp_max = max(abs(sum(a.*repmat(temp_phase,1,N))));
           if temp_max<min_value
               min_value = temp_max;
                Best_n = n;
           end
       end
       %发送最优相位组合信号并计算其PAPR
       aa = sum(a.*repmat(Phase(Best_n,:).',1,N)); 
       Signal_Power = abs(aa.^2);
       Peak_Power   = max(Signal_Power,[],2);
       Mean_Power   = mean(Signal_Power,2);
       PAPR_PTS(nSymbol) = 10*log10(Peak_Power./Mean_Power);
end
%统计原OFDM符号与PTS-OFDM符号的CCDF，并画出其曲线
[cdf1, PAPR1] = ecdf(PAPR_Orignal);
[cdf2, PAPR2] = ecdf(PAPR_PTS);
semilogy(PAPR1,1-cdf1,'-b*',PAPR2,1-cdf2,'-r+')
legend('Orignal','PTS')
title('V=4')
xlabel('PAPR0 [dB]');
ylabel('CCDF (Pr[PAPR>PAPR0])');
grid on
%存储各种状态下的数据，以备比较
data = [PAPR1; 1-cdf1];
save PTS_PAPR_.am data -ascii;
仿真结果见图6-15、图6-16、图6-17、图6-18。
实例6-3 限幅法减小OFDM符号PAPR仿真
功能：利用限幅法降低OFDM符号的PAPR。
程序名称：Example6_3.m
程序代码：
    clear all; clc; close all;
K  = 128;     %OFDM符号子载波数                                                              
IF = 2;       %过采样因子                                                            
N  = K*IF;    %过采样后一个OFDM符号数据个数 ，也即FFT或IFFT长度                                                             
CR = 4;       % 剪切率（CR= 4 即剪切率为6dB)
QPSK_Set  = [1 -1 j -j];%星座映射为4PSK
ITERATE_NUM = 4;%
MAX_SYMBOLS  = 1e4;
PAPR_Orignal = zeros(1,MAX_SYMBOLS);%存原OFDM符号PAPR
PAPR_RCF     = zeros(ITERATE_NUM,MAX_SYMBOLS);%存限幅后的PAPR
for nSymbol=1:MAX_SYMBOLS
        %产生原OFDM符号频域数据
        Index = randint(1,K,length(QPSK_Set))+1;
X = QPSK_Set(Index(1,:));                                              
     % 过采样
        XX = [X(1:K/2) zeros(1,N-K) X(K/2+1:K)];
        %OFDM调制
x = ifft(XX,[],2);                                                     
      %原OFDM符号计算PAPR 
        Signal_Power = abs(x.^2);
        Peak_Power   = max(Signal_Power,[],2);
        Mean_Power   = mean(Signal_Power,2);
        PAPR_Orignal(nSymbol) = 10*log10(Peak_Power./Mean_Power);
        %进行限幅循环。共进行ITERATE_NUM次剪切。
        for nIter=1:ITERATE_NUM
            % 剪切
            x_tmp = x(Signal_Power>CR*Mean_Power);
            x_tmp = sqrt(CR*Mean_Power)*x_tmp./abs(x_tmp);
            x(Signal_Power>CR*Mean_Power) = x_tmp;
            % 滤波
            XX = fft(x,[],2);
            XX(K/2+(1:N-K)) = zeros(1,N-K);
            x = ifft(XX,[],2); 
            % PAPR 计算
            Signal_Power = abs(x.^2);
            Peak_Power   = max(Signal_Power,[],2);
            Mean_Power   = mean(Signal_Power,2);
            PAPR_RCF(nIter,nSymbol) = 10*log10(Peak_Power./Mean_Power);
        end
end
%PAPR的CCDF统计
[cdf0, PAPR0] = ecdf(PAPR_Orignal);
[cdf1, PAPR1] = ecdf(PAPR_RCF(1,:));
[cdf2, PAPR2] = ecdf(PAPR_RCF(2,:));
[cdf3, PAPR3] = ecdf(PAPR_RCF(3,:));
[cdf4, PAPR4] = ecdf(PAPR_RCF(4,:));
semilogy(PAPR0,1-cdf0,'-b',PAPR1,1-cdf1,'-r',PAPR2,1-cdf2,'-g',PAPR3,…
1-cdf3,'-c',PAPR4,1-cdf4,'-m')
legend('Orignal','One clip and filter','Two clip and filter',…
'Three clip and filter','Four clip and filter')
xlabel('PAPR0 [dB]');
ylabel('CCDF (Pr[PAPR>PAPR0])');
xlim([0 12])
grid on
仿真结果如图6-25所示。
 
图6-25 限幅法PAPR仿真图



7.5仿真实例
实例7_1  Chow算法仿真
程序功能：按照Chow算法的原则进行比特和功率分配的仿真
程序名称：Example7_1.m
程序代码：
N_subc=32;%子载波数
BER=1e-4;%误码率
gap=-log(5*BER)/1.5; %信噪比间隔，用dB描述
P_av=1;%每个子载波上的平均功率归一化为1
Pt=P_av*N_subc;%每个OFDM符号的平均功率
SNR_av=16;%每个子载波上的平均信噪比
noise=P_av./10.^(SNR_av./10);%每个子载波上的噪声功率
Rt=128;%一个OFDM符号上待分配的比特数。
%子载波信道增益，仿真瑞利衰落信道。后面针对这个信道特性进行比特、功率分配。
subcar_gains=random('rayleigh',1,1,N_subc);
SNR=(subcar_gains.^2)./(noise*gap); 
%每个子载波上的信噪比，注意功率为归一化的。另外注意此信噪比已除以 。
%调用chow_algo子程序进行功率和比特分配，返回功率分配和比特分配结果。
[bit_alloc power_alloc]=chow_algo(SNR,N_subc,gap,Rt)；
power_alloc=Pt.*(power_alloc./sum(power_alloc));
%绘图
figure(1);
subplot(2,1,1);
plot(subcar_gains,'-r');
legend('信道增益');
hold on;
%stem(bit_alloc,'fill','MarkerSize',3);
stairs(bit_alloc);
title('Chow算法');
ylabel('Bits allocation');
xlabel('Subcarriers');
axis([0 32 0 6]);
subplot(2,1,2);
%stem(power_alloc,'fill','MarkerSize',3);
stairs(power_alloc);
ylabel('Power allocation');
xlabel('Subcarriers');
axis([0 32 0 2]);

chow算法子程序。子程序名称：chow_algo.m
 function [bits_alloc, power_alloc] = chow_algo(SNR,N_subc,gap,target_bits)
%输入参数：SNR为每个子信道的信噪比；N_subc为子载波数；gap为信噪比间隔，为常数，用dB表示。          
% target_bits为每个OFDM上待分配的比特总数。  
%输出参数：bits_alloc为比特分配结果；power_alloc为功率分配结果
% ----------------------------参数初始化---------------------------------------
margin=0;                            %门限值
Max_count=10;                       %最大迭代次数
Iterate_count=0;                   %迭代计数器
N_use=N_subc;                       %可用子载波数
total_bits=0;                       %分配的总比特数
power_alloc=zeros(1,N_subc);     %功率分配结果
bits_alloc=zeros(1,N_subc);      %比特分配结果
temp_bits=zeros(1,N_subc);       %每个子载波上分配的比特数的理论值，非整数
round_bits=zeros(1,N_subc);      %每个子载波上分配的比特数取整
diff=zeros(1,N_subc);             %每个子载波比特分配的余量
%-----------------------------比特分配-------------------------
%此部分参考7.3.3节 。子信道比特分配公式为 。
while (total_bits~=target_bits)&&(Iterate_count<Max_count)
    %当总的比特数被分配完或者迭代次数等于最大迭代次数时，此循环结束
    Iterate_count=Iterate_count+1;%迭代次数加1
    N_use=N_subc;%循环开始时，假设所有子载波都可用
temp_bits=log2(1+SNR./(1+margin/gap)); 
%计算每个子信道上的理论比特分配值，公式为 
%也就是说，程序中的SNR中的信噪比为 。在主程序调用子程序时，信噪比已除以 。
    round_bits=round(temp_bits);%对理论比特分配值进行取整
    diff=temp_bits-round_bits;%计算理论值比特数与实际分配比特数的差值
    total_bits=sum(round_bits);%计算取整后的比特和
    if(total_bits==0)
        disp('the channel is not be used');%如果比特和为零，显示“这个信道不能用”
    end
    nuc=length(find(round_bits==0)); %查找子信道上分配比特数为零的信道数
    N_use=N_subc-nuc; %可用子信道数为子载波数减去子信道上比特数为零的信道数
margin=margin+10*log10(2^((total_bits-target_bits)/N_use));
%根据信道实际情况，计算新的裕量。
end
%------------------------------比特调整--------------------------
%当分配的总的比特数大于目标比特数时，显然分配的比特数分配多了，需要减少比特数。减小的办法是找到diff %中最小的，注意diff是有正有负的，当diff为负时，说明分配给某个子载波的比特数大于理论值。当有需要
%将总的分配比特数减小时，找diff中最小的，这个说明当时分配时，给这个子载波分配的比特数，和理论值比%差最大的，给对应diff最小的子载波分配的比特数减1，之后看看分配的比特数是否还大于目标比特数，如果
%依旧大，按照这个规律下去，总能使得分配的总的比特数等于目标比特数。当给对应diff最小的子载波分配的%比特数减1的同时，将diff最小的加1的目的是，这个最小值不会再次为最小。
while(total_bits>target_bits)
        use_ind=find(round_bits>0);
        diff_use=diff(use_ind);  
        id=find(diff_use==min(diff_use),1); % 
        ind_alter=use_ind(id);  % 
        round_bits(ind_alter)=round_bits(ind_alter)-1;
        diff(ind_alter)=diff(ind_alter)+1;
        total_bits=sum(round_bits);
end
%当分配的总的比特数小于目标比特数时，说明分配比特分配少了，需要继续进行比特分配。分配的原则是，找%diff中最大的，这个是在round指令取整时，误差最大的，也就是少分配比特最多的。将这个子载波上的比
%特数加1，如果这样依旧总的比特数小于目标比特数，依次进行，总能做到分配的比特数等于目标比特数。将%相应的diff减1的目的是这个最大值不会再次成为最大。
while(total_bits<target_bits)
        use_ind=find(round_bits~=0);
        diff_use=diff(use_ind);
        id=find(diff_use==max(diff_use),1);
        ind_alter=use_ind(id);
        round_bits(ind_alter)=round_bits(ind_alter)+1;
        diff(ind_alter)=diff(ind_alter)-1;
        total_bits=sum(round_bits);
end
bits_alloc=round_bits;
%--------------------------功率分配-----------------------------
power_alloc=(2.^bits_alloc-1)./SNR;
end
程序运算结果如图7-4所示。
 
图7-4 Chow算法比特分配与功率分配
实例7_2  Fischer算法仿真
程序功能：按照Fischer算法原则进行比特与功率分配的仿真
程序名称：Example7_2.m
程序代码：
N_subc=64;%子载波数
BER=1e-4;%误码率
gap=-log(5*BER)/1.5; %信噪比间隔，用dB描述
P_av=1;% 每个子载波上的平均功率归一化为1
Pt=P_av*N_subc;%一个OFDM信号的平均功率
Rt=128;%一个OFDM符号待分配的比特数
SNR_av=16;% 每个子载波上的平均信噪比
noise=P_av./10.^(SNR_av./10);%每个子载波上的噪声功率
%子载波信道增益，仿真瑞利衰落信道。后面针对这个信道特性进行比特、功率分配。
gain_subc=random('rayleigh',1,1,N_subc);
%调用Fischer.m子程序，进行Fischer算法的比特与功率分配
[bit_alloc power_alloc]=Fischer(N_subc,Rt,Pt,gain_subc);
power_alloc=Pt.*(power_alloc./sum(power_alloc))
%画图
clf;
figure(1);
subplot(2,1,1);
plot(gain_subc,'-r');
legend('信道增益');
hold on;
stem(bit_alloc,'fill','MarkerSize',3);
title('Fischer算法');
ylabel('Bits allocation');
xlabel('Subcarriers');
axis([0 32 0 8]);
subplot(2,1,2);
stem(power_alloc,'fill','MarkerSize',3);
ylabel('Power allocation');
xlabel('Subcarriers');
axis([0 32 0 6]);

Fischer算法子程序。子程序名称：Fischer.m
%这部分内容参考7.3.4节
function [bit_alloc power_alloc]=Fischer(Num_subc,Rt,Pt,gain_subc)
%输入参数：Num_subc为子载波数；Rt为待分配的比特数；Pt为OFDM符号平均功率；gain_subc为子载波信
%道增益。
%输出参数：bit_alloc为比特分配结果；power_alloc为功率分配结果。
%---------------------------------比特分配 ----------------------------------
N_H=1./gain_subc.^2;%子信道上的噪声功率
LD=log2(N_H);
index_use0=1:Num_subc;     % 激活的子载波集合I初始化为{1：Num_subc}
num_use=Num_subc;           % 初始时假设可用子信道亦即激活的子载波个数为子载波数Num_subc
bit_temp0=(Rt+sum(LD))./Num_subc-LD; % 见式7-23，集合I中的子载波初始比特分配
bit_temp=zeros(1,Num_subc);  % 比特调整前的临时变量
%判断子载波进行比特分配时，是否有分配的比特数小于等于零的，若有，将这些子载波从可用的子载波中剔除，
%之后重新按式7-23进行比特分配，依旧进行上述循环，直到所有子载波上分配的比特数都大于零。
flag=1; 
while flag
         id_remove=find(bit_temp0(index_use0)<=0); 
%返回需要移除的子载波（就是分配的比特数小于等于零的子载波）序号在集合I中的位置  
         index_remove=index_use0(id_remove); % 返回移除的子载波序号
    %如果没有需要从I中移除的子载波，跳出当前的while循环，进入比特取整阶段。
         if(length(index_remove)==0)  
             break;
         end    
        index_use0=setdiff(index_use0,index_remove); % 返回更新后的集合I
        num_use=length(index_use0);   % 更新后的I中可用子载波数目
        %在新的集合I中重新计算比特分配
        bit_temp0(index_use0)=(Rt+sum(LD(index_use0)))./num_use-LD(index_use0);    
        flag=1;
end
index_use=index_use0;   % 将集合I（index_use0）中的激活子载波序号返回给变量index_use。
%将激活的子载波所分配的比特数返回给bit_temp，其他不可用子载波的比特分配都置为0
bit_temp(index_use)=bit_temp0(index_use);   
%------------------------比特取整-----------------------------
bit_round=zeros(1,Num_subc);
bit_round(index_use)=round(bit_temp(index_use));
bit_diff(index_use)=bit_temp(index_use)-bit_round(index_use);
bit_total=sum(bit_round(index_use));
%------------------------比特调整------------------------------
%比特调整部分与Chow算法比特调整部分类似，参考Chow算法。
while(bit_total>Rt)
        id_alter=find(bit_round(index_use)>0);   
        index_alter=index_use(id_alter);        
        min_diff=min(bit_diff(index_alter)); 
        id_min=find(bit_diff(index_alter)==min_diff,1); 
        index_min=index_alter(id_min); 
        bit_round(index_min)=bit_round(index_min)-1;
        bit_total=bit_total-1;
        bit_diff(index_min)=bit_diff(index_min)+1;
end
while(bit_total<Rt)
        id_alter=find(bit_round(index_use)>0); 
        index_alter=index_use(id_alter);  
        max_diff=max(bit_diff(index_alter));
        id_max=find(bit_diff(index_alter)==max_diff,1);  
        index_max=index_alter(id_max); 
        bit_round(index_max)=bit_round(index_max)+1;
        bit_total=bit_total+1;
        bit_diff(index_max)=bit_diff(index_max)-1;
end
bit_alloc=bit_round;
%--------------------------------功率分配-------------------------------------
power_alloc=zeros(1,Num_subc);   
index_use2=find(bit_alloc>0);
%为激活子载波分配比特，不可用子载波发射功率都置为0
power_alloc(index_use2)=Pt.*(N_H(index_use2)).*2.^bit_round(index_use2)./…
        (sum(N_H(index_use2)).*2.^bit_round(index_use2));    

仿真结果见图7-5。
 
图7-5 Fischer算法比特与功率分配图
实例7_3  Hughes-Hartogs算法仿真
程序功能：按照Hughes-Hartogs算法的原则进行比特与功率分配仿真
程序名称：Example7_3.m
程序代码：
N_subc=64;
P_av=1;
Pt=P_av*N_subc;
SNR_av=16;
Noise=P_av./10.^(SNR_av./10);
B=1e6;%OFDM符号的带宽
N_psd=Noise./(B/N_subc);%子信道噪声功率谱密度
BER=1e-4;
M=8;%每个子载波上分配比特数的最大值
Rb=128;
H=random('rayleigh',1,1,N_subc);
%调用Hughes_Hartogs子程序，进行Hughes-Hartogs算法比特与功率分配
[bit_alloc, power_alloc]=Hughes_Hartogs(N_subc,Rb,M,BER,N_psd,H);
power_alloc=Pt.*(power_alloc./sum(power_alloc))
%画图
figure(1);
subplot(2,1,1);
stem(bit_alloc,'fill','MarkerSize',3);
hold on;
plot(H,'-r');
ylabel('Bits allocation');
xlabel('Subcarriers');
title('Hughes-Hartogs Algorithm');
axis([0 64 0 4]);
subplot(2,1,2);
stem(power_alloc,'fill','MarkerSize',3);
ylabel('Power allocation');
xlabel('Subcarriers');
axis([0 64 0 3]);

Hughes-Hartogs算法子程序。子程序名称：Hughes_Hartogs.m
function [bit_alloc,power_alloc]=Hughes_Hartogs(N_subc,Rb,M,BER,Noise,H)
%输入参数：N_subc子载波数；Rb为待分配的比特数；M为 ；BER为误码率；Noise为每个子载波上的噪声功
%          率；H为随机信道增益；
%输出参数：bit_alloc为比特分配结果；power_alloc为功率分配结果。
%初始化
bit_alloc=zeros(1,N_subc);
power_alloc=zeros(1,N_subc);
bit_total=0;
%计算在每个子载波上每增加1比特带来功率的增加
for i=1:N_subc
%采用MPSK调制时用
         power_add(i)=(f_mpsk(bit_alloc(i)+1,BER,Noise)-…
                         f_mpsk(bit_alloc(i),BER,Noise))/H(i)^2;
    %采用MQAM调制时用
%power_add(i)=(f_mqam(bit_alloc(i)+1,BER,Noise)-…
%                f_mqam(bit_alloc(i),BER,Noise))/H(i)^2;
end
min_add=min(power_add);%计算增加功率的最小值
index_min=find(power_add==min_add,1);%返回增加功率最小值对应的子载波序号
%第一个比特分配。把使增加功率最小的子载波上的比特数加1
bit_alloc(index_min)=bit_alloc(index_min)+1;
bit_total=sum(bit_alloc);%计算总的比特数
%如果总的比特数没有分配完，则进入while循环
while(bit_total<Rb)
        if(bit_alloc(index_min)~=M)%询问在一个子载波上分配的比特是否超过M比特
%%计算在每个子载波上每增加1比特带来功率的增加
        for i=1:N_subc
                  power_add(i)=(f_mpsk(bit_alloc(i)+1,BER,Noise)-…
                        f_mpsk(bit_alloc(i),BER,Noise))/H(i)^2;
%             power_add(i)=(f_mqam(bit_alloc(i)+1,BER,Noise)-…
%             f_mqam(bit_alloc(i),BER,Noise))/H(i)^2;
            end
%给增加功率最小的子载波上加1比特      
            min_add=min(power_add);
            index_min=find(power_add==min_add,1);
            bit_alloc(index_min)=bit_alloc(index_min)+1;
            bit_total=sum(bit_alloc);
        else
            %如果在一个子载波上分配的比特为M就不在给这个子载波分配比特，赋这个子载波功率无穷大
            power_add(index_min)=inf; 
%继续找使得增加功率为最小的（实际上是第二小的）子载波，并将其上的比特数加1     
            min_add=min(power_add);
            index_min=find(power_add==min_add,1);
            bit_alloc(index_min)=bit_alloc(index_min)+1;
            bit_total=sum(bit_alloc);
        end
end
%功率分配。根据最终分配的比特数，计算每个子载波上的功率。 
for i=1:N_subc
         power_alloc(i)=f_mpsk(bit_alloc(i),BER,Noise)/H(i)^2;
%power_alloc(i)=f_mqam(bit_alloc(i),BER,Noise)/H(i)^2;
end
end
f_mpsk子程序。程序名称：f_mpsk.m
%计算MPSK信号的功率
function power=f_mpsk(b,Pe,N_psd)
%输入参数：b为比特数；Pe为误码率；N_psd为每个子载波上的功率谱密度。
%输出参数：power为MPSK信号的功率。
switch b
        case 0
            power=0;
        case 1        %2PSK功率
            power=N_psd/2*(Qinv(Pe))^2;
        case 2        %QPSK功率
            power=N_psd*(Qinv(1-sqrt(1-Pe)))^2;
        otherwise    %其他PSK功率
            power=N_psd/2*(Qinv(Pe/2)/sin(pi/2^b))^2;
end
f_mqam子程序。程序名称：f_mqam.m
%计算MQAM信号的功率
function power=f_mqam(c,Pe,N_psd)
%输入参数：c为比特数；Pe为误码率；N_psd为每个子载波上的功率谱密度。
%输出参数：power为MQAM信号的功率。
if(mod(c,2)~=0)
    %要求比特数为偶数。通常使用16QAM信号。
        error('The number of bit must be Even in MQAM ')
end
if c==0
        power=0;
else
M=2^c;%进制数。
power=N_psd/3*(M-1)*(Qinv(Pe*sqrt(M)/(4*(sqrt(M)-1))))^2;%计算MQAM信号的功率
end
end
Q子程序。程序名称：Q.m
function y=Q(x)
y=.5*erfc(x/sqrt(2));
end


Qinv子程序。程序名称：Qinv.m
function y=Qinv(x)
y=sqrt(2)*erfcinv(2*x);
end

仿真结果见图7-6。
 
图7-6 Hughes-Hartogs 算法比特与功率分配图

