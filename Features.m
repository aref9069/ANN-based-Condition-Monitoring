
load matlab2_2_v1x;

for i=1:133
    
X=xy(:,i);


%me is mean
me=mean(X);
% variance is v2
v2=sum((X-mean(X)).^2)/length(X);
% four moments is b
b=sum((X-mean(X)).^4)/length(X);

% kurtosis factor is k
k=b/(v2.^(2));

KURTOSIS(1,i)=k;
% rms is root mean square
rms=[sqrt(sum((X).^2)/length(X))];
RMS(1,i)=rms;

end

% rms=@(x) [sqrt(sum((xy).^2)/length(xy))];
% rms=rms(xy)
% 
% % four moments is b
% b=sum((xy-mean(xy)).^4)/length(xy)
% 
% % variance is v2
% v2=sum((xy-mean(xy)).^2)/length(xy)
% 
% % kurtosis factor is k
% k=b/(v2.^(2))


% plot(rms)
% grid on
% legend('RMS')
% title('Trend of RMS')
% xlabel('Number of Data')
% ylabel('Amplitude of RMS')
