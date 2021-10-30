
partype = [1 2 1 2];
parlam = [.01 .01 .1 .001];
figure
for i=1:4
[outputDCA,outputFB,outputADMM] = testConstructed(partype(i),parlam(i));
subplot(2,2,i)

plot(log10(outputDCA.err), 'r', 'LineWidth',2)
hold on
plot(log10(outputFB.err),'k-.', 'LineWidth',2);
hold on
plot(log10(outputADMM.err),'b--', 'LineWidth',2)
LEG = legend('DCA',  'FBS','ADMM', 'location', 'NorthEast');
end