clear; close all
clc

%% parameter settings
%M = 250;
N = 512;   % matrix dimension M-by-N
K = 130;            % sparsity
tic
for M=230:330
    M
for trial = 1:5
        trial
   
        A   = randn(M,N); 
        A   = orth(A')';    % normalize each column to be zero mean and unit norm
        
        %% construct sparse ground-truth 
        x_ref       = zeros(N,1); % true vector
        xs          = randn(K,1);
        idx         = randperm(N);
        supp        = idx(1:K);
        x_ref(supp) = xs;
        As          = A(:,supp);

        sigma       = 0.1;
        b           = A * x_ref + sigma * randn(M,1); 
        
        MSEoracle(trial) = sigma^2 * trace(inv(As' * As));


        %% parameters
        pm.lambda = 0.08;
        pm.delta = normest(A*A',1e-2)*sqrt(2);
        pm.xg = x_ref; 
        pmL1 = pm; 
        pmL1.maxit = 2*N;
        
        
        %% initialization with inaccurate L1 solution
        x_half  = CS_L_half_uncon_ADMM(A,b,pmL1); 
        x1      = CS_L1_uncon_ADMM(A,b,pmL1); 
        pm.x0   = x1;   
        
        
        %% L1-L2 implementations
        xDCA            = CS_L1L2_uncon_DCA(A,b,pm);
        xADMM           = CS_L1L2_uncon_ADMM(A,b,pm);
        xADMMweighted   = CS_L1L2_uncon_ADMMweighted(A,b,pm,2);

        pmFB = pm; pmFB.delta = 1;
        [xFB,outputFB] = CS_L1L2_uncon_FBweighted(A,b,pmFB);

        
        %% compute MSE
        xall = [x_half x1 xDCA, xADMM, xADMMweighted,xFB];
        for k = 1:size(xall,2)  
            xx = xall(:,k);
            MSE(trial, k) =norm(xx-x_ref);
        end

end
sort(MSEoracle,1)
sort(MSE,1)

res(M-229,:)=[mean(MSEoracle(2:length(MSE)-1),2), mean(MSE(2:length(MSE)-1,:),1)];

end
toc
disp( ['运行时间: ',num2str(toc) ] );

M=230:330;

figure
plot(M,smooth(res(:,1)),'b-')
hold on
plot(M,smooth(res(:,2)),'b-.')
hold on
plot(M,smooth(res(:,3)),'r-')
hold on
plot(M,smooth(res(:,4)),'c-')
hold on
plot(M,smooth(res(:,5)),'m-.')
hold on
plot(M,smooth(res(:,6)),'k-')
hold on
plot(M,smooth(res(:,7)),'r-.')
hold on
LEG = legend('orale',  'L1/2','L1-L2DCA','L1-L2ADMM','L1-L2ADMMweight','L1-L2FBS', 'location', 'NorthEast');