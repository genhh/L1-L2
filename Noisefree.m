
function [alpha_final,sucess_all] = Noisefree(ksta,num,F,numiter,normal)

M = 100; N = 1500;      % matrix dimension M-by-N
%ksta=5;
%kend=25;
K = ksta:ksta+num-1;                 % sparsity
%F = 5;%20;                 % larger for higher coherence


%% parameters
pm.lambda = 1e-7; pm.maxit = 5*N;
pmL1 = pm; pmL1.maxit = 2*N;

% how to dynamically update alpha
if F>10
    pm.alpha_update = 2;
else
    pm.alpha_update = 1;
end


%% highly coherent matrix
A = zeros(M,N);
r = rand(M,1);
l = 1:N;
for k = 1:M
        A(k,:) = sqrt(2/M) * cos(2 * pi * r(k) * (l-1) / F); %oversample DCT
end
        
A = A/norm(A);

for i = 1:num
    fprintf("num:")
    i
    sucess = zeros(4,1);
    for iter = 1:numiter
        fprintf("iter:")
        iter
        %% sparse vector with minimum separation
        supp        = randsample_separated(N,K(i),2*F);
        x_ref       = zeros(N,1);
        xs          = randn(K(i),1);
        x_ref(supp) = xs;
        b           = A * x_ref;

        %% initialize by an inaccurate L1 solution
        [x1,output] = CS_L1_uncon_ADMM(A,b,pmL1); %线1
        pm.x0       = x1;

        [xDCA,outputDCA] = CS_L1L2_uncon_DCA(A,b,pm); %线2
        [xADMM,outputADMM] = CS_L1L2_uncon_ADMM(A,b,pm); %线3
        [xADMMweighted,outputweight]   = CS_L1L2_uncon_ADMMweighted(A,b,pm,normal); %线4
        
        %% exact L1 solution as baseline
        [x1,output] = CS_L1_uncon_ADMM(A,b,pm);

        res = log10([norm(x1-x_ref), norm(xDCA-x_ref), norm(xADMM-x_ref), norm(xADMMweighted-x_ref)]/norm(x_ref)); %小于-3认为成功
        sucess(res<-3) =sucess(res<-3)+1;
    end
    sucess_all(i,:)= sucess;
end

alpha_final = outputweight.alpha;


end