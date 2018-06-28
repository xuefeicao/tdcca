library(PMA)
library(R.matlab)
library(CAPIT)
bb <- seq(0.01, 1, by=0.05)
lambda_x <- bb
lambda_y <- lambda_x
ff <- c('100_20_20_200', '100_50_50_200', '20_20_20_200', '50_100_100_200')
ff <- c('100_20_20_200_1', '100_50_50_200_1')
sim_n <- 50
for (f in ff){
  cc <- matrix(0,sim_n)
  tmp <- strsplit(f, '_')[[1]]
  n1 <- tmp[2]
  n2 <- tmp[3]
  T <- strtoi(tmp[4])
  W_1_all <- array(0, c(n1, T, sim_n))
  W_2_all <- array(0, c(n2, T, sim_n))
  f_name <- paste0('/gpfs/data/xl6/xuefei/research_Rossi_1/data/tcca/test_with_val/test_para_1/', f)
  for (sim in 1:(sim_n)){
  W_1 <- array(0, c(n1, T))
  W_2 <- array(0, c(n2, T))
  tmp_c = 0 
  for (t in 1:T){
    f_sim <- paste0(paste0(paste0(f_name, '/sim_'), sim-1),'/full_data/0/')

    x <- readMat(paste0(f_sim, gsub('l', t-1, 'RSCCA_1/0_l.mat')))$'data'
    y <- readMat(paste0(f_sim, gsub('l', t-1, 'RSCCA_1/1_l.mat')))$'data'
    
    res <- CCA.permute(x, y,  typex='standard', typez='standard', penaltyx=lambda_x, penaltyz=lambda_y)
    lx <- res$bestpenaltyx
    ly <- res$bestpenaltyz
    
    res <- CCA(x, y,  typex='standard', typez='standard', penaltyx=lx, penaltyz=ly)
    W_1[,t] <- res$u
    W_2[,t] <- res$v 
    tmp_c = tmp_c + cor(x%*%res$u, y%*%res$v)
  
  }

  W_1_all[,,sim] <- W_1
  W_2_all[,,sim] <- W_2
  cc[sim] <- tmp_c/(T)
  for (t in 2:T){
    if (sum(abs(W_1_all[,t, sim]-W_1_all[,t-1,sim]) + abs(W_2_all[,t, sim]-W_2_all[,t-1,sim])) > sum(abs(W_1_all[,t, sim]+W_1_all[,t-1,sim]) + abs(W_2_all[,t, sim]+W_2_all[,t-1,sim])))
    { W_1_all[,t,sim] <- - W_1_all[,t,sim]
      W_2_all[,t,sim] <- - W_2_all[,t,sim]
  }
  
  }
  }
  writeMat(paste0(f_name, '/RSCCA.mat'), W_1=W_1_all, W_2=W_2_all, cc=cc)

}

for (f in ff){
  cc <- matrix(0,sim_n)
  tmp <- strsplit(f, '_')[[1]]
  n1 <- tmp[2]
  n2 <- tmp[3]
  T <- strtoi(tmp[4])
  W_1_all <- array(0, c(n1, T, sim_n))
  W_2_all <- array(0, c(n2, T, sim_n))
  f_name <- paste0('/gpfs/data/xl6/xuefei/research_Rossi_1/data/tcca/test_with_val/test_para_1/', f)
  for (sim in 1:(sim_n)){
  W_1 <- array(0, c(n1, T))
  W_2 <- array(0, c(n2, T))
  tmp_c = 0 
  for (t in 1:T){
    f_sim <- paste0(paste0(paste0(f_name, '/sim_'), sim-1),'/full_data/0/')

    x <- readMat(paste0(f_sim, gsub('l', t-1, 'RSCCA_1/0_l.mat')))$'data'
    y <- readMat(paste0(f_sim, gsub('l', t-1, 'RSCCA_1/1_l.mat')))$'data'
    scl <- 1000
    res <- CAPIT(x*scl, y*scl, select.covariance = 'floss', method = 'tapering')

    W_1[,t] <- res$resOLS$ThetaOLS
    W_2[,t] <- res$resOLD$EtaOLS
    tmp_c = tmp_c + cor(x%*%res$u, y%*%res$v)
  
  }

  W_1_all[,,sim] <- W_1
  W_2_all[,,sim] <- W_2
  cc[sim] <- tmp_c/(T)
  for (t in 2:T){
    if (sum(abs(W_1_all[,t, sim]-W_1_all[,t-1,sim]) + abs(W_2_all[,t, sim]-W_2_all[,t-1,sim])) > sum(abs(W_1_all[,t, sim]+W_1_all[,t-1,sim]) + abs(W_2_all[,t, sim]+W_2_all[,t-1,sim])))
    { W_1_all[,t,sim] <- - W_1_all[,t,sim]
      W_2_all[,t,sim] <- - W_2_all[,t,sim]
  }
  
  }
  }
  writeMat(paste0(f_name, '/RCAPIT.mat'), W_1=W_1_all, W_2=W_2_all, cc=cc)

}


