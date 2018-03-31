


source("nn.bp.R")
source("nn.ff2.R")
source("relu.R")
source("sigm.R")


coord<-function(state){
  re_index<-which(state==1)
  xx<-ceiling(re_index/ 10) ## 행
  yy<-re_index %% 10  ## 열
  yy<-ifelse(yy ==0,10,yy)
  c(xx,yy)
}
move<-function(x,action){
  
  if(action == "left"){
    if(x[2]-1<1){
      x
    }else{
      x[2]<-x[2]-1
      x
    }
  }
  if(action == "right"){
    if(x[2]+1>ncol(stm)){
      x
    }else{
      x[2]<-x[2]+1
      x
    }
  }
  if(action == "up"){
    if(x[1]-1<1){
      x
    }else{
      x[1]<-x[1]-1
      x
    }
  }
  if(action == "down"){
    if(x[1]+1>nrow(stm)){
      x
    }else{
      x[1]<-x[1]+1
      x
    }
  }
  x
}
next_where<-function(index){ 
 zero<-rep(0,100)
 zero[index]<-1
 zero
  
  
}


#######state matrix 
stm<-matrix(1:100,ncol=10,nrow=10,byrow=T)

return_reward<-function(state){
  re_index<-which(state==1)
  
  if(  re_index==100){
    reward<- 100# episode end
    done<-T
  }
  else if(re_index==12 | re_index==14 | re_index==15 |re_index==20|re_index==31|re_index==38|re_index==42|re_index==44|re_index==45     |re_index==50|re_index==61|re_index==66 |re_index==68|re_index==72|re_index==80|re_index==91|re_index==96 ){
    reward<- -5 
    done<-F
  }else{
    reward <- -1
    done<-F
  }
  
  xx<-ceiling(re_index/ 10) ## row
  yy<-re_index %% 10  ## col
  yy<-ifelse(yy ==0,10,yy)
  reward_weight<-sqrt(162)-sqrt((yy-10)^2+(xx-10)^2) #weigthed reward by distance from current state to goal
  reward<-reward+reward_weight*0.05
  c(reward,done)
  
}
action<-c("left","right","down","up")
## 10 x 10  frozen lake problem
# S : start, F : Frozen, H : Hole, G : Goal 
# SFFFF|FFFFF
# FHFHH|FFFFH
# FFFFF|FFFFF
# HFFFF|FFHFF
# FHFHH|FFFFH
# FFFFF|FFFFF
# HFFFF|HFHFF
# FHFFF|FFFFH
# FFFFF|FFFFF
# HFFFF|HFFFG
### initialize neural network

  {
    
    input_dim<-100
    hidden<-c(30)
    output_dim<-4
    size <- c(input_dim, hidden, output_dim)
     activationfun<-"relu"
    output<-"linear"

    momentum<-0
    learningrate_scale<-1
    hidden_dropout = 0
    visible_dropout = 0
    numepochs = 10
    learningrate<-0.1
    
    vW <- list()
    vB <- list()
    W <- list()
    B <- list()
    
    
    
    for (i in 2:length(size)) {
      W[[i - 1]] <- matrix(runif(size[i] * size[i - 1], 
                                 min = -0.1, max = 0.1), c(size[i], size[i - 1]))
      B[[i - 1]] <- runif(size[i], min = -0.1, max = 0.1)
      vW[[i - 1]] <- matrix(rep(0, size[i] * size[i - 1]), 
                            c(size[i], size[i - 1]))
      vB[[i - 1]] <- rep(0, size[i])
    }
    qn1<- list(input_dim = input_dim, output_dim = output_dim, 
               hidden = hidden, size = size, activationfun = activationfun, 
               learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
               hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
               output = output, W = W, vW = vW, B = B, vB = vB)
    
  }
  



  init_data<-c(1,rep(0,99))
  dis_f<-0.99
  reward_list<-c()
  final_action_list<-list()
  step_list<-c()
  q_table<-list()
  
  for(i in 1:10000){
    total_r<-0
    episode_done<-0
    da<-diag(100)
    diag(da)<-init_data
    qn1<-nn.ff2(qn1,da)
    step<-1
    action_list<-NULL
    st<-c(1,1)
    
    
  while(episode_done==0){
    
  if(step >1){
   da<-diag(100)
   diag(da)<-next_state
   qn1<-nn.ff2(qn1,da)
   action_index<-which.max(qn1$post[[length(size)]][next_state==1,])
   current_state<-next_state

  }else{
  current_state<-init_data
  action_index<-which.max(qn1$post[[length(size)]][1,])
  }
  
  
  
  ### max step
  if(step == 300){
  cat("\final location")
  print(coord(next_state))
  ad<-apply(nn.ff2(qn1,diag(100))$post[[3]],1,which.max);ad
  print(matrix(action[ad],ncol=10,byrow=T))
  step_list<-c(step_list,step)
  final_action_list[[i]]<-action_list
  reward_list<-c(reward_list,total_r)
  ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
  break;

  }

  th<-1/(i/50+10)
  if(runif(1) < th){ ## e-greedy search
  action_index<-sample(1:4,1)
  next_action<-  action[action_index]
  }else{
  next_action<-action[action_index] ### action
  }

  action_list<-c(action_list,next_action)
  st<-move(st,next_action)
  state_index<-stm[st[1],st[2]]
  if(state_index==100){
    # break
  }
  next_state<-next_where(state_index)
  
  #### target value  
  da<-diag(100)
  diag(da)<-next_state
  qn2<-nn.ff2(qn1,da)
  re_ep<-return_reward(next_state) ## get a reward and Whether the episode ends for action(next state)
  qv<-qn2$post[[length(qn2$size)]] 
 
  if(state_index==100){ ## if episonde done
  
    re<-    rep(0,100)
    qv[current_state==1,action_index] <- re_ep[1]
    true_y<-qv
    
  }else{
   re<- rep(0,100)
   qv[current_state==1,action_index] <-  re_ep[1]+ dis_f * max(qv[current_state==1])
   true_y<- qv
  }
  qn1$e<-   true_y-qn1$post[[length(size)]]
  qn1$L[step]<- mean(qn1$e^2)
  
  
  
  ######## qnetwork backpropagation
  qn1<-nn.bp(qn1)

  
  ## totail reward
  total_r<-total_r+re_ep[1] 
  
  ## episode end??
  episode_done<-re_ep[2]
  step<-step+1

  if(episode_done==1){
    
    cat("\n",i," epsode-",step)
    ad<-apply(nn.ff2(qn1,diag(100))$post[[3]],1,which.max);ad
    q_table[[i]]<-matrix(action[ad],ncol=10,byrow=T)
    
    cat("\final location")
    print(coord(next_state))
    print(matrix(action[ad],ncol=10,byrow=T))
    
    step_list<-c(step_list,step)
    final_action_list[[i]]<-action_list
    reward_list<-c(reward_list,total_r)
    
    ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
    break;
  }
  
    }
  
  }
  
  ts.plot(step_list,main="step")
  ts.plot(reward_list,main="Qnetwork-reward")
  