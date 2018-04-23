

relu<-function(x){
  ifelse(x>0,x,0)
}


nn.ff2<-function (nn, batch_x) 
{
  m <- nrow(batch_x)
  if (nn$visible_dropout > 0) {
    nn$dropout_mask[[1]] <- dropout.mask(ncol(batch_x), nn$visible_dropout)
    batch_x <- t(t(batch_x) * nn$dropout_mask[[1]])
  }
  nn$post[[1]] <- batch_x
  i<-2
  for (i in 2:(length(nn$size) - 1)) {
    nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) + 
                       nn$B[[i - 1]])
    if (nn$activationfun == "sigm") {
      nn$post[[i]] <- sigm(nn$pre[[i]])
    }
    else if (nn$activationfun == "tanh") {
      nn$post[[i]] <- tanh(nn$pre[[i]])
    }
    else if (nn$activationfun == "relu") {
      nn$post[[i]] <- relu(nn$pre[[i]])
    }
    else if (nn$activationfun == "linear") {
      nn$post[[i]] <- (nn$pre[[i]])
    }
    else {
      stop("unsupport activation function!")
    }
    if (nn$hidden_dropout > 0) {
      nn$dropout_mask[[i]] <- dropout.mask(ncol(nn$post[[i]]), 
                                           nn$hidden_dropout)
      nn$post[[i]] <- t(t(nn$post[[i]]) * nn$dropout_mask[[i]])
    }
  }
  dim(nn$W[[i - 1]])
  dim(t(nn$post[[(i - 1)]]))
  
  i <- length(nn$size)
  nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) + 
                     nn$B[[i - 1]])
  if (nn$output == "sigm") {
    nn$post[[i]] <- sigm(nn$pre[[i]])
    
    
  } else if (nn$output == "linear") {
    nn$post[[i]] <-  nn$pre[[i]] 
    
  } else if (nn$output == "softmax") {
    # nn$post[[i]] <- 2.71818^(nn$pre[[i]])
    nn$post[[i]] <- exp(nn$pre[[i]])
    if(sum(is.infinite(nn$post[[i]]))>0){
      infin<-is.infinite(nn$post[[i]])
    nn$post[[i]][infin]<-1
    nn$post[[i]][!infin]<-0
    }else{
    nn$post[[i]] <- nn$post[[i]]/rowSums(nn$post[[i]])
    }
    # nn$post[[i]][is.na(nn$post[[i]] )]<-1
  }
  
  nn
}

nn.bp<-function (nn) 
{
  n <- length(nn$size)
  d <- list()
  if (nn$output == "sigm") {
    d[[n]] <- -nn$e * (nn$post[[n]] * (1 - nn$post[[n]]))
  }
  else if (nn$output == "linear" || nn$output == "softmax") {
    d[[n]] <- -nn$e
  }
  for (i in (n - 1):2) {
    if (nn$activationfun == "sigm") {
      d_act <- nn$post[[i]] * (1 - nn$post[[i]])
    }
    else if (nn$activationfun == "tanh") {
      d_act <- 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn$post[[i]]^2)
    }
    else if (nn$activationfun == "relu") {
      d_act <-  ifelse(nn$post[[i]]>0,1,0)
    }
    d[[i]] <- (d[[i + 1]] %*% nn$W[[i]]) * d_act
    if (nn$hidden_dropout > 0) {
      d[[i]] <- t(t(d[[i]]) * nn$dropout_mask[[i]])
    }
  }
  for (i in 1:(n - 1)) {
    dw <- t(d[[i + 1]]) %*% nn$post[[i]]/nrow(d[[i + 1]])
    dw <- dw * nn$learningrate
    if (nn$momentum > 0) {
      nn$vW[[i]] <- nn$momentum * nn$vW[[i]] + dw
      dw <- nn$vW[[i]]
    }
    nn$W[[i]] <- nn$W[[i]] - dw
    db <- colMeans(d[[i + 1]])
    db <- db * nn$learningrate
    if (nn$momentum > 0) {
      nn$vB[[i]] <- nn$momentum * nn$vB[[i]] + db
      db <- nn$vB[[i]]
    }
    nn$B[[i]] <- nn$B[[i]] - db
  }
  nn
}



sigm<-function (x) 
{
  1/(1 + exp(-x))
}


relu<-function(x){
  ifelse(x>0,x,0)
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





# SFFFF|FFFFF
# FFFFF|FFFFF
# FFFFF|FFFFF
# HHHHH|FFHFF
# FHFHH|FFFFH
# FFFFF|FFFFF
# HFFFF|HHHHH
# FHFFF|FFFFF
# FFFFF|FFFFF
# FFFFF|FFFFG
### init

stm<-matrix(1:100,ncol=10,nrow=10,byrow=T)
state_size<-ncol(stm)*nrow(stm)
next_where<-function(index){ 
  zero<-rep(0,state_size)
  zero[index]<-1
  zero
}


return_reward<-function(state,current_state){
  re_index<-which(state==1)
  
  if(  re_index==100){
    reward<- 100# episode end
    done<-T
  }
  else if(re_index %in% c(31,32,33,34,35,38,42,44,45,50,61,66,67,68,69,70,72)){
    # else if(re_index==12){
    reward<- -1
    done<-F
  }else{
    reward <- -0.1
    done<-F
  }
  if(re_index==which(current_state==1)){
    reward<-reward*2
  }
  # xx<-ceiling(re_index/ 10) ## row
  # yy<-re_index %% 10  ## col
  # yy<-ifelse(yy ==0,10,yy)
  # reward_weight<-sqrt((yy-10)^2+(xx-10)^2) #weigthed reward by distance from current state to goal
  # reward<-reward*reward_weight/sqrt(200)
  c(reward,done)
  
}



action<-c("left","right","down","up")
action2<-c("l","r","d","u")


wv<-rep(0,nrow(stm))
yv<-rep(0,ncol(stm))
convert_coord<-function(x){
  wv2<-wv
  yv2<-yv
  wv2[x[1]]<-1
  yv2[x[2]]<-1
  c(wv2,yv2)
}


reverse_coord<-function(x){
  # re_index<-which(current_state==1)
  xx<-ceiling(x/ ncol(stm)) ## 행
  yy<-x %% ncol(stm)  ## 열
  yy<-ifelse(yy ==0,ncol(stm),yy)
  
  
  convert_coord(c(xx,yy))
}



coord<-function(state){
  re_index<-which(state==1)
  xx<-ceiling(re_index/ ncol(stm)) ## 행
  yy<-re_index %% ncol(stm)  ## 열
  yy<-ifelse(yy ==0,ncol(stm),yy)
  
  c(xx,yy)
}



todata<-sapply(1:100,reverse_coord)

  
  {
    
    input_dim<-ncol(stm)+nrow(stm)
    hidden<-c(30)
    output_dim<-4
    size <- c(input_dim, hidden, output_dim)
    activationfun<-"relu"
    # activationfun<-"tanh"
    output<-"linear"
    output<-"softmax"
    batchsize<-30
    momentum<-0
    learningrate_scale<-1
    hidden_dropout = 0
    visible_dropout = 0
    numepochs = 10
    learningrate<-0.001
    
    
    vW <- list()
    vB <- list()
    W <- list()
    B <- list()
    
    
    
    for (i in 2:length(size)) {
      W[[i - 1]] <- matrix(rnorm(size[i] * size[i - 1],0,2/input_dim),
                           c(size[i], size[i - 1]))
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
  
  {
    
    input_dim<-ncol(stm)+nrow(stm)
    hidden<-c(30)
    output_dim<-1
    size <- c(input_dim, hidden, output_dim)
    activationfun<-"relu"
    # activationfun<-"tanh"
    output<-"linear"
    
    batchsize<-30
    momentum<-0
    learningrate_scale<-0.99
    hidden_dropout = 0
    visible_dropout = 0
    numepochs = 10
    learningrate<-0.01
    
    
    vW <- list()
    vB <- list()
    W <- list()
    B <- list()
    
    
    
    for (i in 2:length(size)) {
      W[[i - 1]] <- matrix(rnorm(size[i] * size[i - 1],0,2/input_dim),
                           c(size[i], size[i - 1]))
      B[[i - 1]] <- runif(size[i], min = -0.1, max = 0.1)
      vW[[i - 1]] <- matrix(rep(0, size[i] * size[i - 1]), 
                            c(size[i], size[i - 1]))
      vB[[i - 1]] <- rep(0, size[i])
    }
    vg<- list(input_dim = input_dim, output_dim = output_dim, 
              hidden = hidden, size = size, activationfun = activationfun, 
              learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
              hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
              output = output, W = W, vW = vW, B = B, vB = vB)
    
    
    
    
  }
  
  
  
  epoch<-10
  mini_batch<-100
  
  init_data<-convert_coord(c(1,1))
  dis_f<-0.99
  reward_list<-c()
  final_action_list<-list()
  step_list<-c()
  bi<-1
  q_table<-list()
  mrlist<-c()
  L<-c()
  # st
  p<-1
  r<-1
  bi2<-1
  th<-0.1


  for(i in 1:30000){
    
    total_r<-0
    episode_done<-0
    
    qn1<-nn.ff2(qn1,t(init_data))
    step<-1
    action_list<-NULL
    st<-c(1,1)
    th<-th*0.999
   
    while(episode_done==0){
      if(step >1){
        qn1<-nn.ff2(qn1,t(cov_next_state))
        action_prob<-qn1$post[[length(size)]]
        current_state<-next_state
        store_current_state<-t(cov_next_state)
      }else{
        current_state<- next_where(c(1,1))
        store_current_state<-t(init_data)
        action_prob<-qn1$post[[length(size)]]
        
     
      }
      
      
        (next_action<-  action[sample(1:4,1,prob=action_prob)])
  
      action_list<-c(action_list,next_action)
      st<-move(st,next_action)
      state_index<-stm[st[1],st[2]]
   
      next_state<-next_where(state_index)
      
      re_ep<-return_reward(next_state,current_state) ## next state에 대한 reward와 epsiode 종료 여부 
      cov_next_state<-convert_coord(st)
      
      total_r<-total_r+re_ep[1]
      episode_done<-re_ep[2]
        
      
      y <- next_action
      ya<-rep(0,4)
      ya[action %in% y] <-1
      err<-ya - (action_prob)
    
      step<-step+1
  
      vg<-nn.ff2(vg,store_current_state)
      v_<-vg$post[[length(size)]]
      vg<-nn.ff2(vg,t(cov_next_state))
      v_next<-vg$post[[length(size)]]
      td_error<-re_ep[1]+dis_f*v_next-v_
  
      if(episode_done){
        td_error<-re_ep[1]-v_
       
      }
 
      
      vg<-nn.ff2(vg,store_current_state)
     
      vg$e<-td_error
      vg<-nn.bp(vg)
    
      action_prob[action %in% y]<-log(action_prob[action %in% y]+0.00001)
      ya<-action_prob*as.numeric(td_error)
      qn1$e<- -(as.matrix(ya))
      qn1<-nn.bp(qn1)
      
      
      if(step == 100){
        cat("\n",i,"번째 episode-",step)
        step_list<-c(step_list,step)
        final_action_list[[i]]<-action_list
        reward_list<-c(reward_list,total_r)
        cat("\n final coordinate")
        print(coord(next_state))
        ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
   
        break;
      }
      
      
      
      if(episode_done==1){
        
        cat("\n",i,"번째 episode-",step)
        cat("\n final coordinate")
        print(coord(next_state))
        
        step_list<-c(step_list,step)
        final_action_list[[i]]<-action_list
        reward_list<-c(reward_list,total_r)
        ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
         break;
      }
    }
    
    if(i %% 100 ==0){
      nn.ff2(qn1,t(todata))$post[[length(qn1$size)]]
      ad<-apply(nn.ff2(qn1,t(todata))$post[[length(qn1$size)]],1,which.max);ad
      q_table[[i]]<-matrix(action2[ad],ncol=sqrt(state_size),byrow=T)
      print(matrix(action2[ad],ncol=sqrt(state_size),byrow=T))
    }
    
  }

final_action_list[which.max(reward_list)]


