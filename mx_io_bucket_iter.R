########################################################################################
### Create a bucketed iterator
########################################################################################

mx_io_bucket_iter<- function(buckets, batch_size, data_mask_element=0, shuffle=F){
  
  bucket_names<- names(buckets)
  
  init<- function(){
    epoch<<- 0
    batch<<- 0
    bucket_plan<<- NULL
    bucketID<<-NULL
  }
  
  bucket_plan_fun<- function(){
    bucketID<<-batch
    return(bucketID)
  }  
  
  bucket_fun<- function(){
    bucketID<- bucketID
    return(bucketID)
  }
  
  reset<- function(){
    buckets_nb<- length(bucket_names)
    buckets_id<- 1:buckets_nb
    buckets_size<- sapply(buckets, function(x) last(dim(x$data)))
    batch_per_bucket<- floor(buckets_size/batch_size)
    
    ### Number of batches per epoch given the batch_size
    batch_per_epoch<<- sum(batch_per_bucket)
    epoch<<- epoch+1
    batch<<- 0
    
    set.seed(123)
    bucket_plan_names<- sample(rep(names(batch_per_bucket), times=batch_per_bucket))
    bucket_plan<<- ave(bucket_plan_names==bucket_plan_names, bucket_plan_names, FUN=cumsum)
    names(bucket_plan)<<- bucket_plan_names
    
    ### Return first BucketID at reset for initialization of the model
    bucketID<<- bucket_plan[1]
    bucket_fun()
    
    #### Shuffling of observations within a bucket
    if (shuffle==T) {
      buckets<<- lapply(buckets, function(x){
        set.seed(123)
        shuffle_id<- sample(ncol(x$data))
        list(data=x$data[, shuffle_id], label=x$label[shuffle_id])
      })
    }
    
    return(!is.null(bucket_plan))
  }
  
  iter.next<- function(){
    batch<<- batch+1
    bucketID<<- bucket_plan[batch]
    bucket_fun()
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    
    ### bucketID is a named integer: 
    ###   the integer indicates the batch id for the given bucket (used to fetch appropriate samples within the bucket)
    ###   the name is the a character containing the sequence length of the bucket (used to unroll the rnn to appropriate sequence length)
    idx<- (bucketID-1)*(batch_size)+(1:batch_size)
    data<- buckets[[names(bucketID)]]$data[,idx, drop=F]
    data_mask<- as.integer(names(bucketID)) - apply(data==data_mask_element, 2, sum)
    data_mask_array<- (!data==0)
    label<- buckets[[names(bucketID)]]$label[idx]
    return(list(data=mx.nd.array(data), label=mx.nd.array(label), data_mask=mx.nd.array(data_mask), data_mask_array=mx.nd.array(data_mask_array)))
  }
  
  return(list(init=init, reset=reset, iter.next=iter.next, value=value, bucketID=bucket_fun, bucket_names=bucket_names, batch_size=batch_size))
}


