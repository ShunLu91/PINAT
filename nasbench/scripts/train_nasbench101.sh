if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi
if [ ! -d "./checkpoints" ]; then
  mkdir ./checkpoints
fi
if [ ! -d "./results" ]; then
  mkdir ./results
fi

# arguments
IDX=0
Loss=mse
Bench=101
Epochs=300
Model=PINAT
Dataset=cifar10
Train_batch_size=10
Eval_batch_size=10240
Train_Split_List=(100 172 424 424 4236)
Eval_Split_List=(all all 100 all all)
Script=train_nasbench.py

for((t=0; t<${#Train_Split_List[*]}; t++)); do
  # set gpu and data splits
  GPU=$((${IDX} % 8))
  let IDX+=1
  Train_Split=${Train_Split_List[t]}
  Eval_Split=${Eval_Split_List[t]}
  EXP_Name=${Bench}_${Dataset}_${Model}_${Loss}_t${Train_Split}_v${Eval_Split}_e${Epochs}_bs${Train_batch_size}

  # run
  nohup python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU \
    --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
    --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} \
    > logdir/$EXP_Name.log 2>&1 &

  echo "GPU:$GPU EXP:$EXP_Name"
  if [ $GPU = 7 ] ; then
      echo "sleep 30s"
      sleep 30s
  fi

done

tail -f logdir/$EXP_Name.log
