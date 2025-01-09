current_data=$(date "+%Y_%m_%d__%H_%M_%S")
for i in {1..5}
do
  echo "Sequence: $i [$current_data]"
  python generate_data.py \
  --carla_path /home/enrico/Projects/Carla/CARLA_0.9.15/ \
  --town 10 \
  --num_of_vehicle 0 \
  --num_of_walkers 0 \
  --dataset_path /media/enrico/Enrico_Datasets/carla_events/"$current_data" \
  --sequence_id "$i"
  exit_code=$?
  if [ $exit_code -eq 99 ]; then
    exit 0
  fi
done
