current_data=$(date "+%Y_%m_%d__%H_%M_%S")
for i in {1..300}
do
  echo "Sequence: $i [$current_data]"
  python generate_data.py \
  --carla_path /home/carla/CarlaUE5/ \
  --town 10 \
  --num_of_vehicle 0 \
  --num_of_walkers 0 \
  --dataset_path /home/carla/"$current_data" \
  --sequence_id "$i" \
  --show_carla_window
  exit_code=$?
  if [ $exit_code -eq 99 ]; then
    exit 0
  fi
done
