SEED_MAX=$1

SEEDS=()

rm -f seeds.txt

for ((seeds=0; seeds<SEED_MAX; seeds++)); do
         ./hca ../instances/le450_5a.col -k 14 -C 1 > temp.txt
	sed -n 's/Seed..........................: //p' temp.txt > temp2.txt
	seed=`cut -d " " -f 3 temp2.txt`
	equal=0;
	for ((s=0; s<${#SEEDS[@]}; s++)); do
		if [[ ${SEEDS[${s}]} == $seed ]]; then 
			equal=1;
			break;
		fi
	done
	if [[ $equal == 0 ]]; then
		SEEDS=("${SEEDS[@]}" $seed)
	fi
done

rm temp.txt temp2.txt

echo "SEEDS=("${SEEDS[@]}")" > seeds.txt 
echo "" >> seeds.txt
echo "" >> seeds.txt

for ((s=0; s<${#SEEDS[@]}; s++)); do
	echo ${SEEDS[${s}]} >> seeds.txt
done
