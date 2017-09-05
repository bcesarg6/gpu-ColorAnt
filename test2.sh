#!/bin/bash
#Author Bruno Cesar Puli Dala Rosa, bcesar.g6@gmail.com
#1/06/2016 17:58
#Script 2: automatizador de testes para o gpu_Colorant

#le450_15d -> 15 | latin_square -> 100 | DSJC500.5 -> 48 | DSJR500.5 -> 122 | r1000.1.c -> 98 | flat300_28_0 -> 28

main(){
    user=$(whoami)
    echo "Script automatizador de testes - Rodando como $user"
    run="-C 5 -v"
    instances=(le450_15d latin_square DSJC500.5 DSJR500.5 r1000.1.c flat300_28_0)
    colors=(15 100 48 122 98 28) #Match instances
    #${array_name[index]}
    ants=(16 64 128 256)
    versions=(1)
    gpu_vers=(0 1) #1 de fora
    declare -i kc=-1
    for i in ${instances[*]}
    do
        kc=$(($kc+1))
        for j in ${ants[*]}
        do
            for k in ${versions[*]}
            do
                for l in ${gpu_vers[*]}
                do
                    data=$(date +"%T, %d/%m/%y, %A")
                    echo -e "Código executado via script-2 automatizador de testes.\n$data\nparams: $run -c $k -z $l -n $j -k ${colors[kc]}\n" >./testes/"$i"/"$j"/"$i"_"$l"_"$k".txt
                    ./gpu_colorant $run -c "$k" -z $l -n "$j" -k "${colors[kc]}" ./instances/"$i".col >>./testes/"$i"/"$j"/"$i"_"$l"_"$k".txt
                    #mv gmon.out gmon_"$i"_"$l"_"$k".out
                    #mv gmon_"$i"_"$l"_"$k".out ./testes/"$i"/"$j"/
                    echo -e "$i $l $k $j Feito!\n"
                done
            done
        done
    done

    cd ./testes/
    ./read.sh
}

main $*
