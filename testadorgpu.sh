#!/bin/bash
#Author Bruno Cesar Puli Dala Rosa, bcesar.g6@gmail.com
#4/07/2017 16:00
#Script automatizador de testes do Novo colorant

#dsjc250.5;    k = 28
#dsjr500.5;    k = 122
#le450_15d;    k = 15
#flat300_28_0; k = 28

main(){
    user=$(whoami)
    echo "Script automatizador de testes - Rodando como $user"
    params="-E 1800 -p 1 -n 10 -a 2 -b 8 -r 0.9 -t 100000 -g 10 -m 25 -M -d 0.9 -N 100 -v -z"
    instances=(dsjc250.5 dsjr500.5 le450_15d flat300_28_0)
    colors=(28 122 15 28) #Match instances
    testes=(1 2 3)
    ants=(64 128 256 512 1024 2048)

    mkdir testes_gpu
    declare -i kc=-1
    for i in ${instances[*]}
    do
        echo -e "Começando testes de $i\n"
        mkdir testes_gpu/"$i"
        kc=$(($kc+1))

        for a in ${ants[*]}
        do
            for t in ${testes[*]}
            do
                data=$(date +"%T, %d/%m/%y, %A")
                echo -e "Código executado via script automatizador de testes.\n$data\nparams: $params -k ${colors[kc]}\n" >./testes_gpu/"$i"/gpu_"$i"_"$a"_"$t".txt
                ./gpu_colorant $params -A $a -k "${colors[kc]}" ./instances/"$i".col >>./testes_gpu/"$i"/gpu_"$i"_"$a"_"$t".txt

                echo -e " GPU $i _ $a - $t Feito!\n"
            done
        done

    done

    zip -r testes_gpu.zip testes_gpu

    data=$(date +"%T, %d/%m/%y, %A")
    echo -e "Terminado em $data"
}

main $*
