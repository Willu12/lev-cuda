# lev-cuda
cuda levenstein distance

# usage
możliwe wywołania programu

./cuda_lev {path1, path2}
./cuda_lev -gc {path1, path2}
./cuda_lev -c {path1, path2}
./cuda_lev -g {path1_path2} 

brak podania ścieżki sprawi, że zostanie wczytany domyślny plik
flagi:
-g - wywołanie programu na gpu
-c - wywołanie programu na cpu
-gc - wywołanie programu na gpu i cpu
brak flag - wywołanie tylko na gpu