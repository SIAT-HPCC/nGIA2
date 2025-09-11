all:
	nvcc -O3 -Xcompiler -fopenmp main.cu -o ngia2
run:
	srun -p g_v100 -G 1 -n 1 -c 10 ngia2 clustering -p ../geneTest.packed -r result -i 90
