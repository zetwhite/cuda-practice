main : mat_add.o
	nvcc $< -o $@ -lcudart -L/user/local/cuda/lib64

mat_add.o : mat_add.cu
	nvcc -c -o $@ $< -O3 -I/user/local/cuda/include

clean:
	rm -rf *.o main
