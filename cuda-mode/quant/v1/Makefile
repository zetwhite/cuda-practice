NVCC        = /usr/local/cuda-12.1/bin/nvcc 
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = quant_dequant
OBJ	        = quant_dequant.o

default: $(EXE)

quant_dequant.o: quant_dequant.cu
	$(NVCC) -c -o $@ quant_dequant.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

