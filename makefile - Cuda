CC=g++ -m64
LD=nvcc
CFLAGS= -c -g -I . -DLINUX -fpermissive
LDFLAGS=-lm -Wno-deprecated-declarations

all: nnml_cuda

nnml_cuda: nnml_cuda.o common.o iniparser.o dictionary.o 
	${LD} nnml_cuda.o iniparser.o dictionary.o common.o -o nnml_cuda ${LDFLAGS}

nnml_cuda.o: nnml_cuda.cu
	$(LD) -c nnml_cuda.cu $(LDFLAGS)

.c.o:; $(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o nnml_cuda