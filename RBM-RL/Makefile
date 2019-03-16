OBJS = main.o problems.o rbm.o
EXEC = Indenteur
CC = g++ -std=c++11

S = source

BITS = $(getconf LONG_BIT)
ifeq ($(BITS),64)
	FBITS=-m64
else
	FBITS=
endif

#use this line to compile with openmp
#FLAGS = $(FBITS) -O3 -fopenmp
#use this line to compile without openmp
FLAGS = $(FBITS) -O3 


all : RBM

RBM : $(OBJS)
	$(CC) $(FLAGS) -o bin/RBM $(OBJS) -larmadillo -framework Accelerate

main.o : $(S)/main.cpp $(S)/rbm.h $(S)/problems.h
	$(CC) -c $(FLAGS) $(S)/main.cpp 

problems.o : $(S)/problems.cpp $(S)/problems.h $(S)/rbm.h
	$(CC) -c $(FLAGS) $(S)/problems.cpp 

rbm.o : $(S)/rbm.cpp $(S)/rbm.h 
	$(CC) -c $(FLAGS) $(S)/rbm.cpp 

#per cancellare i file oggetto fai make clean
clean :
	@rm *.o

rmexec :
	@rm $(EXEC)
