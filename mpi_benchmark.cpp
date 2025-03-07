
#include <iostream>
#include <vector>
#include <mpi.h>


#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>

struct SimpleData{
	int a; 
	double b;
	double values[100] ; 
	double values2[10000]; 


	template <class Archive>
		void serialize(Archive & ar, const unsigned int){
			ar & a ; 
			ar & b ; 
			ar & values ; 
			ar & values2 ; 
		}

};

struct VectorData {
	std::vector<int> vec ; 
};


struct ArrayData{
	int id ; 
	double values[1000] ; 
};


void benchmark_raw_mpi_simple(int rank, int size, int num_iterations){
	SimpleData data; 
	if (rank == 0){
		double start = MPI_Wtime() ; 
		for (int i = 0 ; i < num_iterations; i++){
			MPI_Send(&data.a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
			MPI_Send(&data.b, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(data.values, 100, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                        MPI_Send(data.values2, 10000, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
										      
		}
		int ack ; 
	       MPI_Recv(&ack, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	       double end = MPI_Wtime();
	       std::cout << "Raw MPI SimpleData: " << (end - start) / num_iterations << " s/op\n";		
	}	
	else if (rank == 1){
		for (int i = 0 ; i < num_iterations; i++){
                        MPI_Recv(&data.a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&data.b, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
			MPI_Recv(data.values, 100, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(data.values2, 10000, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
		}
	        int ack = 1;
	        MPI_Send(&ack, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	}
}




void benchmark_boost_mpi_simple(int num_iterations) {
    boost::mpi::communicator world;  
    int rank = world.rank();
    SimpleData data;


    if (rank == 0) {
        double start = MPI_Wtime(); 
        for (int i = 0; i < num_iterations; ++i) {
            world.send(1, 0, data); 
        }
        int ack;
        world.recv(1, 1, ack);  
        double end = MPI_Wtime();
        std::cout << "Boost MPI SimpleData: " << (end - start) / num_iterations << " s/op\n";
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            world.recv(0, 0, data);
        }
        int ack = 1;
        world.send(0, 1, ack); 
    }
}


int main(int argc, char** argv)
{

	MPI_Init(&argc, &argv) ; 
	int rank = -1 ; 
	int size = 0 ; 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int num_iterations = 1000 ; 
	benchmark_raw_mpi_simple(rank, size, num_iterations);

        benchmark_boost_mpi_simple(num_iterations);


	MPI_Finalize() ; 

	return 0 ; 
}



