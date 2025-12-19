#include "types.hpp"
#include <boost/mpi.hpp>
#include <boost/mpi/packed_oarchive.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <mpi.h>

void benchmark_boost_mpi_vector(int rank, int size, int num_iterations) {
    boost::mpi::communicator world;
    VectorOfVectors vec;

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            for (int dest = 1; dest < size; dest++) {
                world.send(dest, 0, vec);
            }
        }
        int ack;
        for (int dest = 1; dest < size; dest++) {
            world.recv(dest, 1, ack);
        }
    }
    else {
        VectorOfVectors vec(0);
        for (int i = 0; i < num_iterations; i++) {
            world.recv(0, 0, vec);
        }
        int ack = 1;
        world.send(0, 1, ack);
    }
}

void benchmark_boost_packed_mpi_vector(int rank, int size, int num_iterations) {
    boost::mpi::communicator world;
    VectorOfVectors vec;

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            boost::mpi::packed_oarchive::buffer_type buffer;
            boost::mpi::packed_oarchive oa(world, buffer);
            oa << vec;
            for (int dest = 1; dest < size; dest++) {
                world.send(dest, 0, buffer);
            }
        }
        int ack;
        for (int dest = 1; dest < size; dest++) {
            world.recv(dest, 1, ack);
        }
    }
    else {
        VectorOfVectors vec(0);
        for (int i = 0; i < num_iterations; i++) {
            world.recv(0, 0, vec);
        }
        int ack = 1;
        world.send(0, 1, ack);
    }
}
