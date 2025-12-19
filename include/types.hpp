#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <boost/serialization/vector.hpp>

#define DEFAULT_OUTER_SIZE 10
#define DEFAULT_INNER_SIZE 1
#define NUM_ITERATIONS 10000

struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    // Constructeur avec tailles variables pour l'émetteur
    VectorOfVectors() {
        data.resize(DEFAULT_OUTER_SIZE);
        for (int i = 0; i < DEFAULT_OUTER_SIZE; i++) {
            data[i].resize(DEFAULT_INNER_SIZE + i * i * i * i, 0);
        }
    }

    // Constructeur par défaut pour le récepteur
    VectorOfVectors(int) : data() {}

    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }
};

#endif // TYPES_HPP
