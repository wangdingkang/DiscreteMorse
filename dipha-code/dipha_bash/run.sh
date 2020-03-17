#!/bin/bash
#$ -l m_mem_free=1024G

mpiexec -n 32 dipha/dipha-graph-recon/build/dipha dipha/inputs/new_stp_2/complex.bin dipha/inputs/new_stp_2/diagram.bin dipha/inputs/new_stp_2/dipha.edges 1137 855 270

