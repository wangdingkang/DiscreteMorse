#!/bin/bash
#$ -l m_mem_free=2G

# python3 dipha/python/write_3d_test_file.py dipha/inputs/size_less_print.complex 2000 2400 1200

python dipha/python/write_dipha_file_3d.py new_stp_2/ dipha/inputs/new_stp_2/complex.bin dipha/inputs/new_stp_2/vert.txt 

