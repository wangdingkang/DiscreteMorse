#!/bin/bash
#$ -l m_mem_free=16G


matlab -r 'load_persistence_diagram("dipha/inputs/new_stp_2/dipha.edges", "dipha/inputs/new_stp_2/dipha_edges.txt")'

