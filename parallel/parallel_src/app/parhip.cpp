/******************************************************************************
 * parhip.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Christian Schulz <christian.schulz.phone@gmail.com>
 *****************************************************************************/

#include <argtable3.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <mpi.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h> 

#include "communication/mpi_tools.h"
#include "communication/dummy_operations.h"
#include "data_structure/parallel_graph_access.h"
#include "data_structure/processor_tree.h"
#include "distributed_partitioning/distributed_partitioner.h"
#include "io/parallel_graph_io.h"
#include "io/parallel_vector_io.h"
#include "macros_assertions.h"
#include "parse_parameters.h"
#include "ppartition_config.h"
#include "random_functions.h"
#include "timer.h"
#include "tools/distributed_quality_metrics.h"

#include "system_info.h"


int main(int argn, char **argv) {

        MPI_Init(&argn, &argv);    /* starts MPI */

        PPartitionConfig partition_config;
        std::string graph_filename;

        int ret_code = parse_parameters(argn, argv, 
                        partition_config, 
                        graph_filename); 

        if(ret_code) {
                MPI_Finalize();
                return 0;
        }

        int rank, size;
        MPI_Comm communicator = MPI_COMM_WORLD; 
        MPI_Comm_rank( communicator, &rank);
        MPI_Comm_size( communicator, &size);


[[maybe_unused]] double myMem;
if( rank == ROOT ) std::cout<< __LINE__ << ", before reading graph, starting mem usage" << std::endl;
getFreeRam(MPI_COMM_WORLD, myMem, true);


        timer t;
        MPI_Barrier(MPI_COMM_WORLD);
        {
                t.restart();
                if( rank == ROOT ){
                        //the running time of this is calculated in the dummy operation but I think it is negligible
                        std::string callingCommand = "";
                        for (int i = 0; i < argn; i++) {
                            callingCommand += std::string(argv[i]) + " ";
                        }
                        std::cout << "Calling command: " << callingCommand << std::endl;
                        std::cout << "running collective dummy operations ";
                    }
                dummy_operations dop;
                dop.run_collective_dummy_operations();
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if( rank == ROOT ) {
                std::cout <<  "took " <<  t.elapsed()  << std::endl;
        }

        if( communicator != MPI_COMM_NULL) {
                MPI_Comm_rank( communicator, &rank);
                MPI_Comm_size( communicator, &size);

                if(rank != 0) partition_config.seed = partition_config.seed*size+rank; 

                srand(partition_config.seed);

                parallel_graph_access in_G(communicator);
                parallel_graph_io::readGraphWeighted(partition_config, in_G, graph_filename, rank, size, communicator);
                //parallel_graph_io::readGraphWeightedFlexible(G, graph_filename, rank, size, communicator);
                if( rank == ROOT ){
                        std::cout <<  "took " <<  t.elapsed()  << std::endl;
                        std::cout <<  "n: " <<  in_G.number_of_global_nodes() << " m: " <<  in_G.number_of_global_edges()  << std::endl;
                }
		if( rank == ROOT ) std::cout<< __LINE__ << ", read graph " << std::endl;
		getFreeRam(MPI_COMM_WORLD, myMem, true);

                //
                // mapping activity : read processor tree if given 
                //
                processor_tree PEtree( partition_config.distances, partition_config.group_sizes );
                if( rank == ROOT ) {
                        PEtree.print();
                        if( PEtree.get_numPUs()<11){
                                PEtree.print_allPairDistances();
                        }
                }

	        NodeID global_max_degree = 0;
		NodeID local_max_degree  = in_G.get_local_max_degree();
		
		MPI_Allreduce(&local_max_degree, &global_max_degree, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
		std::cout << " R  = " << rank << " local max " << local_max_degree
			  << " global max " << global_max_degree << std::endl;


		/******************* ignore *************************/
		int c = 0;
		NodeID local_node_bound = (NodeID) ceil((in_G.number_of_global_nodes()*0.1)/size);
		// local node_list for each PE
		std::vector<NodeID> node_list;
		//		NodeID degree_bound = (NodeID) (global_max_degree*0.9);
		// temporarily -- for testing
		 NodeID degree_bound = (NodeID) (local_max_degree*0.9);
		// forall_local_nodes(in_G,n) {
		// 	if (c > local_node_bound) break;
		// 	if (in_G.getNodeDegree(n) > degree_bound) {
		// 		node_list.push_back(n);
		// 		c++;
		// 	}
		// } endfor
		    
		// std::cout << " local_node_bound  = " << local_node_bound
		// 	  << " max_degree  = "  << in_G.get_local_max_degree()
		// 	  << " degree_bound  = "  << degree_bound
		// 	  << " c = " << c << std::endl;

		// std::cout << " Rank  = " << rank
		// 	  << " node list [ "  << std::endl;
		// for (auto i = node_list.begin(); i != node_list.end(); ++i)
		// 	std::cout << *i << ' ';
		// std::cout  <<" ]"<< std::endl;


		std::vector<std::vector<NodeID>> edge_list; 
		//in_G.get_removed_edges(node_list,edge_list);			
		/******************* ignore *************************/


		parallel_graph_access G(communicator);
	        parallel_graph_access::get_graph_copy(in_G, G, communicator);
		//parallel_graph_access::get_reduced_graph(in_G, G, node_list, communicator);
		if (rank==ROOT)
			std::cout << " ============     Copying graph  =========== " <<  std::endl;

		

                if( partition_config.refinement_focus ){
                        //in this version, the coarsening factor depends on the input size. As cluster_coarsening_factor sets a limit to the size
                        //of the clusters when coarsening, it should be more than 2, thus, coarsening_factor should be greater than 2
                        const double coarsening_factor = partition_config.coarsening_factor; 
                        partition_config.cluster_coarsening_factor = G.number_of_global_nodes() / (coarsening_factor*partition_config.k);
                        const int coarsening_levels = partition_config.max_coarsening_levels;
                        if( !partition_config.stop_factor ){
                               partition_config.stop_factor = G.number_of_global_nodes()/(coarsening_factor*(coarsening_levels-1));
                               //set a minimum for the global size of the coarsest graph as this greatly affects the time for initial partitioning
                               //this way, each PE will have 2000 vertices
                               //TODO: maybe differentiate between meshes and complex networks?
                               partition_config.stop_factor = std::min( partition_config.stop_factor, size*3000 );
                        }
                }
                
                if(rank == ROOT) {
                        PRINT(std::cout <<  "log> cluster coarsening factor is set to " <<  partition_config.cluster_coarsening_factor  << std::endl;)
                }
                
                //TODO: not sure about this but I think it makes more sense not to divide it. If not divided, coarsening will stop when the global 
                //number of vertices of the coarsest graph is less than stop_factor*k. If we divide by k, then stop_factor is the limit for the global size
                //of the coarsest graph
                if( rank == ROOT ) {
                        std::cout << "log> coarsening will stop if coarsest graph has less than " << partition_config.stop_factor << " vertices" << std::endl;
                }
                partition_config.stop_factor /= partition_config.k;
                
                MPI_Barrier(MPI_COMM_WORLD); //for better printing

                //TODO: what to do when distance and hierarchy is not given
                //update: flag partition_config.integrated_mapping is set to true; use this flag later in refinement
                
                random_functions::setSeed(partition_config.seed);
                parallel_graph_access::set_comm_rounds( partition_config.comm_rounds/size );
                parallel_graph_access::set_comm_rounds_up( partition_config.comm_rounds/size);
                distributed_partitioner::generate_random_choices( partition_config );

                //G.printMemoryUsage(std::cout);
if( rank == ROOT ) std::cout<< __LINE__ << ", allocated data structs" << std::endl;
getFreeRam(MPI_COMM_WORLD, myMem, true);
                //compute some stats
                EdgeWeight interPEedges = 0;
                EdgeWeight localEdges = 0;
                NodeWeight localWeight = 0;
                forall_local_nodes(G, node) {
                        localWeight += G.getNodeWeight(node);
                        forall_out_edges(G, e, node) {
                                NodeID target = G.getEdgeTarget(e);
                                if(!G.is_local_node(target)) {
                                        interPEedges++;
                                } else {
                                        localEdges++;
                                }
                        } endfor
                } endfor

                EdgeWeight globalInterEdges = 0;
                EdgeWeight globalIntraEdges = 0;
                EdgeWeight globalWeight = 0;
                MPI_Reduce(&interPEedges, &globalInterEdges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT, communicator);
                MPI_Reduce(&localEdges, &globalIntraEdges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT, communicator);
                MPI_Allreduce(&localWeight, &globalWeight, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

                if( rank == ROOT ) {
                        std::cout <<  "log> ghost edges/m " <<  globalInterEdges/(double)G.number_of_global_edges() << std::endl;
                        std::cout <<  "log> local edges/m " <<  globalIntraEdges/(double)G.number_of_global_edges() << std::endl;
                }

                t.restart();
                double epsilon = (partition_config.inbalance)/100.0;
                if( partition_config.vertex_degree_weights ) {
                        NodeWeight total_load = G.number_of_global_edges()+G.number_of_global_edges();
                        partition_config.number_of_overall_nodes = G.number_of_global_nodes();
                        partition_config.upper_bound_partition   = (1+epsilon)*ceil(total_load/(double)partition_config.k);

                        forall_local_nodes(G, node) {
                                G.setNodeWeight(node, G.getNodeDegree(node)+1);
                        } endfor

                } else {
                        partition_config.number_of_overall_nodes = G.number_of_global_nodes();
                        partition_config.upper_bound_partition   = (1+epsilon)*ceil(globalWeight/(double)partition_config.k);
                        if( rank == ROOT) {
                                std::cout <<  "upper bound on blocks " << partition_config.upper_bound_partition  << std::endl;
                        }
                }

                distributed_partitioner dpart;
                distributed_quality_metrics qm;

                try{
                        //qm = dpart.perform_partitioning( communicator, partition_config, G, PEtree);
                        dpart.perform_partitioning( communicator, partition_config, G, qm, PEtree);
                } 
                catch (std::bad_alloc & exception) 
                { 
                        std::cerr << " !!! bad_alloc detected: " << exception.what(); 
                } 

                MPI_Barrier(communicator);
                double running_time = t.elapsed();

if( rank == ROOT ) std::cout<< __LINE__ << ", finished partitioning " << std::endl;
getFreeRam(MPI_COMM_WORLD, myMem, true);

                //qm.evaluateMapping(G, PEtree, communicator);

                EdgeWeight edge_cut = qm.edge_cut( G, communicator );
                EdgeWeight qap = 0;
                //if tree is empty, qap is not to be calculated
                if (partition_config.integrated_mapping)
                        qap = qm.total_qap( G, PEtree, communicator );
                double balance  = qm.balance( partition_config, G, communicator );
                PRINT(double balance_load  = qm.balance_load( partition_config, G, communicator );)
                PRINT(double balance_load_dist  = qm.balance_load_dist( partition_config, G, communicator );)


                if( rank == ROOT ) {

                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout << "log>" << "============AND WE R DONE============" << std::endl;
                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout << "log> METRICS" << std::endl;
                        std::cout << "log> total partitioning time elapsed " <<  running_time << std::endl;
                        std::cout << "log> total coarse time " <<  qm.get_coarse_time() << std::endl;
                        std::cout << "log> total inpart time " <<  qm.get_inpart_time() << std::endl;
                        std::cout << "log> total refine time " <<  qm.get_refine_time() << std::endl;
                        std::cout << "log> initial numNodes " <<  qm.get_initial_numNodes() << std::endl;
                        std::cout << "log> initial numEdges " <<  qm.get_initial_numEdges() << std::endl;
                        std::cout << "log> initial edge cut  " <<  qm.get_initial_cut()  << std::endl;
			
                        std::cout << "log> final edge cut " <<  edge_cut  << std::endl;
                        std::cout << "log> initial qap  " <<  qm.get_initial_qap()  << std::endl;
                        std::cout << "log> final qap  " <<  qap  << std::endl;
                        std::cout << "log> final balance "  <<  balance   << std::endl;
                        std::cout << "log> max congestion " <<  qm.get_max_congestion() << std::endl;
                        std::cout << "log> max dilation " <<  qm.get_max_dilation() << std::endl;
                        std::cout << "log> sum dilation " <<  qm.get_sum_dilation() << std::endl;
                        std::cout << "log> avg dilation  " << qm.get_avg_dilation()  << std::endl;
			
                        PRINT(std::cout << "log> final balance load "  <<  balance_load   << std::endl;)
                        PRINT(std::cout << "log> final balance load dist "  <<  balance_load_dist   << std::endl;)
                }
                PRINT(qm.comm_vol( partition_config, G, communicator );)
                PRINT(qm.comm_bnd( partition_config, G, communicator );)
                PRINT(qm.comm_vol_dist( G, communicator );)
		  // qm.comm_vol(partition_config, G, communicator);
		  // qm.comm_vol_dist(G, communicator);
		  


#ifndef NDEBUG
                MPI_Status st; int flag; 
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator, &flag, &st);
                while( flag ) {
                        std::cout <<  "attention: still incoming messages! rank " <<  rank <<  " from " <<  st.MPI_SOURCE << std::endl;
                        int message_length;
                        MPI_Get_count(&st, MPI_UNSIGNED_LONG_LONG, &message_length);
                        MPI_Status rst;
                        std::vector<NodeID> message; message.resize(message_length);
                        MPI_Recv( &message[0], message_length, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, st.MPI_TAG, communicator, &rst); 
                        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator, &flag, &st);
                };
#endif

                // write the partition to the disc
                std::string filename = "partition_"+ std::to_string(partition_config.k);
                if( !partition_config.filename_output.empty() ){
                       filename = partition_config.filename_output;
                }

                if( partition_config.save_partition ) {
                        if( partition_config.filename_output.empty() ){
                                filename += ".txtp";
                        }
                        parallel_vector_io pvio;
                        pvio.writePartitionSimpleParallel(G, filename);
                }

                if( partition_config.save_partition_binary ) {
                        if( partition_config.filename_output.empty() ){
                                filename += ".binp";
                        }
                        parallel_vector_io pvio;
                        pvio.writePartitionBinaryParallelPosix(partition_config, G, filename);
                }

                if( rank == ROOT && (partition_config.save_partition || partition_config.save_partition_binary) ) {
                        std::cout << "wrote partition to " << filename << " ... " << std::endl;
                }

        }//if( communicator != MPI_COMM_NULL) 

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
}
