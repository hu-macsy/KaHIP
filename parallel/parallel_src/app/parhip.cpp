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

#include "parallel_label_compress/parallel_label_compress.h"

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
                if( in_G.number_of_global_nodes()==0 ){
                    MPI_Finalize();
                    throw std::runtime_erro("Input graph has no nodes! corrupted file or error while reading file "+graph_filename);
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

                    // copy or reduce graph. This is meant to reduce the memory consumption for certain complex graphs
                    t.restart();
                    const NodeID global_max_degree = in_G.get_global_max_degree(communicator);
                    const double avg_degree = in_G.number_of_global_edges()/ (double) in_G.number_of_global_nodes();
                    if( rank == ROOT ) std::cout<<"log> max degree " << global_max_degree << " average degree " << avg_degree << std::endl;
                    //std::vector<NodeID> global_hdn = in_G.get_high_degree_global_nodes_by_degree( avg_degree*1.5 , false);


                    const NodeID numLocalNodes = partition_config.hdn_percent*in_G.number_of_global_nodes()/size;
                    std::vector<NodeID> global_hdn = in_G.get_high_degree_global_nodes_by_num( numLocalNodes, partition_config.use_ghost_degree );

                    parallel_graph_access G(communicator);

		if(global_hdn.empty()) {
			
		        in_G.copy_graph(G, communicator);
			if (rank == ROOT) {
				std::cout << "WARNING : Empty list of high degree nodes! " << std::endl;
				std::cout << " ======================================== " << std::endl;
				std::cout << " ===========  Copying graph   =========== " << std::endl;
				std::cout << " ======================================== " << std::endl;
				std::cout << "log> global graph number of nodes "
					  << G.number_of_global_nodes() << " number of edges "
					  << G.number_of_global_edges() << std::endl;
				std::cout << "log> ghost nodes, original graph " << in_G.number_of_ghost_nodes() << " copied g " << G.number_of_ghost_nodes() << std::endl;
		    
			}
		} else {
			// reducing graph
			if( partition_config.aggressive_removal ) {
				if (rank == ROOT)
					std::cout << "log>  Enable aggressive removal of edges. " << std::endl;
				in_G.reduce_graph(G, global_hdn, communicator, partition_config.aggressive_removal);
			}
			else {
				in_G.reduce_graph(G, global_hdn, communicator);
			}
			if (rank==ROOT){
				std::cout << " ========================================= " << std::endl;
				std::cout << " ============  Reducing graph  =========== " <<  std::endl;
				std::cout << " ========================================= " << std::endl;
				std::cout << "log> number of affected nodes " << global_hdn.size() << ", on each PE " << numLocalNodes << std::endl;
				std::cout << "log> reduced graph number of nodes "
					  << G.number_of_global_nodes() << " number of edges "
					  << G.number_of_global_edges() << std::endl;
				std::cout << "log> ghost nodes, original graph " << in_G.number_of_ghost_nodes() << " reduced g " << G.number_of_ghost_nodes() << std::endl;
                        }
		}
		assert( G.number_of_local_nodes() == in_G.number_of_local_nodes() );    //number of nodes should be the same
		assert( G.number_of_local_edges() <= in_G.number_of_local_edges() );    //edges are less or equal

		
		double reducing_graph_time = t.elapsed(); // including finding high degree nodes // barrier in get_reduced, get_copy

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
                [[maybe_unused]] auto [red_globalInterEdges, red_globalIntraEdges, red_globalWeight ] = G.get_ghostEdges_nodeWeight();
                auto [globalInterEdges, globalIntraEdges, globalWeight ] = in_G.get_ghostEdges_nodeWeight();

                if( rank == ROOT ) {
                        std::cout <<  "log> ghost edges/m " <<  globalInterEdges/(double)in_G.number_of_global_edges() << std::endl;
                        std::cout <<  "log> local edges/m " <<  globalIntraEdges/(double)in_G.number_of_global_edges() << std::endl;
                        std::cout <<  "log> reduced G ghost edges/m " <<  red_globalInterEdges/(double)G.number_of_global_edges() << std::endl;
                        std::cout <<  "log> reduced G local edges/m " <<  red_globalIntraEdges/(double)G.number_of_global_edges() << std::endl;
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

        //partition_config.label_iterations = partition_config.label_iterations_refinement;
		partition_config.label_iterations = 3; //temporary,hardwire to 2

	        EdgeWeight inter_ref_edge_cut = 0;
	        double inter_ref_balance = 0;
		
		if( partition_config.label_iterations != 0 ) {
			partition_config.total_num_labels = partition_config.k;
			partition_config.upper_bound_cluster = partition_config.upper_bound_partition;
			const EdgeWeight edge_cut = qm.edge_cut( G, communicator );
			const double balance = qm.balance( partition_config, G, communicator );

			if ( rank == ROOT ) {
				std::cout << " log> config.label_iterations = " << partition_config.label_iterations << std::endl; 
				std::cout << " log> config.total_num_labels = " << partition_config.total_num_labels << std::endl;
				std::cout << " log> config.k = " <<  partition_config.k << std::endl;
				std::cout << " log> current cut = " << edge_cut << " current balance " << balance << std::endl;
			}
			assert( G.number_of_local_nodes() == in_G.number_of_local_nodes() );    //number of nodes should be the same
			assert( G.number_of_local_edges() <= in_G.number_of_local_edges() );    //edges are less or equal
			assert( G.number_of_global_nodes() == in_G.number_of_global_nodes() );    //number of nodes should be the same
			if (rank == ROOT)
				std::cout << "G.number_of_ghost_nodes() = " << G.number_of_ghost_nodes() <<
					" in_G.number_of_ghost_nodes() = " << in_G.number_of_ghost_nodes() << std::endl;
			//assert( G.number_of_ghost_nodes() <= in_G.number_of_ghost_nodes() );    //edges are less or equal

                t.restart();

                forall_local_nodes(G, node) {
                    in_G.setNodeLabel(node, G.getNodeLabel(node));
                    in_G.setSecondPartitionIndex(node, G.getSecondPartitionIndex(node));
                } endfor
                // init_balance_management should be called after setting
                // the labels of local nodes equal to the block labels
                // (as if already partitioned in k parts ).
                partition_config.total_num_labels = partition_config.k; //forces refinement balance
                in_G.init_balance_management( partition_config );

                //update the ghost nodes of in_G
                in_G.update_ghost_node_data();
                in_G.update_ghost_node_data_global();
                in_G.update_ghost_node_data_finish();

                inter_ref_edge_cut = qm.edge_cut( in_G, communicator );
                inter_ref_balance = qm.balance( partition_config, in_G, communicator );

			PPartitionConfig working_config = partition_config;
			working_config.vcycle = false; // assure that we actually can improve the cut
			parallel_label_compress< std::vector< NodeWeight> > plc_refinement;
			if (!global_hdn.empty()) {
			  if ( rank == ROOT )
			    std::cout << "log> LAST REFINEMENT STEP ON FINEST GRAPH " << std::endl; 
			  plc_refinement.perform_parallel_label_compression( working_config, in_G, false, false, PEtree); // balance, for_coarsening
			}
		 }
		
		double final_refine_time = t.elapsed(); 


                //qm.evaluateMapping(in_G, PEtree, communicator);

                EdgeWeight edge_cut = qm.edge_cut( in_G, communicator );
                EdgeWeight qap = 0;
                //if tree is empty, qap is not to be calculated
                if (partition_config.integrated_mapping)
                        qap = qm.total_qap( in_G, PEtree, communicator );
                double balance  = qm.balance( partition_config, in_G, communicator );
                PRINT(double balance_load  = qm.balance_load( partition_config, in_G, communicator );)
                PRINT(double balance_load_dist  = qm.balance_load_dist( partition_config, in_G, communicator );)

                distributed_quality_metrics qm_no_final_ref;
                EdgeWeight edge_cut_no_final_ref = qm_no_final_ref.edge_cut( G, communicator );
                EdgeWeight qap_no_final_ref = 0;
                if (partition_config.integrated_mapping)
                    qap_no_final_ref = qm_no_final_ref.total_qap( G, PEtree, communicator );

                double balance_no_final_ref  = qm_no_final_ref.balance( partition_config, G, communicator );
                if (!global_hdn.empty()) {
                    if( rank == ROOT ) std::cout<< __LINE__ << ", " << edge_cut_no_final_ref << " < " << edge_cut << std::endl; // in_G has more edges, thus a higher cut
                    if( rank == ROOT ) std::cout<< __LINE__ << ", " <<  "inter_ref_balance = " << inter_ref_balance << std::endl;
                    if( rank == ROOT ) std::cout<< __LINE__ << ", " <<  balance << " = " << balance_no_final_ref << std::endl; // currently true because balance = false in plc

                    if( rank == ROOT ) {
                        if (inter_ref_edge_cut  <= edge_cut ){
                            std::cout<< __LINE__ << ", WARNING: last refinement step did not improve edgecut: (" << inter_ref_edge_cut << " <= " << edge_cut  << ")" << std::endl;
                        }else{
                            std::cout<< __LINE__ << ", last refinement step returned edgecut " << edge_cut << ", cut before " << inter_ref_edge_cut  << std::endl;
                        }
                    }
                }else{
                    if( rank == ROOT ){ std::cout<< __LINE__ << ", " << edge_cut_no_final_ref << " = " << edge_cut << std::endl; 
                        std::cout<< __LINE__ << ", " <<  balance << " = " << balance_no_final_ref << std::endl; 
                        std::cout<< __LINE__ << ", " <<  qap << " = " << qap_no_final_ref << std::endl; 
                    }
                    assert(edge_cut_no_final_ref == edge_cut);
                    assert(balance_no_final_ref == balance);
                    assert(qap_no_final_ref == qap);
                }

                if( rank == ROOT ) {
                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout << "log>" << "============AND WE R DONE============" << std::endl;
                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout << "log> METRICS" << std::endl;
                        std::cout << "log> total reducing graph time elapsed " << reducing_graph_time  << std::endl;
                        std::cout << "log> total partitioning time (c+ip+r) elapsed " <<  running_time << std::endl;
                        std::cout << "log> total coarse time " <<  qm.get_coarse_time() << std::endl;
                        std::cout << "log> total inpart time " <<  qm.get_inpart_time() << std::endl;
                        std::cout << "log> total refine time " <<  qm.get_refine_time() << std::endl;
                        std::cout << "log> total final refine time elapsed " << final_refine_time  << std::endl;
                        std::cout << "log> total solution time (c+ip+r+r) elapsed " << final_refine_time + running_time  << std::endl;
                        std::cout << "log> initial numNodes " <<  qm.get_initial_numNodes() << std::endl;
                        std::cout << "log> initial numEdges " <<  qm.get_initial_numEdges() << std::endl;
                        std::cout << "log> initial edge cut  " <<  qm.get_initial_cut()  << std::endl; // comparing to adding network information
                        std::cout << "log> final edge cut " <<  edge_cut  << std::endl;
                        std::cout << "log> initial qap  " <<  qm.get_initial_qap()  << std::endl; // comparing to adding network information
                        std::cout << "log> final qap  " <<  qap  << std::endl;
                        std::cout << "log> final balance "  <<  balance   << std::endl;
                        std::cout << "log> max congestion " <<  qm.get_max_congestion() << std::endl;
                        std::cout << "log> max dilation " <<  qm.get_max_dilation() << std::endl;
                        std::cout << "log> sum dilation " <<  qm.get_sum_dilation() << std::endl;
                        std::cout << "log> avg dilation  " << qm.get_avg_dilation()  << std::endl;
                        std::cout << "log> final balance load "  <<  balance_load   << std::endl;
                        std::cout << "log> final balance load dist "  <<  balance_load_dist   << std::endl;
                }
                PRINT(qm.comm_vol( partition_config, in_G, communicator );)
                PRINT(qm.comm_bnd( partition_config, in_G, communicator );)
                PRINT(qm.comm_vol_dist( in_G, communicator );)
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
                        pvio.writePartitionSimpleParallel(in_G, filename);
                        pvio.writePartitionSimpleParallel(G, "G_partition.txtp");
                }

                if( partition_config.save_partition_binary ) {
                        if( partition_config.filename_output.empty() ){
                                filename += ".binp";
                        }
                        parallel_vector_io pvio;
                        pvio.writePartitionBinaryParallelPosix(partition_config, in_G, filename);
                }

                if( rank == ROOT && (partition_config.save_partition || partition_config.save_partition_binary) ) {
                        std::cout << "wrote partition to " << filename << " ... " << std::endl;
                        std::cout << "wrote partition to G_partition.txtp ... " << std::endl;
                }

        }//if( communicator != MPI_COMM_NULL) 

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
}
