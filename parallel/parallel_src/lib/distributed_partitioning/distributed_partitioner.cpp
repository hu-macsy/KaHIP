/******************************************************************************
 * distributed_partitioner.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Christian Schulz <christian.schulz.phone@gmail.com>
 *****************************************************************************/

#include <sstream>
#include "communication/mpi_tools.h"
#include "distributed_partitioner.h"
#include "initial_partitioning/initial_partitioning.h"
#include "io/parallel_graph_io.h"
#include "parallel_contraction_projection/parallel_contraction.h"
#include "parallel_contraction_projection/parallel_block_down_propagation.h"
#include "parallel_contraction_projection/parallel_projection.h"
#include "parallel_label_compress/parallel_label_compress.h"
#include "stop_rule.h"
#include "tools/distributed_quality_metrics.h"
#include "tools/random_functions.h"
#include "data_structure/linear_probing_hashmap.h"

std::vector< NodeID > distributed_partitioner::m_cf = std::vector< NodeID >();
std::vector< NodeID > distributed_partitioner::m_sf = std::vector< NodeID >();
std::vector< NodeID > distributed_partitioner::m_lic = std::vector< NodeID >();

distributed_partitioner::distributed_partitioner() {
       m_total_graph_weight = std::numeric_limits< NodeWeight >::max(); 
       m_cur_rnd_choice = 0;
       m_level = -1;
       m_cycle = 0;
}

distributed_partitioner::~distributed_partitioner() {
}

void distributed_partitioner::generate_random_choices( PPartitionConfig & config ){
        for( int i = 0; i < config.num_tries; i++) {
                for( int j = 0; j < config.num_vcycles; j++) {
                        m_cf.push_back(random_functions::nextDouble( 10, 25 ));
                        m_sf.push_back(random_functions::nextInt( 20, 500 ));
                        m_lic.push_back(random_functions::nextInt( 2, 15 ));
                }
        }
}


void distributed_partitioner::perform_partitioning( PPartitionConfig & partition_config, parallel_graph_access & G) {
        perform_partitioning( MPI_COMM_WORLD, partition_config, G);
}

distributed_quality_metrics distributed_partitioner::perform_partitioning( MPI_Comm communicator, PPartitionConfig & partition_config, parallel_graph_access & G, distributed_quality_metrics& qm, const processor_tree & PEtree) {

        timer t; 
        double elapsed = 0;
        m_cur_rnd_choice = 0;
        PPartitionConfig config = partition_config;
        config.vcycle = false;
        //if the coarsest graph has more vertices, we will coarsen further 
        partition_config.percent_to_force_further_coarsening = 0.01;
        NodeID coarsest_graph_size_max = G.number_of_global_nodes()*partition_config.percent_to_force_further_coarsening;

        PEID rank;
        MPI_Comm_rank( communicator, &rank);

        for( int cycle = 0; cycle < partition_config.num_vcycles; cycle++) {
                t.restart();
                m_cycle = cycle;
                #ifndef NOOUTPUT
                        if( rank == ROOT ) {
                                PRINT( std::cout << "\n\t\tStarting vcycle " << m_cycle+1 <<" out of " << partition_config.num_vcycles <<"\n\n");
                        }
			
                #endif

	        if(cycle+1 == partition_config.num_vcycles && partition_config.no_refinement_in_last_iteration) {
                        config.label_iterations_refinement = 0;
                }
                
if( rank == ROOT ) {
    PRINT(std::cout <<  "log>part: " << cycle << " , coarsest_graph_size_max " << coarsest_graph_size_max << std::endl;)
}

                //the core partitioning routine
                vcycle( communicator, config, G, qm, PEtree, coarsest_graph_size_max, cycle!=0 );

                if( rank == ROOT ) {
                        PRINT(std::cout <<  "log>part: " << m_cycle << " qap " << qm.get_initial_qap()  << std::endl;)
                        PRINT(std::cout <<  "log>cycle: " << m_cycle << " uncoarsening took " << m_t.elapsed()  << std::endl;)
                }
#ifndef NDEBUG
                check_labels(communicator, config, G);
#endif

                elapsed += t.elapsed();

#ifndef NOOUTPUT

                EdgeWeight edge_cut = qm.edge_cut( G, communicator );
                double balance      = qm.balance( config, G, communicator );

                if( rank == ROOT ) {
                        std::cout <<  "log>cycle: " << cycle << " k " <<  config.k << " cut " << edge_cut << " balance " << balance << " time " <<  elapsed  << std::endl;
                }
#endif 
                t.restart();
                m_t.restart();
                if( cycle+1 < config.num_vcycles ) {
                        forall_local_nodes(G, node) {
                                G.setSecondPartitionIndex(node, G.getNodeLabel(node));
                                G.setNodeLabel(node, G.getGlobalID(node));
                        } endfor

                        forall_ghost_nodes(G, node) {
                                G.setSecondPartitionIndex(node, G.getNodeLabel(node));
                                G.setNodeLabel(node, G.getGlobalID(node));
                        } endfor
                }

                config.vcycle = true;

                if( rank == ROOT && config.eco ) {
                        config.cluster_coarsening_factor = m_cf[m_cur_rnd_choice++];
                }

                if(config.eco) {
                        MPI_Bcast(&(config.cluster_coarsening_factor), 1, MPI_DOUBLE, ROOT, communicator);
                        std::cout << "eco configuration, cluster_coarsening_factor " << config.cluster_coarsening_factor  << std::endl;
                }
                config.evolutionary_time_limit = 0;
                elapsed += t.elapsed();
                MPI_Barrier(communicator);
                
        }

        // if( rank == ROOT )
        //          qm.print();
        return qm;
}


void distributed_partitioner::vcycle( 
    MPI_Comm communicator, 
    PPartitionConfig & partition_config, 
    parallel_graph_access & G,
    distributed_quality_metrics &qm, 
    const processor_tree & PEtree, 
    const NodeID coarsest_graph_size_max,
    const bool forceNewVcycle) {
        PPartitionConfig config = partition_config;
        PPartitionConfig config_orig = partition_config;

        mpi_tools mpitools;
        timer t;

        std::vector< double > vec(3, 0.0);

        if( m_total_graph_weight == std::numeric_limits< NodeWeight >::max() ) {
                m_total_graph_weight = G.number_of_global_nodes();
        }

        PEID rank;
        MPI_Comm_rank( communicator, &rank);

#ifndef NOOUTPUT
        if( rank == ROOT ) {
                std::cout << "log>" << "=====================================" << std::endl;
                std::cout << "log>" << "=============NEXT LEVEL==============" << std::endl;
                std::cout << "log>" << "=====================================" << std::endl;
        }
#endif
        t.restart();

        //---------------------------------------------------------------
        //
        // first, recursively contract/coarsen the graph
        //

        m_level++;
        config.label_iterations = config.label_iterations_coarsening;
        config.total_num_labels = G.number_of_global_nodes();
        //
        config.upper_bound_cluster = config.upper_bound_partition/(1.0*config.cluster_coarsening_factor);
        G.init_balance_management( config );

        //parallel_label_compress< std::unordered_map< NodeID, NodeWeight> > plc;
        parallel_label_compress< linear_probing_hashmap  > plc;
        // TODO: decide if we want to pass PEtree as an argument during coarsening.
        // For now we do think there is no need ...
        plc.perform_parallel_label_compression ( config, G, true );

#ifndef NOOUTPUT
        if( rank == ROOT ) {
                std::cout <<  "log> cluster upper bound used is " << config.upper_bound_cluster << ", update step size= " << config.update_step_size 
                << std::endl;
                std::cout <<  "log>cycle: " << m_cycle << " level: " << m_level  << " parallel label compression took " <<  t.elapsed() << std::endl;
        }
#endif

        parallel_graph_access Q(communicator); //the contracted graph
        t.restart();

        {
                parallel_contraction parallel_contract;
                parallel_contract.contract_to_distributed_quotient( communicator, config, G, Q); // contains one Barrier

                parallel_block_down_propagation pbdp;
                if( config.vcycle ) {
                        // in this case we have to propagate the partition index down
                        pbdp.propagate_block_down( communicator, config, G, Q);
                }
        
                MPI_Barrier(communicator);
        }

#ifndef NOOUTPUT

{
NodeID global_ghost_nodes = 0;
NodeID G_local_ghost_nodes = G.number_of_ghost_nodes();
NodeID G_local_nodes = G.number_of_local_nodes();
NodeID G_all_local_data = G_local_nodes+G_local_ghost_nodes;
NodeID max_ghost_nodes = 0;
NodeID max_local_nodes = 0;
NodeID max_all_local_nodes = 0;
MPI_Reduce(&G_local_ghost_nodes, &global_ghost_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT, communicator);
MPI_Reduce(&G_local_ghost_nodes, &max_ghost_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT, communicator);
MPI_Reduce(&G_local_nodes, &max_local_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT, communicator);
MPI_Reduce(&G_all_local_data, &max_all_local_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT, communicator);
        if( rank == ROOT ) {
                std::cout <<  "log>cycle: " << m_cycle << " level: " << m_level << " contraction took " <<  t.elapsed() << std::endl;
                std::cout <<  "log>cycle: " << m_cycle << " level: " << m_level << " coarse nodes n=" << Q.number_of_global_nodes() << ", coarse edges m=" << Q.number_of_global_edges() << " ghost nodes n=" << Q.number_of_ghost_nodes()<< std::endl;
PRINT(std::cout << "global_ghost_nodes "<< global_ghost_nodes << " , max_ghost_nodes " << max_ghost_nodes << ", max_local_nodes " << max_local_nodes << ", max_all_local_nodes "<< max_all_local_nodes << std::endl;)
        }
}
#endif

        if( !contraction_stop_decision.contraction_stop(config, G, Q)) {
                if( config.refinement_focus ){
                        double contraction_factor = Q.number_of_global_nodes() / (double) G.number_of_global_nodes();
                        if( rank == ROOT ) std::cout << "contraction_factor = " << contraction_factor << std::endl;
                        contraction_factor = std::min( 1.0/config.coarsening_factor , contraction_factor);
                        config.cluster_coarsening_factor *= contraction_factor; //multiply to keep the same contraction factor in every level
                }

                //
                // recursively call to coarsen
                //
                vcycle( communicator, config, Q, qm, PEtree, coarsest_graph_size_max, forceNewVcycle );

        } else {
                //----------------------------------------------------
                //
                // coarsening stopped, get the initial partition
                //

                if( rank == ROOT ) vec[0] +=  m_t.elapsed();
#ifndef NOOUTPUT
                if( rank == ROOT ) {
                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout << "log>" << "================ IP =================" << std::endl;
                        std::cout << "log>" << "=====================================" << std::endl;
                        std::cout <<  "log>cycle: " << m_cycle << " total number of levels " <<  (m_level+1) << std::endl;
                        std::cout <<  "log>cycle: " << m_cycle << " number of coarsest nodes " <<  Q.number_of_global_nodes() << std::endl;
                        std::cout <<  "log>cycle: " << m_cycle << " number of coarsest edges " <<  Q.number_of_global_edges() << std::endl;
                        std::cout <<  "log>cycle_m: " << m_cycle << " coarsening took  " <<  vec[0] << std::endl;
                }
#endif

                t.restart();

                NodeID max_coarsest_size = coarsest_graph_size_max; 

                if( forceNewVcycle && Q.number_of_global_nodes()>max_coarsest_size ){
                        //partition if coarsest graph is too large
#ifndef NOOUTPUT
                        if( rank == ROOT ) {
                                std::cout << "log>" << "=====================================" << std::endl;
                                std::cout << "log>" << "coarsest graph has "<< Q.number_of_global_nodes() << " that is more than "<< max_coarsest_size << " nodes, will store the partition" << std::endl;
                                std::cout << "log>" << "\t==\t==\t==\t==\t==\t==" << std::endl;
                        }
#endif

                        further_coarsen( communicator, rank, config_orig, Q, qm, PEtree );

                }else{
                        //perform initial partition as normal
                        qm.set_initial_numNodes( Q.number_of_global_nodes() );
                        qm.set_initial_numEdges( Q.number_of_global_edges() );
			// std::cout << "part> rank " << rank << " num_global_nodes = " << Q.number_of_global_nodes() << "num_global_edges = " << Q.number_of_global_edges()
			// 	  << " num_local_nodes = " << Q.number_of_local_nodes() << "num_local_edges = " << Q.number_of_local_edges() 
			// 	  << std::endl;
                        initial_partitioning_algorithm ip;
                        ip.perform_partitioning( communicator, config, Q );
                }

                if( rank == ROOT ) vec[1] += t.elapsed();

#ifndef NOOUTPUT
                if( rank == ROOT ) {
                        std::cout <<  "log>cycle_m: " << m_cycle << " initial partitioning took " << vec[1]  << std::endl;
                }
                m_t.restart();
#endif
        }

        //----------------------------------------------------------------
        //
        //start refinement/uncoarsening 
        //

#ifndef NOOUTPUT
        if( rank == ROOT ) {
                std::cout << "log>" << "=====================================" << std::endl;
                std::cout << "log>" << "============PREV LEVEL ==============" << std::endl;
                std::cout << "log>" << "=====================================" << std::endl;
        }
#endif

        t.restart();

        parallel_projection parallel_project;
        parallel_project.parallel_project( communicator, G, Q ); // contains a Barrier

#ifndef NOOUTPUT
        EdgeWeight cut = qm.edge_cut(G, communicator);
        if( rank == ROOT ) {
                std::cout <<  "log>cycle: " << m_cycle << " level: " << m_level << " projection took " <<  t.elapsed() << ", cut is " << cut << std::endl;
        }
#endif

        static int counter = 0;
        if (!counter) {
            EdgeWeight cut = qm.edge_cut(G, communicator);
            qm.set_initial_cut(cut);
            EdgeWeight  qap = qm.total_qap( G, PEtree, communicator );
            qm.set_initial_qap( qap );
            #ifndef NOOUTPUT
            if( rank == ROOT ) {
                std::cout <<  "log>cycle_m: SETTING INITIAL METRICS " << std::endl;
                std::cout <<  "log>cycle_m: initial partitioning cut " <<  qm.get_initial_cut()  << std::endl;
                std::cout <<  "log>cycle_m: initial partitioning qap " <<  qm.get_initial_qap() << std::endl;
            }
            #endif
        }
        counter++;

        t.restart();
        config.label_iterations = config.label_iterations_refinement;

        if( config.label_iterations != 0 ) {
                config.total_num_labels = config.k;
                config.upper_bound_cluster = config.upper_bound_partition;

                G.init_balance_management( config );
                PPartitionConfig working_config = config;
                working_config.vcycle = false; // assure that we actually can improve the cut

                parallel_label_compress< std::vector< NodeWeight> > plc_refinement;
                plc_refinement.perform_parallel_label_compression( working_config, G, false, false, PEtree);
        }

        // after refinement
        //EdgeWeight qap = qm.total_qap( G, PEtree, communicator);
        if( rank == ROOT ) vec[2] += t.elapsed();
#ifndef NOOUTPUT

        if( rank == ROOT ) {
                std::cout <<  "log>cycle_m: " << m_cycle <<" level: " << m_level << " label compression refinement took " << vec[2]  << std::endl;
        }
#endif
        m_level--;
        if( rank == ROOT ) qm.add_timing(vec);
}


void distributed_partitioner::further_coarsen( 
    MPI_Comm communicator,
    PEID rank,
    PPartitionConfig config,
    parallel_graph_access & Q,
    distributed_quality_metrics & qm,
    const processor_tree & PEtree ) {

        const NodeID num_local_nodes = Q.number_of_local_nodes();
        const NodeID num_ghost_nodes = Q.number_of_ghost_nodes();
        std::vector<PartitionID> prev_partition( num_local_nodes );
        std::vector<PartitionID> prev_partition_ghost( num_ghost_nodes );
        //most likely, in NOT the first round, the second cut (from the previous cycle) is lower
        const EdgeWeight prev_edge_cut = std::min( qm.edge_cut(Q, communicator), qm.edge_cut_second(Q, communicator) );

        //store previous partition
        forall_local_nodes( Q, i ){
                prev_partition[i] = Q.getNodeLabel(i);
                //Q.setSecondPartitionIndex( i, Q.getNodeLabel(i) ); //TODO: is this needed?
                Q.setNodeLabel( i, Q.getGlobalID(i) ); //TODO: is this needed? check how it behaves without this line
        }endfor
        forall_ghost_nodes( Q, node ) {
                //Q.setSecondPartitionIndex( node, Q.getNodeLabel(node) );
                Q.setNodeLabel( node, Q.getGlobalID(node) );
        } endfor                  

        distributed_quality_metrics new_qm;
        //config.stop_factor = 5000; //TODO: try different parameter combinations
        config.label_iterations_coarsening = 10;
        //config.coarsening_factor = 4;
        //TODO: experiment with setting the forceNewVcycle to true
        //since last parameter is false, the previous is not used
if( rank == ROOT ) std::cout<< __LINE__ << ", inside further_coarsen()" << std::endl;
        vcycle( communicator, config, Q, new_qm, PEtree, 1, false );

        qm.set_initial_numNodes( new_qm.get_initial_numNodes() );
        qm.set_initial_numEdges( new_qm.get_initial_numEdges() );
        const EdgeWeight new_edge_cut = new_qm.edge_cut( Q, communicator );

#ifndef NOOUTPUT
        if( rank == ROOT ) {
                std::cout << "log>" << "=====================================" << std::endl;
                std::cout << "log>" << "old edge cut " << prev_edge_cut << " new " <<  new_edge_cut << std::endl;
                std::cout << "log>" << "\t==\t==\t==\t==\t==\t==" << std::endl;
        }
#endif
        
        if( prev_edge_cut<new_edge_cut ){
                #ifndef NOOUTPUT
                if( rank == ROOT ) std::cout << "log>" << "old cut is lower, will re-apply it to graph" << std::endl;
                #endif
                //re-apply old partition
                forall_local_nodes( Q, i){
                        Q.setNodeLabel( i, prev_partition[i] );
                }endfor
                forall_ghost_nodes( Q, node ){
                        Q.setNodeLabel( node, prev_partition_ghost[node] );
                }endfor
        }
}


void distributed_partitioner::check_labels( MPI_Comm communicator, PPartitionConfig & config, parallel_graph_access & G) {
        PEID m_rank, m_size;
        MPI_Comm_rank( communicator, &m_rank);
        MPI_Comm_size( communicator, &m_size);
        
        std::vector< std::vector< NodeID > > send_buffers; // buffers to send messages
        send_buffers.resize(m_size);
        std::vector<bool> m_PE_packed;
        m_PE_packed.resize(m_size); 
        for( unsigned peID = 0; peID < m_PE_packed.size(); peID++) {
                m_PE_packed[ peID ] = false;
        }

        forall_local_nodes(G, node) {
                forall_out_edges(G, e, node) {
                        NodeID target = G.getEdgeTarget(e);
                        if( !G.is_local_node(target)  ) {
                                PEID peID = G.getTargetPE(target);
                                if( !m_PE_packed[peID] ) { // make sure a node is sent at most once
                                        send_buffers[peID].push_back(G.getGlobalID(node));
                                        send_buffers[peID].push_back(G.getNodeLabel(node));
                                        m_PE_packed[peID] = true;
                                }
                        }
                } endfor
                forall_out_edges(G, e, node) {
                        NodeID target = G.getEdgeTarget(e);
                        if( !G.is_local_node(target)  ) {
                                m_PE_packed[G.getTargetPE(target)] = false;
                        }
                } endfor
        } endfor

        //send all neighbors their packages using Isends
        //a neighbor that does not receive something gets a specific token
        for( PEID peID = 0; peID < (PEID)send_buffers.size(); peID++) {
                if( G.is_adjacent_PE(peID) ) {
                        //now we have to send a message
                        if( send_buffers[peID].size() == 0 ){
                                // length 1 encode no message
                                send_buffers[peID].push_back(0);
                        }

                        MPI_Request rq; int tag = peID+17*m_size;
                        MPI_Isend( &send_buffers[peID][0], 
                                    send_buffers[peID].size(), MPI_UNSIGNED_LONG_LONG, peID, tag, communicator, &rq);
                        
                }
        }

        //receive incomming
        PEID counter = 0;
        while( counter < G.getNumberOfAdjacentPEs()) {
                // wait for incomming message of an adjacent processor
                unsigned int tag = m_rank+17*m_size;
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, tag, communicator, &st);
                
                int message_length;
                MPI_Get_count(&st, MPI_UNSIGNED_LONG_LONG, &message_length);
                
                std::vector<NodeID> message; message.resize(message_length);

                MPI_Status rst;
                MPI_Recv( &message[0], message_length, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, tag, communicator, &rst); 
                
                counter++;

                // now integrate the changes
                if(message_length == 1) continue; // nothing to do

                for( int i = 0; i < message_length-1; i+=2) {
                        NodeID global_id = message[i];
                        NodeID label     = message[i+1];

                        if(G.getNodeLabel(G.getLocalID(global_id)) != label) {
                                std::cout <<  "labels not ok"  << std::endl;
                                exit(0);
                        }
                }
        }

        MPI_Barrier(communicator);
}


void distributed_partitioner::check( MPI_Comm communicator, PPartitionConfig & config, parallel_graph_access & G) {
        PEID m_rank, m_size;
        MPI_Comm_rank( communicator, &m_rank);
        MPI_Comm_size( communicator, &m_size);
        
        std::vector< std::vector< NodeID > > send_buffers; // buffers to send messages
        send_buffers.resize(m_size);
        std::vector<bool> m_PE_packed;
        m_PE_packed.resize(m_size); 
        for( unsigned peID = 0; peID < m_PE_packed.size(); peID++) {
                m_PE_packed[ peID ]           = false;
        }

        forall_local_nodes(G, node) {
                forall_out_edges(G, e, node) {
                        NodeID target = G.getEdgeTarget(e);
                        if( !G.is_local_node(target)  ) {
                                PEID peID = G.getTargetPE(target);
                                if( !m_PE_packed[peID] ) { // make sure a node is sent at most once
                                        send_buffers[peID].push_back(G.getGlobalID(node));
                                        send_buffers[peID].push_back(G.getSecondPartitionIndex(node));
                                        m_PE_packed[peID] = true;
                                }
                        }
                } endfor
                forall_out_edges(G, e, node) {
                        NodeID target = G.getEdgeTarget(e);
                        if( !G.is_local_node(target)  ) {
                                m_PE_packed[G.getTargetPE(target)] = false;
                        }
                } endfor
        } endfor

        //send all neighbors their packages using Isends
        //a neighbor that does not receive something gets a specific token
        for( PEID peID = 0; peID < (PEID)send_buffers.size(); peID++) {
                if( G.is_adjacent_PE(peID) ) {
                        //now we have to send a message
                        if( send_buffers[peID].size() == 0 ){
                                // length 1 encode no message
                                send_buffers[peID].push_back(0);
                        }

                        MPI_Request rq; 
                        MPI_Isend( &send_buffers[peID][0], 
                                    send_buffers[peID].size(), MPI_UNSIGNED_LONG_LONG, peID, peID+17*m_size, communicator, &rq);
                }
        }

        //receive incomming
        PEID counter = 0;
        while( counter < G.getNumberOfAdjacentPEs()) {
                // wait for incomming message of an adjacent processor
                unsigned int tag = m_rank+17*m_size;
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, tag, communicator, &st);

                int message_length;
                MPI_Get_count(&st, MPI_UNSIGNED_LONG_LONG, &message_length);
                
                std::vector<NodeID> message; message.resize(message_length);

                MPI_Status rst;
                MPI_Recv( &message[0], message_length, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, tag, communicator, &rst); 
                
                counter++;

                // now integrate the changes
                if(message_length == 1) continue; // nothing to do

                for( int i = 0; i < message_length-1; i+=2) {
                        NodeID global_id = message[i];
                        NodeID label     = message[i+1];

                        if(G.getSecondPartitionIndex(G.getLocalID(global_id)) != label) {
                                std::cout <<  "second partition index weird"  << std::endl;
                                exit(0);
                        }
                }
        }

        MPI_Barrier(communicator);
}

