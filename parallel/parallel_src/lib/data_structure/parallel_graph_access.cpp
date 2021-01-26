/******************************************************************************
 * parallel_graph_access.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Christian Schulz <christian.schulz.phone@gmail.com>
 *****************************************************************************/

#include "balance_management_coarsening.h"
#include "balance_management_refinement.h"
#include "parallel_graph_access.h"


ULONG parallel_graph_access::m_comm_rounds = 128; 
ULONG parallel_graph_access::m_comm_rounds_up = 128; 

parallel_graph_access::parallel_graph_access( MPI_Comm communicator ) : m_num_local_nodes(0), 
                                                 from(0), 
                                                 to(0),
                                                 m_num_ghost_nodes(0), m_local_max_node_degree(0), m_bm(NULL) {


                m_communicator = communicator;
                MPI_Comm_rank( m_communicator, &rank);
                MPI_Comm_size( m_communicator, &size);
                
                m_gnc = new ghost_node_communication(m_communicator);
                m_gnc->setGraphReference(this);
}

parallel_graph_access& parallel_graph_access::operator=( parallel_graph_access& G ){

        const NodeID num_local_nodes = G.number_of_local_nodes();
        const NodeID num_local_edges = G.number_of_local_edges();
        const NodeID num_global_nodes = G.number_of_global_nodes();
        const NodeID num_global_edges= G.number_of_global_edges();

        start_construction(num_local_nodes, num_local_edges, num_global_nodes, num_global_edges );

        m_num_local_nodes = num_local_nodes;
        set_range( G.get_from_range(), G.get_to_range() );
        set_range_array( G.get_range_array() ); 
        m_num_ghost_nodes = G.number_of_ghost_nodes();
        m_local_max_node_degree = G.get_local_max_degree();
        //m_bm = 
        m_communicator = G.getCommunicator();
        MPI_Comm_rank( m_communicator, &rank);
        MPI_Comm_size( m_communicator, &size);
        
        m_gnc = new ghost_node_communication(m_communicator);
        m_gnc->setGraphReference(this);
        finish_construction();

        return *this;
};

parallel_graph_access::~parallel_graph_access() {
        m_comm_rounds = std::min(m_comm_rounds, m_comm_rounds_up); 
        delete m_gnc;
        if ( m_bm ) delete m_bm;
}

void parallel_graph_access::init_balance_management( PPartitionConfig & config ) {
        if( m_bm != NULL ) {
                delete m_bm;
        }

        if( config.total_num_labels != config.k ) {
                m_bm = new balance_management_coarsening( this, config.total_num_labels );
        } else {
                m_bm = new balance_management_refinement( this, config.total_num_labels );
        }
}

void parallel_graph_access::update_non_contained_block_balance( PartitionID from, PartitionID to, NodeWeight node_weight) {
        m_bm->update_non_contained_block_balance( from, to, node_weight);
}
void parallel_graph_access::update_block_weights() {
        m_bm->update();
}

void parallel_graph_access::update_ghost_node_data( bool check_iteration_counter ) {
        m_gnc->update_ghost_node_data( check_iteration_counter );
}
void parallel_graph_access::update_ghost_node_data_global() {
        m_gnc->update_ghost_node_data_global();
}

void parallel_graph_access::update_ghost_node_data_finish() {
        m_gnc->update_ghost_node_data_finish();
}

void parallel_graph_access::set_comm_rounds(ULONG comm_rounds) {
        m_comm_rounds = comm_rounds;
        set_comm_rounds_up(comm_rounds);
}

void parallel_graph_access::set_comm_rounds_up(ULONG comm_rounds) {
        m_comm_rounds_up = comm_rounds;
}

