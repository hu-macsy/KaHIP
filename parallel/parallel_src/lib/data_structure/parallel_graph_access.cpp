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

//TODO?: provide copy constructor and assignment operator
/*
parallel_graph_access::parallel_graph_access( parallel_graph_access& other ){
    *this=other;
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
*/

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

std::vector<NodeID> parallel_graph_access::get_high_degree_local_nodes(const NodeID minDegree) {
    std::vector<NodeID> local_high_degree_nodes;

    forall_local_nodes((*this), node) {
        const double node_degree = getNodeDegree(node);
        //if this node has a higher degree than we allow
        if (node_degree>minDegree){
            assert( is_local_node(node) );
            local_high_degree_nodes.push_back( getGlobalID(node) );
        } 
    }endfor
    
    return local_high_degree_nodes;
}

std::vector<NodeID> parallel_graph_access::get_high_ghost_degree_local_nodes(const NodeID minDegree) {
    std::vector<NodeID> local_high_degree_nodes;
    std::vector<NodeID> ghostDeg = get_local_ghost_degrees();
    assert( ghostDeg.size() == number_of_local_nodes() );

    forall_local_nodes((*this), node) {
        const double node_degree = ghostDeg[node]; 
        //if this node has a higher degree than we allow
        if (node_degree>minDegree){
            assert( is_local_node(node) );
            local_high_degree_nodes.push_back( getGlobalID(node) );
        } 
    }endfor
    
    return local_high_degree_nodes;
}

std::vector<NodeID> parallel_graph_access::get_high_degree_global_nodes(const NodeID minDegree, const bool useGhostDegree ) {
    
    std::vector<NodeID> local_high_degree_nodes;
    if( useGhostDegree ){
        local_high_degree_nodes = get_high_ghost_degree_local_nodes(minDegree);
    }else{
        local_high_degree_nodes = get_high_degree_local_nodes(minDegree);
    }

    //from the local high degree nodes we need to construct a replicated vector with all nodes
    const int num_local_hdn = local_high_degree_nodes.size();

    //gather the number of nodes in the root
    std::vector<int> hdn_root( size );
    MPI_Gather( &num_local_hdn, 1, MPI_INT, hdn_root.data(), 1, MPI_INT, ROOT, m_communicator );

    //calculate prefix sum, aka displacement
    std::vector<int> displ( size, 0 );              //we do not need the last element
    for( unsigned int i=1; i<size; i++){
        displ[i] = displ[i-1]+ hdn_root[i-1] ;   
    }

    std::vector<NodeID> all_hdn;
    if( rank==ROOT){
        all_hdn.resize( displ.back()+hdn_root.back() );
    }

    MPI_Gatherv( local_high_degree_nodes.data(), num_local_hdn, MPI_UNSIGNED_LONG_LONG, \
        all_hdn.data(), hdn_root.data(), displ.data(), MPI_UNSIGNED_LONG_LONG, ROOT, m_communicator );

    //send all high degree nodes to other PEs; vector must be replicated everywhere
    int all_hdn_size = all_hdn.size();
    MPI_Bcast( &all_hdn_size, 1, MPI_INT, ROOT, m_communicator );
    all_hdn.resize( all_hdn_size );
    MPI_Bcast( all_hdn.data(), all_hdn_size, MPI_UNSIGNED_LONG_LONG, ROOT, m_communicator );

    return all_hdn;
}

std::vector<NodeID> parallel_graph_access::get_local_ghost_degrees(){
    std::vector<NodeID> ghost_degrees( number_of_local_nodes() );
    forall_local_nodes( (*this), node) {
        NodeID gDeg = 0;
        forall_out_edges( (*this), e, node) {
            NodeID target = getEdgeTarget(e);
            if(!is_local_node(target)) {
                gDeg++;
            }
        } endfor
        ghost_degrees[node] = gDeg;
    }endfor
    return ghost_degrees;
}

std::tuple<EdgeWeight,EdgeWeight,NodeWeight> parallel_graph_access::get_ghostEdges_nodeWeight(){
    EdgeWeight interPEedges = 0;
    EdgeWeight localEdges = 0;
    NodeWeight localWeight = 0;
    forall_local_nodes( (*this), node) {
        localWeight += getNodeWeight(node);
        forall_out_edges( (*this), e, node) {
            NodeID target = getEdgeTarget(e);
            if(!is_local_node(target)) {
                interPEedges++;
            } else {
                localEdges++;
            }
        } endfor
    } endfor

    EdgeWeight globalInterEdges = 0;
    EdgeWeight globalIntraEdges = 0;
    NodeWeight globalWeight = 0;
    MPI_Reduce(&interPEedges, &globalInterEdges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT, m_communicator);
    MPI_Reduce(&localEdges, &globalIntraEdges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT, m_communicator);
    MPI_Allreduce(&localWeight, &globalWeight, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, m_communicator);

    return {globalInterEdges, globalIntraEdges, globalWeight };
}