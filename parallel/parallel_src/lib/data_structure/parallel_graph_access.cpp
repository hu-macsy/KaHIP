/******************************************************************************
 * parallel_graph_access.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Christian Schulz <christian.schulz.phone@gmail.com>
 *****************************************************************************/

#include <numeric>
#include <algorithm>

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
/* init balance management based on an input partitioned graph */
void parallel_graph_access::init_balance_management_from_graph( PPartitionConfig & config,
								parallel_graph_access & G) {
        if( m_bm != NULL ) {
                delete m_bm;
        }

        assert( config.total_num_labels == config.k );
	m_bm = new balance_management_refinement( this, config.total_num_labels, G );
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




void parallel_graph_access::compute_reduced_adjacent_edges_aggressive(std::vector<bool> is_high_degree_node ,
								     std::vector< std::vector< NodeID > > &local_edge_lists,
								     std::vector< std::vector< NodeID > > &local_edge_weights,
								     EdgeID & edge_counter ) {
	assert(is_high_degree_node.size() == (*this).number_of_global_nodes());
	assert( local_edge_lists.size() == local_edge_weights.size() );
	assert( local_edge_lists.size() == (*this).number_of_local_nodes()); 
	ULONG numIsolatedHdnodes = 0;
	ULONG numIsolatedNodes = 0;
	
	forall_local_nodes((*this),u) {
		if (is_high_degree_node[(*this).getNodeLabel(u)] == true) {
			// for high degree nodes
			forall_out_edges((*this), e, u) {
				NodeID v = (*this).getEdgeTarget(e);
				EdgeWeight weight = (*this).getEdgeWeight(e);
				// add the first local node u --> v
				if ((*this).is_local_node(v)) {
					local_edge_lists[u].push_back((*this).getNodeLabel(v));
					local_edge_weights[u].push_back(weight);
					edge_counter++;
					// add also the other direction v --> u
					local_edge_lists[v].push_back((*this).getNodeLabel(u));
					local_edge_weights[v].push_back(weight);
					edge_counter++;
					break;
				} // if no local adjacent node exists, u stays isolated // highly possible
				else {
		        	  numIsolatedHdnodes++;
		        	}
			
			} endfor

			    }
		else {
			forall_out_edges((*this), e, u) {
				NodeID v = (*this).getEdgeTarget(e);
				if (is_high_degree_node[(*this).getNodeLabel(v)] != true) {
					EdgeWeight weight = (*this).getEdgeWeight(e);
					local_edge_lists[u].push_back((*this).getNodeLabel(v));
					local_edge_weights[u].push_back(weight);
					edge_counter++;
				}
				else {
		        	  numIsolatedNodes++;
		        	}
				
			} endfor
				  }
	} endfor

	    if(numIsolatedHdnodes > 0 || numIsolatedNodes > 0){
	      std::cout << "WARNING, rank " << rank << " now has isolated " << numIsolatedHdnodes << " (high degree) nodes and " << numIsolatedNodes  << " normal nodes. " <<  std::endl; 
	    }
	    
}

void parallel_graph_access::compute_reduced_adjacent_edges(std::vector<bool> is_high_degree_node ,
							   std::vector< std::vector< NodeID > > &local_edge_lists,
							   std::vector< std::vector< NodeID > > &local_edge_weights,
							   EdgeID &edge_counter) {

	assert(is_high_degree_node.size() == (*this).number_of_global_nodes());
	assert( local_edge_lists.size() == local_edge_weights.size());
	assert( local_edge_lists.size() == (*this).number_of_local_nodes());
	ULONG numIsolatedHdnodes = 0;
	// ULONG numIsolatedNodes = 0;
	
	forall_local_nodes((*this),u) {
		if (is_high_degree_node[(*this).getNodeLabel(u)] == true) {
			/* parse the edge list of u */
			EdgeID edge_count_per_node = 0;
			long first_local_edge = -1;
			forall_out_edges((*this), e, u) {
				NodeID v = (*this).getEdgeTarget(e);
				EdgeWeight weight = (*this).getEdgeWeight(e);
				// add edges to all non high degree nodes, v
				if ( !is_high_degree_node[(*this).getNodeLabel(v)] ) {
					local_edge_lists[u].push_back((*this).getNodeLabel(v));
					local_edge_weights[u].push_back(weight);	      
					edge_counter++;
					edge_count_per_node++;
				}
				else if (first_local_edge == -1 && (*this).is_local_node(v)) {
					/* store the first local node in the edge list of u */
					first_local_edge = e;
				}
			} endfor
			if (edge_count_per_node == 0) {
				// u has no non high degree neighbors, connect it with first local (high degree) node
				// if local node exists:
				if ( first_local_edge != -1 ) {
					// connect u --> v (local node)
					NodeID v = (*this).getEdgeTarget((EdgeID) first_local_edge);
					EdgeWeight weight = (*this).getEdgeWeight((EdgeID) first_local_edge);
					local_edge_lists[u].push_back((*this).getNodeLabel(v));
					local_edge_weights[u].push_back(weight);              
					edge_counter++;
					// connect also v --> u
					NodeID w = (*this).getNodeLabel(u);
					local_edge_lists[v].push_back(w);
					local_edge_weights[v].push_back(weight);              
					edge_counter++;
				}
				// there is not a local neighboring node -- weird behaviour
				else{
				  numIsolatedHdnodes++;
				} 
				
			}
		}
		else {
		  /* Nodes that are not high degree edges should simply add their entire edge list. */
			/* Their edge list is not expected to be long anyway. */
			forall_out_edges((*this), e, u) {
				NodeID v = (*this).getEdgeTarget(e);
				EdgeWeight weight = (*this).getEdgeWeight(e);
				local_edge_lists[u].push_back((*this).getNodeLabel(v));
				local_edge_weights[u].push_back(weight);
				edge_counter++;					
			} endfor
	        }
	} endfor

	    if(numIsolatedHdnodes > 0){
	      std::cout << "WARNING, rank " << rank << " now has " << numIsolatedHdnodes << " isolated (high degree) nodes " <<  std::endl; 
	    }


}

void parallel_graph_access::get_reduced_graph(parallel_graph_access & outG, std::vector< NodeID > node_list, MPI_Comm communicator,
					      const bool aggressive_removal) {
	assert(!node_list.empty());
	int rank, comm_size;
	MPI_Comm_rank( communicator, &rank);
	MPI_Comm_size( communicator, &comm_size);
	NodeID global_nnodes = (*this).number_of_global_nodes();
	NodeID local_nnodes = (*this).number_of_local_nodes();

		
	std::vector<bool> is_high_degree_node(global_nnodes, false);
	for(auto& u : node_list)
		is_high_degree_node[u] = true;
		
	NodeID n = global_nnodes;
	ULONG from  = rank     * ceil(n / (double)comm_size);
	ULONG to    = (rank+1) * ceil(n / (double)comm_size) - 1;
	to = std::min<unsigned long>(to, n-1);
	  
	  
	std::vector< std::vector< NodeID > > local_edge_lists;
	local_edge_lists.resize(local_nnodes);
	std::vector< std::vector< NodeID > > local_edge_weights;
	local_edge_weights.resize(local_nnodes);
	EdgeID edge_counter = 0;
	
	if (aggressive_removal)
		(*this).compute_reduced_adjacent_edges_aggressive(is_high_degree_node,local_edge_lists,
								 local_edge_weights, edge_counter);
	else
		(*this).compute_reduced_adjacent_edges(is_high_degree_node,local_edge_lists,
						       local_edge_weights, edge_counter);

		  
	int t_edge_count = 0;
	MPI_Allreduce(&edge_counter, &t_edge_count, 1, MPI_INT, MPI_SUM, communicator);
	
	outG.start_construction(local_nnodes, edge_counter*2, global_nnodes, t_edge_count);
	outG.set_range(from, to);
	
		
	std::vector< NodeID > vertex_dist( comm_size+1, 0 );
	for( PEID peID = 0; peID <= comm_size; peID++) {
		vertex_dist[peID] = peID * ceil(n / (double)comm_size); // from positions
	}
	outG.set_range_array(vertex_dist);
	
	
	
	for (NodeID i = 0; i < local_nnodes; ++i) {
		NodeID node = outG.new_node();
		outG.setNodeWeight(node, 1);
		outG.setNodeLabel(node, from+node);
		outG.setSecondPartitionIndex(node, 0);
		for( ULONG j = 0; j < local_edge_lists[i].size(); j++) {
			NodeID target = local_edge_lists[i][j];
			EdgeID e = outG.new_edge(node, target);
			EdgeWeight weight = local_edge_weights[i][j];
			outG.setEdgeWeight(e, weight);
		}		
	}
    	
	
	outG.finish_construction(); 
	MPI_Barrier(communicator);
}
  

void parallel_graph_access::get_graph_copy(parallel_graph_access & outG, MPI_Comm communicator) {

	int rank, comm_size;
	MPI_Comm_rank( communicator, &rank);
	MPI_Comm_size( communicator, &comm_size);
	
	NodeID global_nnodes = (*this).number_of_global_nodes();
	
	 	
	NodeID n = global_nnodes;
	ULONG from  = rank     * ceil(n / (double)comm_size);
	ULONG to    = (rank+1) * ceil(n / (double)comm_size) - 1;
	to = std::min<unsigned long>(to, n-1);
	ULONG local_no_nodes = 0;
	if (from <= to)
		local_no_nodes = to - from + 1;
	
	

	std::vector< std::vector< NodeID > > local_edge_lists;
	local_edge_lists.resize(local_no_nodes);
	std::vector< std::vector< NodeID > > local_edge_weights;
	local_edge_weights.resize(local_no_nodes);
	
	EdgeID edge_counter = 0;
	
	forall_local_nodes((*this),u) {
		forall_out_edges((*this), e, u) {
			NodeID v = (*this).getEdgeTarget(e);
			EdgeWeight weight = (*this).getEdgeWeight(e);
			local_edge_lists[u].push_back((*this).getNodeLabel(v));
			local_edge_weights[u].push_back(weight);
			edge_counter++;
		} endfor
	} endfor
				      

       // for (int i = 0; i < local_edge_lists.size(); i++) {
       // 	std::cout << "R:" << rank << " node-local_edge_list: " << i << " ";
       // 	for (int j = 0; j < local_edge_lists[i].size(); j++)
       // 		std::cout <<  local_edge_lists[i][j] << "  ";
       // 	std::cout << std::endl;
       // }		      
	  
        int t_edge_count = 0;
	MPI_Allreduce(&edge_counter, &t_edge_count, 1, MPI_INT, MPI_SUM, communicator);
	
	outG.set_number_of_global_edges(t_edge_count);     
	outG.start_construction((NodeID) local_no_nodes, edge_counter*2, global_nnodes, t_edge_count);
	outG.set_range(from, to);
	
	std::vector< NodeID > vertex_dist( comm_size+1, 0 );
	for( PEID peID = 0; peID <= comm_size; peID++) {
		vertex_dist[peID] = peID * ceil(n / (double)comm_size); // from positions
	}
	outG.set_range_array(vertex_dist);
	
	
	
	for (NodeID i = 0; i < local_no_nodes; ++i) {
		NodeID node = outG.new_node();
		outG.setNodeWeight(node, 1);
		outG.setNodeLabel(node, from+node);
		outG.setSecondPartitionIndex(node, 0);
		for( ULONG j = 0; j < local_edge_lists[i].size(); j++) {
			NodeID target = local_edge_lists[i][j];
			EdgeID e = outG.new_edge(node, target);
			EdgeWeight weight = local_edge_weights[i][j];
			outG.setEdgeWeight(e, weight);
		}
		
	}
	
	outG.finish_construction(); 
	MPI_Barrier(communicator);
         
}


std::vector<NodeID> parallel_graph_access::get_high_degree_local_nodes_by_degree(const NodeID minDegree) {
    std::vector<NodeID> local_high_degree_nodes;

    forall_local_nodes((*this), node) {
        const double node_degree = getNodeDegree(node); //TODO: change to ghost_degree?
        //if this node has a higher degree than we allow
        if (node_degree>minDegree){
            assert( is_local_node(node) );
            local_high_degree_nodes.push_back( getGlobalID(node) );
        } 
    }endfor
    
    return local_high_degree_nodes;
}

std::vector<NodeID> parallel_graph_access::get_high_degree_local_nodes_by_num(const NodeID numNodes) {
    //store local degrees 
    std::vector<NodeID> local_degrees( number_of_local_nodes() );
    forall_local_nodes((*this), node) {
        local_degrees[node] = getNodeDegree(node);
    }endfor
    
    //store the global IDs of the local vertices
    std::vector<NodeID> global_indices( number_of_local_nodes() );
    forall_local_nodes((*this), node) {
        global_indices[node] = getGlobalID(node);
    }endfor
    //if node IDs are consecutive, the for above is equivalent with
    //std::iota( local_indices.begin(), local_indices.end(), get_from_range() );

    //sort vertex IDs in increasing order based of their degree
    std::sort( global_indices.begin(), global_indices.end(), 
        [&](NodeID i, NodeID j){
            return local_degrees[getLocalID(i)]>local_degrees[getLocalID(j)]; 
        }
    );
    //in case numNodes is greater than all the local nodes
    NodeID size = std::min( numNodes, (NodeID) global_indices.size() );
    return {global_indices.begin(), global_indices.begin()+size};
}

std::vector<NodeID> parallel_graph_access::get_high_ghost_degree_local_nodes_by_degree(const NodeID minDegree) {
    std::vector<NodeID> local_high_degree_nodes;
    std::vector<NodeID> ghostDeg = get_local_ghost_degrees();
    assert( ghostDeg.size() == number_of_local_nodes() );

    forall_local_nodes((*this), node) {
        const double node_degree = ghostDeg[node]; //TODO: change to ghost_degree?
        //if this node has a higher degree than we allow
        if (node_degree>minDegree){
            assert( is_local_node(node) );
            local_high_degree_nodes.push_back( getGlobalID(node) );
        } 
    }endfor
    
    return local_high_degree_nodes;
}

std::vector<NodeID> parallel_graph_access::get_high_ghost_degree_local_nodes_by_num( const NodeID numNodes) {
    std::vector<NodeID> local_high_degree_nodes;
    std::vector<NodeID> ghostDeg = get_local_ghost_degrees();
    assert( ghostDeg.size() == number_of_local_nodes() );

    //store the global IDs of the local vertices
    std::vector<NodeID> global_indices( number_of_local_nodes() );
    forall_local_nodes((*this), node) {
        global_indices[node] = getGlobalID(node);
    }endfor
    //if node IDs are consecutive, the for above is equivalent with
    //std::iota( local_indices.begin(), local_indices.end(), get_from_range() );

    //sort vertex IDs in increasing order based of their ghost degree
    std::sort( global_indices.begin(), global_indices.end(), 
        [&](NodeID i, NodeID j){
            return ghostDeg[getLocalID(i)]>ghostDeg[getLocalID(j)]; 
        }
    );

    //in case numNodes is greater than all the local nodes
    NodeID size = std::min( numNodes, (NodeID) global_indices.size() );
    return {global_indices.begin(), global_indices.begin()+size};
}

std::vector<NodeID> parallel_graph_access::get_high_degree_global_nodes_by_degree(const NodeID minDegree, const bool useGhostDegree ) {
    
    std::vector<NodeID> local_high_degree_nodes;
    if( useGhostDegree ){
        local_high_degree_nodes = get_high_ghost_degree_local_nodes_by_degree(minDegree);
    }else{
        local_high_degree_nodes = get_high_degree_local_nodes_by_degree(minDegree);
    }

    return get_all_global_nodes(local_high_degree_nodes);
}

std::vector<NodeID> parallel_graph_access::get_high_degree_global_nodes_by_num(const NodeID numNodes, const bool useGhostDegree ) {
    
    std::vector<NodeID> local_high_degree_nodes;
    if( useGhostDegree ){
        local_high_degree_nodes = get_high_ghost_degree_local_nodes_by_num(numNodes);
    }else{
        local_high_degree_nodes = get_high_degree_local_nodes_by_num(numNodes);
    }

    return get_all_global_nodes(local_high_degree_nodes);

}

std::vector<NodeID> parallel_graph_access::get_all_global_nodes(const std::vector<NodeID> local_nodes ) {

    //from the local high degree nodes we need to construct a replicated vector with all nodes
    const int num_local_hdn = local_nodes.size();

    //gather the number of nodes in the root
    std::vector<int> hdn_root( size );
    MPI_Gather( &num_local_hdn, 1, MPI_INT, hdn_root.data(), 1, MPI_INT, ROOT, m_communicator );

    //calculate prefix sum, aka displacement
    std::vector<int> displ( size, 0 );              //we do not need the last element
    for( unsigned int i=1; i<size; i++){
        displ[i] = displ[i-1]+ hdn_root[i-1] ;   //TODO/CHECK: is the +1 needed?
    }

    std::vector<NodeID> all_hdn;
    if( rank==ROOT){
        all_hdn.resize( displ.back()+hdn_root.back() );
    }

    MPI_Gatherv( local_nodes.data(), num_local_hdn, MPI_UNSIGNED_LONG_LONG, \
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
