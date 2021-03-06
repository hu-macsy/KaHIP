/******************************************************************************
 * distributed_quality_metrics.h
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Christian Schulz <christian.schulz.phone@gmail.com>
 *****************************************************************************/

#ifndef DISTRIBUTED_QUALITY_METRICS_UAVSEXBT
#define DISTRIBUTED_QUALITY_METRICS_UAVSEXBT



#include "pdefinitions.h"
#include "data_structure/parallel_graph_access.h"
#include "data_structure/processor_tree.h"
#include "ppartition_config.h"

class distributed_quality_metrics {
public:
        distributed_quality_metrics();
	distributed_quality_metrics(EdgeWeight qap, EdgeWeight cut);
        virtual ~distributed_quality_metrics();

        EdgeWeight local_edge_cut( parallel_graph_access & G, int * partition_map, MPI_Comm communicator );
        EdgeWeight edge_cut( parallel_graph_access & G, MPI_Comm communicator );
        EdgeWeight edge_cut_second( parallel_graph_access & G, MPI_Comm communicator  );
        NodeWeight local_max_block_weight( PPartitionConfig & config, parallel_graph_access & G, int * partition_map, MPI_Comm communicator  );
        double balance( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator  );
        double balance_load( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator  );
        double balance_load_dist( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator  );
        double balance_second( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator  );
	EdgeWeight comm_vol( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator  );
	EdgeWeight comm_bnd( PPartitionConfig & config, parallel_graph_access & G, MPI_Comm communicator );
	EdgeWeight comm_vol_dist( parallel_graph_access & G, MPI_Comm communicator );

	EdgeWeight total_qap( parallel_graph_access & G, const processor_tree &PEtree, MPI_Comm communicator);

	void set_initial_qap(EdgeWeight qap) {initial_qap = qap;};
	void set_initial_cut(EdgeWeight cut) {initial_cut = cut;};
	void set_initial_numNodes(NodeWeight size) {initial_numNodes = size;};
	void set_initial_numEdges(NodeWeight size) {initial_numEdges = size;};
	void add_timing(std::vector<double> vec);
	EdgeWeight get_initial_qap() { return initial_qap; };
	EdgeWeight get_initial_cut() { return initial_cut; };
        std::vector< double > get_cycle_time() { return ml_time; };
        double get_coarse_time() { return ml_time[0]; };
	double get_inpart_time() { return ml_time[1]; };
	double get_refine_time() { return ml_time[2]; };
	int get_max_congestion() { return max_congestion; };
	int get_max_dilation() { return max_dilation; };
	int get_sum_dilation() { return sum_dilation; };
	double get_avg_dilation() { return avg_dilation; };
	NodeWeight get_initial_numNodes(){ return initial_numNodes; };
	NodeWeight get_initial_numEdges(){ return initial_numEdges; };
	void print();
	/******************************************************/
	/*                  evaluateMapping                   */
	/******************************************************/
	void evaluateMapping(parallel_graph_access & C, const processor_tree & PEtree,
			     MPI_Comm communicator);
	void evaluateMappingDEBUG(parallel_graph_access & C, const processor_tree & PEtree,
			      MPI_Comm communicator);

	
	
private: 

        EdgeWeight initial_qap, initial_cut, initial_numEdges=0;
        std::vector< double > ml_time;
	int max_congestion, max_dilation, sum_dilation = 0;
	double avg_dilation = 0.0;
        NodeWeight initial_numNodes = 0;

};



#endif /* end of include guard: DISTRIBUTED_QUALITY_METRICS_UAVSEXBT */
