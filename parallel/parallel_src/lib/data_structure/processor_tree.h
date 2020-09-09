/* From TiMer processor_graph_access 
 */

#ifndef PROCESSOR_TREE_H
#define PROCESSOR_TREE_H

using namespace std;

#include <vector>

class processor_tree
{


 public:

  /** @brief Default constructor.
   */
  processor_tree();

  /**	@brief Constructor to create a processor tree based on leaves.
    	@param[in] distances[vector]: each vector element corresponds to the communication costs on each level from the leaves
    	@param[in] descendants[vector]: each vector element corresponds to the number of descendants on each level
	@brief convention that ascending element positions (0,1,2, ...) corresponds to levels in the tree from higher to lower.
    */
  processor_tree(const vector<uint64_t> &distances, const vector<uint64_t> descendants ) {
    assert( distances.size() == descendants.size());
    traversalDistances = distances;
    traversalDescendants = descendants;
    numOfLevels = distances.size();
  }
  
  ~processor_tree();

  int getDistance_xy(int label_size, uint64_t x, uint64_t y) {
    uint64_t labelDiff = x ^ y;
    if(!labelDiff)
      return 0;
    int count_leading_zeros = __builtin_clzll(labelDiff); // index of highest bit
    int total_n_bits = 8*sizeof(unsigned long long int);
    int idx = total_n_bits - count_leading_zeros;
    assert(idx < label_size); // label of no more than label_size bits
    int j = label_size - idx - 1;
    if(j >= traversalDistances.size())
      return 0;
    return traversalDistances[j];
  }

  inline vector<uint64_t> get_traversalDistances() {return traversalDistances;};
  inline vector<uint64_t> get_traversalDescendants() {return traversalDescendants;};
  inline unsigned int get_numOfLevels() {return numOfLevels;};

  
 private:
  unsigned int numOfLevels;
  // Q: make distances dou
  vector<uint64_t> traversalDistances;
  vector<uint64_t> traversalDescendants;

};
  

#endif/* PROCESSOR_ΤREE_H */