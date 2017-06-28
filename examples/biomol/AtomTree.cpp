#include "AtomTree.h"

ATCell::ATCell(const AtomTree& my_tree, long int i_, long int j_, long int k_, int level_)
  : i(i_),
    j(j_),
    k(k_),
    xc(my_tree.x_fr_i(i_)),
    yc(my_tree.y_fr_j(j_)),
    zc(my_tree.z_fr_k(k_)),
    dx((my_tree.brick.xyz_max[0] - my_tree.brick.xyz_min[0])/(1<<level_)),
    dy((my_tree.brick.xyz_max[1] - my_tree.brick.xyz_min[1])/(1<<level_)),
    dz((my_tree.brick.xyz_max[2] - my_tree.brick.xyz_min[2])/(1<<level_)),
    diag(sqrt(SQR(dx) + SQR(dy) + SQR(dz))),
    level(level_) {

  // default: one recursively refines the tree, one does not create a parent cell...
  is_leaf  = !(level_ < my_tree.finest_level);
}

void ATCell::set_node_index(const int ioff, const int joff, const int koff, const long long int node_index_in_node_table)
{
#ifdef CASL_THROWS
  if ((ioff != 0 && ioff != 1) || (joff != 0 && joff != 1) || (koff != 0 && koff != 1))
    throw std::invalid_argument("ATCell::set_node_index(const int, const int, const int, const int), three first arguments must be 0 or 1");
#endif
  node_indices[ioff][joff][koff] = node_index_in_node_table;
}
void ATCell::set_node_index(const bool ioff, const bool joff, const bool koff, const long long int node_index_in_node_table)
{
  set_node_index((ioff)?1:0, (joff)?1:0, (koff)?1:0, node_index_in_node_table);
}

long long int ATCell::get_node_index(const int ioff, const int joff, const int koff) const
{
#ifdef CASL_THROWS
  if ((ioff != 0 && ioff != 1) || (joff != 0 && joff != 1) || (koff != 0 && koff != 1))
    throw std::invalid_argument("ATCell::get_node_index(const int, const int, const int, const int), the three arguments must be 0 or 1");
#endif
  return node_indices[ioff][joff][koff];
}
long long int ATCell::get_node_index(const bool ioff, const bool joff, const bool koff) const
{
  return get_node_index((ioff)?1:0, (joff)?1:0, (koff)?1:0);
}

long long int ATCell::get_node_index_vtk_labeled(const int vtk_label) const
{
#ifdef CASL_THROWS
  if (vtk_label < 0 || vtk_label>=8)
    throw std::invalid_argument("ATCell::get_node_vtk_ordered(const int) requires a positive integer strictly smaller than 8.");
#endif
  return get_node_index((vtk_label%4==1||vtk_label%4==2),(vtk_label%4==2||vtk_label%4==3), vtk_label>3);
}

void ATCell::add_atom(const Atom &a)
{
  local_atoms.push_back(a);
}

void ATCell::clear_local_atoms()
{
  local_atoms.clear();
}

std::vector<Atom> ATCell::get_local_atoms() const
{
  return local_atoms;
}

double ATCell::dist_from_vdW_surface_to(const double& x, const double& y, const double& z) const
{
#ifdef CASL_THROWS
  if (! this->contains_point(x, y, z))
    throw std::invalid_argument("ATCell::dist_from_vdW_surface_to(const double&, const double&, const double&), the point must be in the ATCell.");
#endif
  double phi = -DBL_MAX;
  double test;
  for (size_t m = 0; m < local_atoms.size(); m++) {
    const Atom& a = local_atoms[m];
    test    = a.r - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z));
    phi     = MAX(phi, test);
  }
  return phi;
}

double ATCell::shortest_distance_from_vdW_surf_of(const Atom &a) const
{
  double shortest_distance; // negative outside the sphere associated with atom a
  if (contains_atom(a))
    shortest_distance = a.r;
  else
  {
    int ioff = (a.x > xc+.5*dx)? 1: (a.x < xc-.5*dx)? -1: 0;
    int joff = (a.y > yc+.5*dy)? 1: (a.y < yc-.5*dy)? -1: 0;
    int koff = (a.z > zc+.5*dz)? 1: (a.z < zc-.5*dz)? -1: 0;
    shortest_distance = a.r - sqrt(SQR(ioff*(a.x-(xc+ioff*.5*dx)))+SQR(joff*(a.y-(yc+joff*.5*dy)))+SQR(koff*(a.z-(zc+koff*.5*dz))));
  }
  return shortest_distance;
}

long long int AtomTree::morton_from_indices(long int index1_, long int index2_, long int index3_) const
{ // pack 3 20-bit indices into a 64-bit Morton code

  // convert to long long int to make sure it's encoded on 64 bits:
  long long int index1 = (long long int) index1_;
  long long int index2 = (long long int) index2_;
  long long int index3 = (long long int) index3_;
  // Consider a 64-bits integer
  // index = 0bxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxabcdefghijklmnopqrst where
  // x, a, b, ..., t = 0 or 1 (only the last 20 bits are relevant)
  // Assumption: no more than 2**19 logical grid points along any direction in the atom
  // tree (--> logical representation within 20 bits along each direction, otherwise, one
  // needs even longer encoding representation for the integer Morton value),
  // hence the x's are supposedly 0 --> trim the first 44 bits by bit-wise AND
  // with 0b0000000000000000000000000000000000000000000011111111111111111111 = 0x00000000000fffff
  index1 &= 0x00000000000fffff;
  index2 &= 0x00000000000fffff;
  index3 &= 0x00000000000fffff;
  // Now, it is sure that
  // index =  0b00000000000000000000000000000000000000000000abcdefghijklmnopqrst (as assumed)
  // We finally want
  // index =  0b000000a00b00c00d00e00f00g00h00i00j00k00l00m00n00o00p00q00r00s00t

  // Then, here below, we have
  // index =  0b00000000000000000000000000000000000000000000abcdefghijklmnopqrst
  //          0b000000000000abcdefghijklmnopqrst00000000000000000000000000000000
  index1 |= ( index1 << 32 );
  index2 |= ( index2 << 32 );
  index3 |= ( index3 << 32 );
  // index =  0b000000000000abcdefghijklmnopqrst000000000000abcdefghijklmnopqrst
  //        & 0b0000000000001111000000000000000000000000000000001111111111111111 (= 0x000f00000000ffff here below)
  index1 &= 0x000f00000000ffff;
  index2 &= 0x000f00000000ffff;
  index3 &= 0x000f00000000ffff;
  // index =  0b000000000000abcd00000000000000000000000000000000efghijklmnopqrst
  //        | 0b00000000000000000000000000000000efghijklmnopqrst0000000000000000
  index1 |= ( index1 << 16 );
  index2 |= ( index2 << 16 );
  index3 |= ( index3 << 16 );
  // index =  0b000000000000abcd0000000000000000efghijklmnopqrstefghijklmnopqrst
  //        & 0b0000000000001111000000000000000011111111000000000000000011111111 (= 0x000f0000ff0000ff here below)
  index1 &= 0x000f0000ff0000ff;
  index2 &= 0x000f0000ff0000ff;
  index3 &= 0x000f0000ff0000ff;
  // index =  0b000000000000abcd0000000000000000efghijkl0000000000000000mnopqrst
  //        | 0b0000abcd0000000000000000efghijkl0000000000000000mnopqrst00000000
  index1 |= ( index1 << 8 );
  index2 |= ( index2 << 8 );
  index3 |= ( index3 << 8 );
  // index =  0b0000abcd0000abcd00000000efghijklefghijkl00000000mnopqrstmnopqrst
  //        & 0b0000000000001111000000001111000000001111000000001111000000001111 (= 0x000f00f00f00f00f here below)
  // index =  0b000000000000abcd00000000efgh00000000ijkl00000000mnop00000000qrst
  index1 &= 0x000f00f00f00f00f;
  index2 &= 0x000f00f00f00f00f;
  index3 &= 0x000f00f00f00f00f;
  // index =  0b000000000000abcd00000000efgh00000000ijkl00000000mnop00000000qrst
  //        | 0b00000000abcd00000000efgh00000000ijkl00000000mnop00000000qrst0000
  index1 |= ( index1 << 4 );
  index2 |= ( index2 << 4 );
  index3 |= ( index3 << 4 );
  // index =  0b00000000abcdabcd0000efghefgh0000ijklijkl0000mnopmnop0000qrstqrst
  //        & 0b0000000011000011000011000011000011000011000011000011000011000011 (= 0x00c30c30c30c30c3 here below)
  index1 &= 0x00c30c30c30c30c3;
  index2 &= 0x00c30c30c30c30c3;
  index3 &= 0x00c30c30c30c30c3;
  // index =  0b00000000ab0000cd0000ef0000gh0000ij0000kl0000mn0000op0000qr0000st
  //        | 0b000000ab0000cd0000ef0000gh0000ij0000kl0000mn0000op0000qr0000st00
  index1 |= ( index1 << 2 );
  index2 |= ( index2 << 2 );
  index3 |= ( index3 << 2 );
  // index =  0b000000abab00cdcd00efef00ghgh00ijij00klkl00mnmn00opop00qrqr00stst
  //        & 0b0000001001001001001001001001001001001001001001001001001001001001 (= 0x0249249249249249 here below)
  index1 &= 0x0249249249249249;
  index2 &= 0x0249249249249249;
  index3 &= 0x0249249249249249;
  // index =  0b000000a00b00c00d00e00f00g00h00i00j00k00l00m00n00o00p00q00r00s00t YEAHAAA
  return( index1 | ( index2 << 1 ) | ( index3 << 2 ) );
}

void  AtomTree::build_tree(const std::vector<Atom> &atoms)
{
  total_atoms = atoms;
  // the maximum level of the tree will be self-determined by the parameter atoms_per_cell
  // (or abs_max_level if atoms_per_cell is too small for the given molecule)
  max_level = 0; // only the root so far
  long int i = max_i>>1; // logical coordinates of the center of root cell
  long int j = max_i>>1;
  long int k = max_i>>1;
  build_subtree(total_atoms, 0, i, j, k);
}

bool AtomTree::is_built() const
{
  return (max_level > 0);
}

void AtomTree::clear_tree()
{
  cell_table.clear();
  node_table.clear();
  total_atoms.clear();
  cells.clear();
  nodes.clear();
  interface_resolution.reset();
  probe_radius.reset();
  max_level = 0.;
}

void AtomTree::build_subtree(const std::vector<Atom> &atoms, int level, long int i, long int j, long int k)
{
  long long int cell_index = add_cell(atoms, level, i, j, k);

  ATCell *cell;
  cell = &(cells[cell_index]);

  if(!cell->is_leaf)
  {
    long int off = (max_i>>(level+2));
    const std::vector<Atom> new_atoms = cell->get_local_atoms();
    cell->clear_local_atoms(); // no need for those anymore, it is no longer a leaf cell
    build_subtree(new_atoms, level+1, i+off, j+off, k+off);
    build_subtree(new_atoms, level+1, i+off, j+off, k-off);
    build_subtree(new_atoms, level+1, i+off, j-off, k+off);
    build_subtree(new_atoms, level+1, i+off, j-off, k-off);
    build_subtree(new_atoms, level+1, i-off, j+off, k+off);
    build_subtree(new_atoms, level+1, i-off, j+off, k-off);
    build_subtree(new_atoms, level+1, i-off, j-off, k+off);
    build_subtree(new_atoms, level+1, i-off, j-off, k-off);
    max_level = MAX(max_level, level+1);
  }
}

long long int  AtomTree::add_cell(const std::vector<Atom> &atoms, int level, long int i, long int j, long int k)
{
  ATCell new_cell(*this, i, j, k, level);

  long long int cell_index = cells.size();
  long long int morton_index = morton_from_indices(i,j,k);
  cell_table[morton_index] = cell_index;

  set_cell_nodes(new_cell);
  set_atoms_belonging_to_cell(new_cell, atoms);
  cells.push_back(new_cell);

  return cell_index;
}

void AtomTree::set_atoms_belonging_to_cell(ATCell &cell, const std::vector<Atom> &atoms)
{
#ifdef CASL_THROWS
  if (!interface_resolution.is_assigned() || !probe_radius.is_assigned())
    throw std::runtime_error("[CASL_ERROR] AtomTree::build_tree(const std::vector<Atoms>& ), critical parameter(s) (probe radius and/or interface resolution) is/are not defined...");
#endif
  const double rp = probe_radius.get_value();
  const double threshold = interface_resolution.get_value();
  for (size_t a=0; a<atoms.size(); a++)
  {
    if (rp+cell.shortest_distance_from_vdW_surf_of(atoms[a]) > -threshold)
    {
      cell.add_atom(atoms[a]);
    }
  }

  if(cell.get_number_of_atoms() <= atoms_per_cell)
  {
    cell.is_leaf = true;
  }

  // if empty, any atom in the list is good, ok after reinitialization
  if(cell.is_empty())
    cell.add_atom(atoms[0]);
}

void AtomTree::set_cell_nodes(ATCell &cell)
{
  long int node_i, node_j, node_k;
  long long int morton_index, node_index;
  for (int xx = 0; xx < 2; ++xx) {
    for (int yy = 0; yy < 2; ++yy) {
      for (int zz = 0; zz < 2; ++zz) {
        node_i = cell.i+(xx==1?1:-1)*(max_i>>(cell.level+1));
        node_j = cell.j+(yy==1?1:-1)*(max_i>>(cell.level+1));
        node_k = cell.k+(zz==1?1:-1)*(max_i>>(cell.level+1));
        morton_index = morton_from_indices(node_i, node_j, node_k);
        if(node_table.find(morton_index) == node_table.end()) //Node doesn't exist yet, so create it
        {
          node_index = nodes.size();
          node_table[morton_index] = node_index;
          ATNode new_node(node_i, node_j, node_k);
          nodes.push_back(new_node);
          cell.set_node_index(xx, yy, zz, node_index);
        }
        else
          cell.set_node_index(xx, yy, zz, node_table.at(morton_index));
      }
    }
  }
}

double AtomTree::dist_from_SAS(const double& x, const double& y, const double& z) const
{
  const double rp = probe_radius.get_value();
  long long int cell_index = find_smallest_cell_containing_point(x,y,z);
  const ATCell& my_cell = cells[cell_index];
  return rp + my_cell.dist_from_vdW_surface_to(x, y, z);
}

int AtomTree::num_atoms_queried(const double& x, const double& y, const double& z) const
{

  if( x<brick.xyz_min[0] || x >brick.xyz_max[0] ||
      y<brick.xyz_min[1] || y >brick.xyz_max[1] ||
      z<brick.xyz_min[2] || z >brick.xyz_max[2] ){ //outside range
    return (int) total_atoms.size();
  }
  else //inside domain
  {
    long long int cell_index = find_smallest_cell_containing_point(x,y,z);
    return get_number_of_atoms_in_cell(cell_index);
  }
}

void AtomTree::set_probe_radius(const double rp_)
{
  probe_radius.set_value(rp_);
}

void AtomTree::reset_probe_radius(const double rp_)
{
  if (is_built())
  {
    clear_tree();
    set_probe_radius(rp_);
  }
  else
  {
    probe_radius.reset();
    probe_radius.set_value(rp_);
  }
}

void AtomTree::set_interface_resolution(const double resolution_)
{
  interface_resolution.set_value(resolution_);
  int abs_max_level_copy = abs_max_level; // the compiler needs a clear reference for the following...
  finest_level = MIN((int) ceil(log2(sqrt(SQR(brick.xyz_max[0] - brick.xyz_min[0])+SQR(brick.xyz_max[1] - brick.xyz_min[1])+SQR(brick.xyz_max[2] - brick.xyz_min[2]))/resolution_)), abs_max_level_copy);
}

void AtomTree::reset_interface_resolution(const double resolution_)
{
  if (is_built())
  {
    clear_tree();
    set_interface_resolution(resolution_);
  }
  else
  {
    interface_resolution.reset();
    interface_resolution.set_value(resolution_);
  }
}

long long int AtomTree::find_smallest_cell_containing_point(const double& x, const double& y, const double& z) const
{
  long long int morton_index = morton_from_indices(max_i>>1, max_i>>1, max_i>>1); // brick center
  int level=0;
  long long int cell_index=0;

  while(cell_table.find(morton_index) != cell_table.end() && level<=max_level)
  {
    cell_index = cell_table.at(morton_index);
    //compute next level morton_index
    level++;
    long int i,j,k;
    i = cell_i_fr_x(x,level);
    j = cell_j_fr_y(y,level);
    k = cell_k_fr_z(z,level);


    morton_index = morton_from_indices(i,j,k);
  }
  return cell_index;
}

void AtomTree::print_atom_count_per_cell( std::string file_name) const
{
  std::vector<int> data;

  for (int n=0; n<number_of_cells(); n++)
    if(cells[n].is_leaf)
      data.push_back(cells[n].get_number_of_atoms());

  print_VTK_format(file_name);
  print_VTK_format(data, "num_atoms", file_name);
}

void AtomTree::print_VTK_format( std::string file_name, double time ) const
{
  int num_of_leaf_cells;
  double x, y,z;
  FILE *outFile = fopen(file_name.c_str(),"w");
#ifdef CASL_THROWS
  if(outFile == NULL) throw std::invalid_argument("[CASL_ERROR]: Cannot open file.");
#endif

  fprintf(outFile,"# vtk DataFile Version 2.0 \n");
  fprintf(outFile,"AtomTree Mesh \n");
  fprintf(outFile,"ASCII \n");
  fprintf(outFile,"DATASET UNSTRUCTURED_GRID \n");

  if(time != DBL_MIN)
  {
    fprintf(outFile,"FIELD FieldData 1 \n");
    fprintf(outFile,"TIME 1 1 double \n");
    fprintf(outFile,"%e",time);
  }

  fprintf(outFile,"POINTS %d double \n",(int) nodes.size());
  for (size_t n=0; n<nodes.size(); n++)
  {
    x = x_fr_i(nodes[n].i);
    y = y_fr_j(nodes[n].j);
    z = z_fr_k(nodes[n].k);
    fprintf(outFile,"%f %f %f\n",x,y,z);
  }

  num_of_leaf_cells = number_of_leaves();

  fprintf(outFile,"CELLS %d %d \n",num_of_leaf_cells,9*num_of_leaf_cells);
  for (size_t n=0; n<cells.size(); n++)
  {
    if ( cells[n].is_leaf)
    {
      fprintf(outFile,"%d",8);
      for (int k=0; k<8; k++)
        fprintf(outFile," %lld",cells[n].get_node_index_vtk_labeled(k));
      fprintf(outFile,"\n");
    }
  }
  fprintf(outFile,"CELL_TYPES %d \n",num_of_leaf_cells);
  for (int n=0; n<num_of_leaf_cells; n++)
    fprintf(outFile,"%d \n",12);
  fprintf(outFile,"CELL_DATA %d \n",num_of_leaf_cells);
  fclose (outFile);
}

void AtomTree::print_VTK_format(std::vector<double> &F, std::string data_name, std::string file_name ) const
{
  FILE *outFile;
  outFile = fopen(file_name.c_str(),"a");
  fprintf(outFile,"SCALARS %s double 1 \n",data_name.c_str());
  fprintf(outFile,"LOOKUP_TABLE default \n");
  for (int n=0; n<number_of_leaves(); n++) fprintf(outFile,"%E \n",F[n]);
  fclose (outFile);
}

void AtomTree::print_VTK_format(std::vector<int> &F, std::string data_name, std::string file_name ) const
{
  FILE *outFile;
  outFile = fopen(file_name.c_str(),"a");
  fprintf(outFile,"SCALARS %s double 1 \n",data_name.c_str());
  fprintf(outFile,"LOOKUP_TABLE default \n");
  for (int n=0; n<number_of_leaves(); n++) fprintf(outFile,"%d \n",F[n]);
  fclose (outFile);
}


