#include "AtomTree.h"


int AtomTree::morton_from_indices(int index1, int index2, int index3) const
    { // pack 3 10-bit indices into a 30-bit Morton code
      index1 &= 0x000003ff;
      index2 &= 0x000003ff;
      index3 &= 0x000003ff;
      index1 |= ( index1 << 16 );
      index2 |= ( index2 << 16 );
      index3 |= ( index3 << 16 );
      index1 &= 0x030000ff;
      index2 &= 0x030000ff;
      index3 &= 0x030000ff;
      index1 |= ( index1 << 8 );
      index2 |= ( index2 << 8 );
      index3 |= ( index3 << 8 );
      index1 &= 0x0300f00f;
      index2 &= 0x0300f00f;
      index3 &= 0x0300f00f;
      index1 |= ( index1 << 4 );
      index2 |= ( index2 << 4 );
      index3 |= ( index3 << 4 );
      index1 &= 0x030c30c3;
      index2 &= 0x030c30c3;
      index3 &= 0x030c30c3;
      index1 |= ( index1 << 2 );
      index2 |= ( index2 << 2 );
      index3 |= ( index3 << 2 );
      index1 &= 0x09249249;
      index2 &= 0x09249249;
      index3 &= 0x09249249;
      return( index1 | ( index2 << 1 ) | ( index3 << 2 ) );
    }


void  AtomTree::build_tree(const std::vector<Atom> &atoms, const my_p4est_brick_t &brick_)
{
    total_atoms = atoms;
    brick=brick_;

    max_ar = 0;
    for(int m=0; m<atoms.size(); m++)
        max_ar = max(max_ar, atoms[m].r);

    //Choose a max level for a grid size on the order of the probe radius plus max atom radius.
    threshold =  1.5*(rp+max_ar);
    max_level = int(log2((brick.xyz_max[0] - brick.xyz_min[0])/threshold)) + 1 + 1;
    max_level = min(1*abs_max_level, max_level);

    int level = 0; //root
    int i = max_i>>1; //root cell
    int j = max_i>>1;
    int k = max_i>>1;

    if(total_atoms.size()>0)
        build_subtree(total_atoms, level, i, j, k);

}

void AtomTree::clear_tree()
{
    cell_table.clear();
    node_table.clear();
    total_atoms.clear();
    cells.clear();
    nodes.clear();
}


void  AtomTree::build_subtree(const std::vector<Atom> &atoms, int level, int i, int j, int k)
{
    ATCell *cell;
    int cell_index = add_cell(atoms, level, i, j, k);

    cell = &(cells[cell_index]);

    if(cell->level >= max_level)
        cell->refine = false;

    if(cell->refine)
    {
        int off = (max_i>>(level+2));
        std::vector<Atom> new_atoms = cell->atoms;
        cell->atoms.clear();
        cell->is_leaf = false;
        build_subtree(new_atoms, level+1, i+off, j+off, k+off);
        build_subtree(new_atoms, level+1, i+off, j+off, k-off);
        build_subtree(new_atoms, level+1, i+off, j-off, k+off);
        build_subtree(new_atoms, level+1, i+off, j-off, k-off);
        build_subtree(new_atoms, level+1, i-off, j+off, k+off);
        build_subtree(new_atoms, level+1, i-off, j+off, k-off);
        build_subtree(new_atoms, level+1, i-off, j-off, k+off);
        build_subtree(new_atoms, level+1, i-off, j-off, k-off);
    }

}


int  AtomTree::add_cell(const std::vector<Atom> &atoms, int level, int i, int j, int k)
{


    ATCell new_cell;
    int cell_index = cells.size();
    int morton_index = morton_from_indices(i,j,k);
    cell_table[morton_index] = cell_index;

    new_cell.i=i;
    new_cell.j=j;
    new_cell.k=k;
    new_cell.is_leaf = true;
    new_cell.level = level;
    new_cell.refine = true; //default

    set_cell_nodes(new_cell);

    set_atoms_belonging_to_cell(new_cell, atoms);

    cells.push_back(new_cell);

    return cell_index;

}

void AtomTree::set_atoms_belonging_to_cell(ATCell &cell, const std::vector<Atom> &atoms)
{
    std::vector<Atom> new_atoms;

    double dx = (brick.xyz_max[0] - brick.xyz_min[0])/(1<<cell.level);
    double dy = (brick.xyz_max[1] - brick.xyz_min[1])/(1<<cell.level);
    double dz = (brick.xyz_max[2] - brick.xyz_min[2])/(1<<cell.level);

    double x = x_fr_i(cell.i);
    double y = y_fr_j(cell.j);
    double z = z_fr_k(cell.k);

    double d = sqrt(dx*dx+dy*dy+dz*dz);
    for (int a=0; a<atoms.size(); a++)
    {
        if(fabs(atoms[a].x - x) < .5*dx*(1.5)+threshold &&
           fabs(atoms[a].y - y) < .5*dy*(1.5)+threshold &&
           fabs(atoms[a].z - z) < .5*dz*(1.5)+threshold )
        {
            new_atoms.push_back(atoms[a]);
        }
    }


    if(new_atoms.size()>atoms_per_cell)
        cell.refine = true;
    else
        cell.refine = false;


    if(new_atoms.size()==0) //No atoms in leaf cell. Add closest atom to each node.
    {
        double x[8],y[8],z[8];
        double min_dist[8];
        int best_so_far[8];
        for(int n=0; n<8; n++)
        {
            ATNode *node = &nodes[cell.nodes[n]];

            x[n] = x_fr_i(node->i);
            y[n] = y_fr_j(node->j);
            z[n] = z_fr_k(node->k);
            min_dist[n] = 1.e8;
        }

        for (int a=0; a<atoms.size(); a++)
        {
            for(int n=0; n<8; n++)
            {
                double dist = sqrt( (atoms[a].x - x[n])*(atoms[a].x - x[n])+
                                    (atoms[a].y - y[n])*(atoms[a].y - y[n])+
                                    (atoms[a].z - z[n])*(atoms[a].z - z[n]));

                if(dist<min_dist[n])
                {
                    min_dist[n] = dist;
                    best_so_far[n] = a;
                }

            }

        }



        for(int n=0; n<8; n++) //Check for duplicates
            for(int k=n+1; k<8; k++)
                if(best_so_far[n] == best_so_far[k])
                    best_so_far[k] = -1;

        for(int n=0; n<8; n++)
            if(best_so_far[n]!=-1) //is not a duplicate
                new_atoms.push_back(atoms[best_so_far[n]]);

    }

   cell.atoms = new_atoms;

}

void AtomTree::set_cell_nodes(ATCell &cell)
{

    int ioff[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
    int joff[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
    int koff[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
    for(int n=0; n<8; n++)
    {
        int node_i = cell.i+ioff[n]*(max_i>>(cell.level+1));
        int node_j = cell.j+joff[n]*(max_i>>(cell.level+1));
        int node_k = cell.k+koff[n]*(max_i>>(cell.level+1));
        int morton_index = morton_from_indices(node_i, node_j, node_k);

        if(node_table.find(morton_index) == node_table.end()) //Node doesn't exist yet, so create it
        {
            int node_index = nodes.size();
            node_table[morton_index] = node_index;
            ATNode new_node(node_i, node_j, node_k);
            nodes.push_back(new_node);
            cell.nodes[n] = node_index;

        }
        else
        {
            cell.nodes[n] = node_table[morton_index];
        }
    }

}

double AtomTree::dist_from_surface(double x, double y, double z) const
{

    const std::vector<Atom> *atoms;
    int cell_index;

    cell_index = find_smallest_cell_containing_point(x,y,z);
    atoms = &(cells[cell_index].atoms);



    double phi = -DBL_MAX;
    for (int m = 0; m < atoms->size(); m++) {
      const Atom& a = ((*atoms)[m]);
      double test = a.r + rp - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z));


      phi = MAX(phi, test);
    }

    return phi;

}

double AtomTree::num_atoms_queried(double x, double y, double z) const
{

    const std::vector<Atom> *atoms;
    int cell_index;

    if( x<brick.xyz_min[0] || x >brick.xyz_max[0] ||
        y<brick.xyz_min[1] || y >brick.xyz_max[1] ||
        z<brick.xyz_min[2] || z >brick.xyz_max[2] ){ //outside range

        atoms = &total_atoms;

    }
    else //inside domain
    {
        cell_index = find_smallest_cell_containing_point(x,y,z);
        atoms = &(cells[cell_index].atoms);

    }


    return 1.*(*atoms).size();

}

void AtomTree::set_probe_radius(double rp_)
{
    rp = rp_;
}

void AtomTree::set_max_atom_radius(double rmax_)
{
    rmax = rmax_;
}



int AtomTree::find_smallest_cell_containing_point(double x, double y, double z) const
{
    int morton_index = morton_from_indices(max_i>>1, max_i>>1, max_i>>1);
    int level=0;
    int cell_index=0;

    while(cell_table.find(morton_index) != cell_table.end() && level<=max_level)
    {
        cell_index = cell_table.at(morton_index);// cell_table[morton_index];
        //compute next level morton_index
        level++;
        int i,j,k;
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
            data.push_back(cells[n].atoms.size());

    print_VTK_format(file_name);
    print_VTK_format(data, "num_atoms", file_name);

}



void AtomTree::print_VTK_format( std::string file_name, double time ) const
{
    int num_of_leaf_cells;
    int node_of_cell[8];
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

    fprintf(outFile,"POINTS %d double \n",nodes.size());
    for (int n=0; n<nodes.size(); n++)
    {
        x = x_fr_i(nodes[n].i);
        y = y_fr_j(nodes[n].j);
        z = z_fr_k(nodes[n].k);
        //printf("%d %d %d %f %f %f\n", nodes[n].i, nodes[n].j, nodes[n].k, x, y, z);
        fprintf(outFile,"%f %f %f\n",x,y,z);
    }

    num_of_leaf_cells = number_of_leaves();

    fprintf(outFile,"CELLS %d %d \n",num_of_leaf_cells,9*num_of_leaf_cells);
    for (int n=0; n<cells.size(); n++)
    {
        //cout<<n<<"\t"<<cells[n].level<<"\t"<<(cells[n].is_leaf ? 1:0)<<endl;
        if ( cells[n].is_leaf)
        {
            for (int k=0; k<8; k++)
                node_of_cell[k] = cells[n].nodes[k];

            fprintf(outFile,"%d %d %d %d %d %d %d %d %d\n",8,node_of_cell[0], node_of_cell[1], node_of_cell[2], node_of_cell[3],node_of_cell[4], node_of_cell[5], node_of_cell[6], node_of_cell[7]);
        }
    }
    fprintf(outFile,"CELL_TYPES %d \n",num_of_leaf_cells);
    for (int n=0; n<num_of_leaf_cells; n++)
        fprintf(outFile,"%d \n",12);
    //fprintf(outFile,"POINT_DATA %d \n",nodes.size());
    fprintf(outFile,"CELL_DATA %d \n",num_of_leaf_cells);
    fclose (outFile);
}



void AtomTree::print_VTK_format(std::vector<double> &F, std::string data_name, std::string file_name ) const
{
    int num_of_nodes;
    num_of_nodes = this -> number_of_nodes();
    FILE *outFile;
    outFile = fopen(file_name.c_str(),"a");
    fprintf(outFile,"SCALARS %s double 1 \n",data_name.c_str());
    fprintf(outFile,"LOOKUP_TABLE default \n");
    for (int n=0; n<number_of_leaves(); n++) fprintf(outFile,"%E \n",F[n]);
    fclose (outFile);
}

void AtomTree::print_VTK_format(std::vector<int> &F, std::string data_name, std::string file_name ) const
{
    int num_of_nodes;
    num_of_nodes = this -> number_of_nodes();
    FILE *outFile;
    outFile = fopen(file_name.c_str(),"a");
    fprintf(outFile,"SCALARS %s double 1 \n",data_name.c_str());
    fprintf(outFile,"LOOKUP_TABLE default \n");
    for (int n=0; n<number_of_leaves(); n++) fprintf(outFile,"%d \n",F[n]);
    fclose (outFile);
}

