#include "simplex2_quad_mls_vtk.h"

using namespace std;

void simplex2_quad_mls_vtk::write_simplex_geometry(vector<simplex2_quad_mls_t *>& simplices, string dir, string suffix)
{
  vector<int> n_vtxs, n_vtxs_shift;
  vector<int> n_edgs, n_tris;
  simplex2_quad_mls_t *s;
  int n_vtxs_tot = 0;
  int n_edgs_tot = 0;
  int n_tris_tot = 0;

  int n_s = simplices.size();

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];

    n_vtxs.push_back(s->vtxs.size());
    n_vtxs_shift.push_back(n_vtxs_tot);
    n_vtxs_tot += n_vtxs.back();

    n_edgs.push_back(0);  for (int j = 0; j < s->edgs.size(); j++) {if (!s->edgs[j].is_split) n_edgs.back()++;}  n_edgs_tot += n_edgs.back();
    n_tris.push_back(0);  for (int j = 0; j < s->tris.size(); j++) {if (!s->tris[j].is_split) n_tris.back()++;} n_tris_tot += n_tris.back();
  }

  ofstream ofs;

  /* write vertices */
  string vtxs_vtu = dir + "vtxs_2d_" + suffix + ".vtu";

  ofs.open(vtxs_vtu.c_str());

  ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl
      << "<UnstructuredGrid>" << endl
      << "<Piece NumberOfPoints=\""
      << n_vtxs_tot
      << "\" NumberOfCells=\""
      << n_vtxs_tot
      << "\">" << endl
      << "<PointData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"location\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << (int) s->vtxs[j].loc << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</PointData>" << endl
      << "<CellData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"location\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << (int) s->vtxs[j].loc << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</CellData>" << endl
      << "<Points>" << endl
      << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << s->vtxs[j].x << " "
          << s->vtxs[j].y << " "
          << 0 << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</Points>" << endl
      << "<Cells>" << endl
      << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_vtxs_tot; i++)
  {
    ofs << i << endl;
  }

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_vtxs_tot; i++)
  {
    ofs << (i+1) << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_vtxs_tot; i++)
  {
    ofs << 1 << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "</Cells>" << endl
      << "</Piece>" << endl
      << "</UnstructuredGrid>" << endl
      << "</VTKFile>" << endl;

  ofs.close();

  /* write edges */
  string edgs_vtu = dir + "edgs_2d_" + suffix +".vtu";
  ofs.open(edgs_vtu.c_str());

  ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl
      << "<UnstructuredGrid>" << endl
      << "<Piece NumberOfPoints=\""
      << n_vtxs_tot
      << "\" NumberOfCells=\""
      << n_edgs_tot
      << "\">" << endl
      << "<PointData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"location\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << (int) s->vtxs[j].loc << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</PointData>" << endl
      << "<CellData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"location\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < s->edgs.size(); j++)
    {
      if (!s->edgs[j].is_split)
      {
        ofs << (int) s->edgs[j].loc << " ";
      }
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl;
  ofs << "<DataArray type=\"Float32\" Name=\"c0\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < s->edgs.size(); j++)
    {
      if (!s->edgs[j].is_split)
      {
        ofs << s->edgs[j].c0 << " ";
      }
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</CellData>" << endl
      << "<Points>" << endl
      << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << s->vtxs[j].x << " "
          << s->vtxs[j].y << " "
          << 0 << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</Points>" << endl
      << "<Cells>" << endl
      << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < s->edgs.size(); j++)
    {
      if (!s->edgs[j].is_split)
      {
        ofs << n_vtxs_shift[i] + s->edgs[j].vtx0 << " "
            << n_vtxs_shift[i] + s->edgs[j].vtx1 << " ";
      }
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_edgs_tot; i++)
  {
    ofs << 2*(i+1) << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_edgs_tot; i++)
  {
    ofs << 3 << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "</Cells>" << endl
      << "</Piece>" << endl
      << "</UnstructuredGrid>" << endl
      << "</VTKFile>" << endl;

  ofs.close();

  /* write triangles */
  string tris_vtu = dir + "tris_2d_" + suffix +".vtu";
  ofs.open(tris_vtu.c_str());

  ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl
      << "<UnstructuredGrid>" << endl
      << "<Piece NumberOfPoints=\""
      << n_vtxs_tot
      << "\" NumberOfCells=\""
      << n_tris_tot
      << "\">" << endl
      << "<PointData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"scalars\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << (int) s->vtxs[j].loc << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</PointData>" << endl
      << "<CellData Scalars=\"scalars\">" << endl
      << "<DataArray type=\"Float32\" Name=\"scalars\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < s->tris.size(); j++)
    {
      if (!s->tris[j].is_split)
      {
        ofs << (int)s->tris[j].loc << " ";
      }
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</CellData>" << endl
      << "<Points>" << endl
      << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < n_vtxs[i]; j++)
    {
      ofs << s->vtxs[j].x << " "
          << s->vtxs[j].y << " "
          << 0 << " ";
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "</Points>" << endl
      << "<Cells>" << endl
      << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_s; i++)
  {
    s = simplices[i];
    for (int j = 0; j < s->tris.size(); j++)
    {
      if (!s->tris[j].is_split)
      {
        ofs << n_vtxs_shift[i] + s->tris[j].vtx0 << " "
            << n_vtxs_shift[i] + s->tris[j].vtx1 << " "
            << n_vtxs_shift[i] + s->tris[j].vtx2 << " ";
      }
    }
    ofs << endl;
  }

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_tris_tot; i++)
  {
    ofs << 3*(i+1) << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << endl;

  for (int i = 0; i < n_tris_tot; i++)
  {
    ofs << 5 << " ";
  }
  ofs << endl;

  ofs << "</DataArray>" << endl
      << "</Cells>" << endl
      << "</Piece>" << endl
      << "</UnstructuredGrid>" << endl
      << "</VTKFile>" << endl;

  ofs.close();

}
