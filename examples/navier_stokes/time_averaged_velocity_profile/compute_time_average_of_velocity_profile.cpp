// Source file for time-averaging utility
// ======================================
// This tool is meant to be used to time-average velocity profiles as exported by casl_p4est's navier-stokes objects
// (calculated by means of get_slice_averaged_vnp1_profile or get_line_averaged_vnp1_profiles, see
// casl_p4est/examples/navier_stokes/main_shs_2d.cpp for example usage)
//
// The names of the space-averaged velocity files as exported by casl_p4est must satisfy the only following requirements
// 1) contain 'velocity_profile' somewhere in their name(s);
// 2) have a '.dat' extension;
//
// To compile (basic compilation), type in terminal:
// mpicxx path_to_this_source_file -o path_to_desired_build_directory/time_average
//
// To use after compilation, use terminal (in the build directory):
// - if only one space-averaged profile needs to be time-averaged (example: slice-averaged profiles, or one single
//   line-averaged profile), type
//     ./time_average path_to_exported_space_averaged_files/full_name_of_the_space_averaged_file_with_the_.dat_extension t_start t_end
//   to calculate the time average of the exported velocity profile between t_start and t_end
//   (one can use mpirun -np ### but only the root will work in that case, sorry)
// - if several space-averaged profiles need to be time-averaged (example: several line-averaged profiles), each of these files MUST be
//   named like "whatever_you_want_really#.dat" where
//   1) whatever_you_want_really just needs to contain 'velocity_profile' somehow
//   2) # is an index starting from 0 (0 MUST be there) to a maximum value (without gap).
//   One can then use several CPUs, say N, each working on a different file by typing
//      mpirun -np N ./time_average path_to_exported_space_averaged_files/whatever_you_want_really t_start t_end
//   (note that '#.dat' is NOT provided as an input to the program --> the program knows that it has to find a succession of such files
//   if the provided file name does not contain the '.dat' extension, so it just needs to be given
//   path_to_exported_space_averaged_files/whatever_you_want_really --> this is what will actually be the easiest to type in terminal by
//   using auto-completion, i.e. the tab key)
//
// This utility calculates the desired time-averaged velocity profiles based on the raw exportations and exports the results in similar files
// along with gnuplot automatized files and bash scripts.
// When used on a single file, execute the bash_script in the relevant directory to generate a pdf file illustrating the time-averaged profiles.
// When used on several files, a similar bash script executes a similar task but it can also take in an integer argument to state how many of
// the total number of profiles are desired to be exported...
//
// Developer: Raphael Egan, CASL, April 2019

#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <linux/limits.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <float.h>
#include <sstream>
#include <mpi.h>

using namespace std;

bool file_exists(const char* path)
{
  struct stat info;
  return ((stat(path, &info)== 0) && (info.st_mode & S_IFREG));
}

const string disregard_double = " %*lg";
const string read_double = " %lg";

void read_second_header_line(const char* read_line, vector<double>& coordinates)
{
  string base_format = "%% tn";
  const string double_separator = "  |";
  base_format += double_separator;
  double tmp;

  coordinates.resize(0);
  while (sscanf(read_line, (base_format+read_double).c_str(), &tmp) > 0)
  {
    coordinates.push_back(tmp);
    base_format += disregard_double + double_separator;
  }
}

void read_my_line(const char* read_line, double& time, vector<double> *slice_avg_profile)
{
  string base_format = "%*lg";
  string format = "%lg";
  size_t idx = 0;
  double tmp;

  sscanf(read_line, format.c_str(), &time);
  format = base_format + read_double;
  while (sscanf(read_line, format.c_str(), &tmp) > 0)
  {
    base_format += disregard_double;
    if(idx == slice_avg_profile->size())
      throw runtime_error("read_line: something went wrong when reading a line: more values than expected...");
    slice_avg_profile->at(idx++) = tmp;
    format = base_format + read_double;
  }
  if(idx<slice_avg_profile->size())
    throw runtime_error("read_line: something went wrong when reading a line: less values than expected...");
}

int main (int argc, char* argv[])
{
  int mpirank, mpisize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  double t_start, t_end, time_read_n;
  double time_read_nm1;
  double t_integration_start;
  double t_integration_end;
  double dt_nm1 = 0.0;
  bool integrate;
  size_t len = 0;
  ssize_t len_read;
  char* read_line = NULL;
  vector<double> coordinates;
  vector<double> time_avg_profile;
  vector<double> *slice_avg_n, *slice_avg_nm1, *tmp;
  char filename[PATH_MAX];
  FILE* fp;
  if (argc<2)
    throw runtime_error("This function needs at least one argument: the path to the file containing the velocity averages to be integrated in time.");
  char* path_to_file = argv[1];
  const char extension[5] = ".dat";
  const char end_raw_name[17] = "velocity_profile";
  size_t length_path_to_folder = 0;
  size_t full_length = 0;
  while (*(path_to_file+full_length)!='\0') {
    if(path_to_file[full_length] == '/')
      length_path_to_folder = full_length+1;
    full_length++;
  }
  const bool only_one_file = (full_length>4) && (strcmp(path_to_file+full_length-4, extension) == 0);
  char path_to_folder[length_path_to_folder+1];
  for (unsigned int kk = 0; kk < length_path_to_folder; ++kk)
    *(path_to_folder+kk)=*(path_to_file+kk);
  *(path_to_folder+length_path_to_folder) = '\0';
  size_t length_rawname = 0;
  while (strncmp(end_raw_name, path_to_file+length_path_to_folder+length_rawname, 16)!=0)
    length_rawname++;
  char raw_file_name[NAME_MAX];
  for (unsigned int kk = 0; kk < length_rawname; ++kk)
    *(raw_file_name+kk) = *(path_to_file+length_path_to_folder+kk);
  for (unsigned int kk = 0; kk < 17; ++kk)
    *(raw_file_name+length_rawname+kk) = *(end_raw_name+kk);

  char file_check[PATH_MAX];
  if(only_one_file)
    sprintf(file_check, "%s", path_to_file);
  else
    sprintf(file_check, "%s%d%s", path_to_file, 0, extension); // check first one if several are expected (this one MUST be there, no matter what)
  if(mpirank==0 && !file_exists(file_check))
    throw invalid_argument("Can't find the file.");
  if(argc>=3)
    sscanf(argv[2], "%lg", &t_start);
  else
    t_start = -DBL_MAX;
  if(argc>=4)
    sscanf(argv[3], "%lg", &t_end);
  else
    t_end = +DBL_MAX;

  unsigned int idx = mpirank;
  char file_to_open[PATH_MAX];
  if(only_one_file)
    sprintf(file_to_open, "%s", path_to_file);
  else
    sprintf(file_to_open, "%s%d%s", path_to_file, idx, extension);
  while (file_exists(file_to_open) && (!only_one_file || mpirank == 0)) {
    fp = fopen(file_to_open, "r");
    if(fp == NULL)
      throw invalid_argument("Can't open the file.");

    if((len_read = getline(&read_line, &len, fp)) == -1)
      throw runtime_error("Couldn't read the first header line of the file.");
    if((len_read = getline(&read_line, &len, fp)) == -1)
      throw runtime_error("Couldn't read the second header line of the file.");

    // start reading relevant data
    read_second_header_line(read_line, coordinates);
    time_avg_profile.resize(coordinates.size());
    for (size_t k = 0; k < time_avg_profile.size(); k++)
      time_avg_profile[k] = 0.0;
    integrate           = false;
    slice_avg_n         = new vector<double>(coordinates.size(), 0.0);
    slice_avg_nm1       = new vector<double>(coordinates.size(), 0.0);
    time_read_nm1       = DBL_MAX;
    t_integration_start = DBL_MAX;
    t_integration_end   = -DBL_MAX;
    while ((len_read = getline(&read_line, &len, fp)) != -1) {
      read_my_line(read_line, time_read_n, slice_avg_n);
      if((time_read_n > t_start) && (time_read_n < t_end))
      {
        if(time_read_n < t_integration_start)
          t_integration_start = time_read_n;
        if(time_read_n > t_integration_end)
          t_integration_end = time_read_n;
        dt_nm1 = time_read_n - time_read_nm1;
        if(integrate)
          for (size_t k = 0; k < time_avg_profile.size(); k++)
            time_avg_profile[k] += 0.5*dt_nm1*(slice_avg_nm1->at(k)+slice_avg_n->at(k));
        integrate = true;
      }
      if(time_read_n > t_end)
        break;
      time_read_nm1 = time_read_n;
      tmp           = slice_avg_nm1;
      slice_avg_nm1 = slice_avg_n;
      slice_avg_n   = tmp;
    }
    fclose(fp);
    delete slice_avg_n;
    delete slice_avg_nm1;
    // done reading relevant data

    for (size_t k = 0; k < time_avg_profile.size(); k++)
      time_avg_profile[k] /= (t_integration_end-t_integration_start);

    if(only_one_file)
      sprintf(filename, "%stime_averaged_%s%s", path_to_folder, raw_file_name, extension);
    else
      sprintf(filename, "%stime_averaged_%s_index_%d%s", path_to_folder, raw_file_name, idx, extension);
    fp= fopen(filename, "w");
    if(fp==NULL)
      throw runtime_error("Could not open file for time-averaged output.");
    fprintf(fp, "%% t_integration_start: %.10g\n", t_integration_start);
    fprintf(fp, "%% t_integration_end: %.10g\n", t_integration_end);
    fprintf(fp, "%% coordinate | time average \n");
    for (size_t k = 0; k < time_avg_profile.size(); k++)
      fprintf(fp, "%.12g %.12g\n", coordinates.at(k), time_avg_profile.at(k));
    fclose(fp);

    if(only_one_file)
      break;
    else
    {
      idx += mpisize;
      sprintf(file_to_open, "%s%d%s", path_to_file, idx, extension);
    }
  }
  if(read_line)
    free(read_line);

  MPI_Barrier(MPI_COMM_WORLD);

  if(mpirank == 0)
  {
    sprintf(filename, "%stex_time_average_%s.gnu", path_to_folder, raw_file_name);
    fp = fopen(filename, "w");
    if(fp==NULL)
      throw runtime_error("Could not open file for time-averaged tex figure.");
    fprintf(fp, "set term epslatex color standalone\n");
    fprintf(fp, "set output 'time_average_%s.tex'\n", raw_file_name);
    fprintf(fp, "set xlabel \"$<u>$\"\n"); // =\\\\frac{1}{t_{\\\\mathrm{end}} - t_{\\\\mathrm{start}}} \\\\frac{1}{L_{x}} \\\\frac{1}{L_{z}} \\\\int_{t_{\\\\mathrm{start}}}^{t_{\\\\mathrm{end}}}{\\\\int_{-L_{z}/2}^{L_{z}/2}{\\\\int_{-L_{x}/2}^{L_{x}/2}{ u \\\\,\\\\mathrm{d}x}\\\\,\\\\mathrm{d}z}\\\\,\\\\mathrm{d}t}$\"\n");
    fprintf(fp, "set ylabel \"$y$\"\n");
    if(only_one_file)
      fprintf(fp, "plot\t \"./time_averaged_%s%s\" using 2:1 notitle with linespoints lc rgb \"blue\" dashtype 5 lw 2 pt 7 pi -0.5 ps 0.5", raw_file_name, extension);
    else
    {
      unsigned int last_idx=0;
      sprintf(file_check, "%s%d%s", path_to_file, last_idx, extension);
      while (file_exists(file_check))
      {
        last_idx++;
        sprintf(file_check, "%s%d%s", path_to_file, last_idx, extension);
      }
      fprintf(fp, "ntot=%u\n", last_idx);
      fprintf(fp, "if (!exists(\"nprofiles\")) nprofiles=ntot\n");
      fprintf(fp, "stepping=(ntot+1)/nprofiles\n");
      fprintf(fp, "plot for [i=0:ntot:stepping] \"./time_averaged_line_averaged_velocity_profile_index_\".i.\".dat\" using 2:1 title 'index '.i");
    }
    fclose(fp);

    sprintf(filename, "%splot_tex_time_average_%s.sh", path_to_folder, raw_file_name);
    fp = fopen(filename, "w");
    if(fp==NULL)
      throw runtime_error("Could not open file for bash script plotting time-averaged tex figure.");
    fprintf(fp, "#!/bin/sh\n");
    if(only_one_file)
      fprintf(fp, "gnuplot ./tex_time_average_%s.gnu\n", raw_file_name);
    else
    {
      fprintf(fp, "if [ \"$1\" != \"\" ]; then\n");
      fprintf(fp, "\t gnuplot -e \"nprofiles=$1\" ./tex_time_average_line_averaged_velocity_profile.gnu\n");
      fprintf(fp, "else\n");
      fprintf(fp, "\t gnuplot ./tex_time_average_line_averaged_velocity_profile.gnu\n");
      fprintf(fp, "fi\n");
    }
    fprintf(fp, "latex ./time_average_%s.tex\n", raw_file_name);
    fprintf(fp, "dvipdf -dAutoRotatePages=/None ./time_average_%s.dvi\n", raw_file_name);
    fclose(fp);

    ostringstream chmod_command;
    chmod_command << "chmod +x " << filename << endl;
    int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
  }

  MPI_Finalize();

  return 0;
}
