#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <linux/limits.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <float.h>
#include <sstream>

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
  const string double_separator = "  |"; base_format += double_separator;
  string format = base_format + read_double;
  double tmp;

  coordinates.resize(0);
  while (sscanf(read_line, format.c_str(), &tmp) > 0)
  {
    coordinates.push_back(tmp);
    base_format += disregard_double + double_separator;
    format = base_format + read_double;
  }
}

void read_my_line(const char* read_line, double& time, vector<double>& slice_avg_profile)
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
    if(idx == slice_avg_profile.size())
      throw runtime_error("read_line: something went wrong when reading a line: more values than expected...");
    slice_avg_profile[idx++] = tmp;
    format = base_format + read_double;
  }
  if(idx<slice_avg_profile.size())
    throw runtime_error("read_line: something went wrong when reading a line: less values than expected...");
}

int main (int argc, char* argv[])
{
  double t_start, t_end, time_read_n, time_read_nm1;
  double t_integration_start = DBL_MAX;
  double t_integration_end = -DBL_MAX;
  double dt_nm1 = 0.0;
  bool integrate = false;
  size_t len = 0;
  ssize_t len_read;
  char* read_line = NULL;
  vector<double> coordinates;
  vector<double> time_avg_profile, slice_avg_n, slice_avg_nm1;
  char filename[PATH_MAX];
  FILE* fp;
  if (argc<2)
    throw runtime_error("This function needs at least one argument: the path to the file containing the velocity averages to be integrated in time.");
  char* path_to_file = argv[1];
  if(!file_exists(path_to_file))
    throw invalid_argument("Can't find the file.");
  if(argc>=3)
    sscanf(argv[2], "%lg", &t_start);
  else
    t_start = -DBL_MAX;
  if(argc>=4)
    sscanf(argv[3], "%lg", &t_end);
  else
    t_end = +DBL_MAX;

  fp = fopen(path_to_file, "r");
  if(fp == NULL)
    throw invalid_argument("Can't open the file.");

  if(((len_read = getline(&read_line, &len, fp)) == -1))
    throw runtime_error("Couldn't read the first header line of the file.");
  if(((len_read = getline(&read_line, &len, fp)) == -1))
    throw runtime_error("Couldn't read the second header line of the file.");

  read_second_header_line(read_line, coordinates);
  time_avg_profile.resize(coordinates.size());
  slice_avg_n.resize(coordinates.size());
  for (size_t k = 0; k < time_avg_profile.size(); k++)
    time_avg_profile[k] = 0.0;

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
          time_avg_profile[k] += 0.5*dt_nm1*(slice_avg_nm1[k]+slice_avg_n[k]);
      integrate = true;
    }
    if(time_read_n > t_end)
      break;
    time_read_nm1 = time_read_n;
    slice_avg_nm1 = slice_avg_n;
  }
  fclose(fp);
  if(read_line)
    free(read_line);

  for (size_t k = 0; k < time_avg_profile.size(); k++)
    time_avg_profile[k] /= (t_integration_end-t_integration_start);

  sprintf(filename, "./time_averaged_velocity_profile.dat");
  fp= fopen(filename, "w");
  if(fp==NULL)
    throw runtime_error("Could not open file for time-averaged output.");
  fprintf(fp, "%% t_integration_start: %.10g\n", t_integration_start);
  fprintf(fp, "%% t_integration_end: %.10g\n", t_integration_end);
  fprintf(fp, "%% coordinate | time average \n");
  for (size_t k = 0; k < time_avg_profile.size(); k++)
    fprintf(fp, "%.12g %.12g\n", coordinates.at(k), time_avg_profile.at(k));
  fclose(fp);

  sprintf(filename, "./tex_time_average.gnu");
  fp = fopen(filename, "w");
  if(fp==NULL)
    throw runtime_error("Could not open file for time-averaged tex figure.");
  fprintf(fp, "set term epslatex color standalone\n");
  fprintf(fp, "set output 'time_average.tex'\n");
  fprintf(fp, "set xlabel \"$<u>=\\\\frac{1}{t_{\\\\mathrm{end}} - t_{\\\\mathrm{start}}} \\\\frac{1}{L_{x}} \\\\frac{1}{L_{z}} \\\\int_{t_{\\\\mathrm{start}}}^{t_{\\\\mathrm{end}}}{\\\\int_{-L_{z}/2}^{L_{z}/2}{\\\\int_{-L_{x}/2}^{L_{x}/2}{ u \\\\,\\\\mathrm{d}x}\\\\,\\\\mathrm{d}z}\\\\,\\\\mathrm{d}t}$\"\n");
  fprintf(fp, "set ylabel \"$y$\"\n");
  fprintf(fp, "plot\t \"./time_averaged_velocity_profile.dat\" using 2:1 notitle with lines lw 3 linecolor rgb \"blue\"");
  fclose(fp);

  sprintf(filename, "./plot_tex_time_average.sh");
  fp = fopen(filename, "w");
  if(fp==NULL)
    throw runtime_error("Could not open file for bash script plotting time-averaged tex figure.");
  fprintf(fp, "#!/bin/sh\n");
  fprintf(fp, "gnuplot ./tex_time_average.gnu\n");
  fprintf(fp, "latex ./time_average.tex\n");
  fprintf(fp, "dvipdf -dAutoRotatePages=/None ./time_average.dvi\n");
  fclose(fp);

  ostringstream chmod_command;
  chmod_command << "chmod +x " << filename << endl;
  int sys_return = system(chmod_command.str().c_str()); (void) sys_return;

  return 0;
}
